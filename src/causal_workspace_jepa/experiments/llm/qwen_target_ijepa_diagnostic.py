"""Retrospective diagnostics for the frozen target-encoder Intervention-JEPA.

This module deliberately performs oracle analyses of already observed outcomes. It can diagnose
geometry and reuse failures, but it cannot confirm a preregistered hypothesis or establish causal
fidelity for a learned predictor.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.activation_store import read_hdf5_shards
from causal_workspace_jepa.models.intervention_jepa import effect_correlation
from causal_workspace_jepa.models.target_encoder_intervention_jepa import (
    StandardizedRidgeDecoder,
    TargetEncoderInterventionJEPA,
)


SPLIT_NAMES = ("train", "validation", "test")


class _TargetLatentModel(Protocol):
    def target_latent(self, values: np.ndarray) -> np.ndarray: ...

    def predict_latent(
        self,
        clean_source: np.ndarray,
        donor_source: np.ndarray,
        clean_target: np.ndarray,
        source_delta: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class TrainOnlyPCA:
    """Coordinate-standardized PCA whose statistics are fitted on training rows only."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, components: int) -> "TrainOnlyPCA":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("PCA training values must be a rank-two array with at least two rows")
        maximum = min(values.shape)
        if components < 1 or components > maximum:
            raise ValueError(f"PCA components must lie in [1, {maximum}]")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        standardized = (values - mean) / scale
        _left, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
        power = singular_values**2
        denominator = max(float(power.sum()), 1e-12)
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            components=right[:components].astype(np.float32),
            explained_variance_ratio=(power[:components] / denominator).astype(np.float64),
        )

    def transform(self, values: np.ndarray, components: int | None = None) -> np.ndarray:
        count = self.components.shape[0] if components is None else int(components)
        if count < 1 or count > self.components.shape[0]:
            raise ValueError("requested PCA dimension was not fitted")
        normalized = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        return (normalized @ self.components[:count].T).astype(np.float32)


def effective_rank(values: np.ndarray) -> float:
    """Entropy effective rank after centering examples."""

    centered = np.asarray(values, dtype=np.float64)
    if centered.ndim != 2 or centered.shape[0] == 0:
        raise ValueError("effective rank requires a nonempty rank-two array")
    centered = centered - centered.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = singular_values**2
    if float(power.sum()) <= 1e-12:
        return 0.0
    probability = power / power.sum()
    return float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12)))))


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Multivariate fraction of centered sum of squares explained by group means."""

    matrix = np.asarray(values, dtype=np.float64)
    groups = np.asarray(labels)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or len(groups) != matrix.shape[0]:
        raise ValueError("eta squared requires aligned, nonempty rank-two values and labels")
    mean = matrix.mean(axis=0)
    total = float(np.square(matrix - mean).sum())
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for label in np.unique(groups):
        group = matrix[groups == label]
        between += float(group.shape[0] * np.square(group.mean(axis=0) - mean).sum())
    return float(between / total)


def compute_target_ijepa_diagnostic(
    data: Mapping[str, np.ndarray],
    models: Mapping[int, _TargetLatentModel],
    *,
    pca_dimensions: Sequence[int] = (8, 16, 32, 64),
    decoder_ridge: float = 1.0,
) -> dict[str, Any]:
    """Compute frozen-model and train-only oracle diagnostics from aligned arrays."""

    required = {
        "split_id",
        "recipient_id",
        "donor_id",
        "clean_source",
        "donor_source",
        "source_delta",
        "clean_target_hidden",
        "intervened_target_hidden",
        "clean_answer_logits",
        "target_effect",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"diagnostic arrays missing required fields: {missing}")
    if not models:
        raise ValueError("at least one frozen target-encoder model is required")
    if not np.isfinite(decoder_ridge) or decoder_ridge < 0.0:
        raise ValueError("decoder_ridge must be finite and nonnegative")
    for name in required.difference({"recipient_id", "donor_id"}):
        if not np.all(np.isfinite(np.asarray(data[name]))):
            raise FloatingPointError(f"diagnostic input contains nonfinite values: {name}")
    split_ids = np.asarray(data["split_id"], dtype=np.int64)
    masks = {name: split_ids == index for index, name in enumerate(SPLIT_NAMES)}
    if any(not bool(mask.any()) for mask in masks.values()):
        raise ValueError("train, validation, and test splits must all be nonempty")
    dimensions = tuple(sorted({int(value) for value in pca_dimensions}))
    if not dimensions or dimensions[0] < 1:
        raise ValueError("at least one positive PCA dimension is required")

    clean_source = np.asarray(data["clean_source"], dtype=np.float32)
    donor_source = np.asarray(data["donor_source"], dtype=np.float32)
    source_delta = np.asarray(data["source_delta"], dtype=np.float32)
    clean_target = np.asarray(data["clean_target_hidden"], dtype=np.float32)
    intervened_target = np.asarray(data["intervened_target_hidden"], dtype=np.float32)
    causal_delta = intervened_target - clean_target
    target_effect = np.asarray(data["target_effect"], dtype=np.float32)
    answer_count = int(np.asarray(data["clean_answer_logits"]).shape[1])
    train = masks["train"]

    rank_rows: dict[str, Any] = {}
    identity_rows: dict[str, Any] = {}
    for seed, model in sorted(models.items()):
        rank_rows[str(seed)] = {}
        identity_rows[str(seed)] = {}
        for split_name, mask in masks.items():
            target_latent = model.target_latent(intervened_target[mask])
            clean_latent = model.target_latent(clean_target[mask])
            predicted_latent = model.predict_latent(
                clean_source[mask], donor_source[mask], clean_target[mask], source_delta[mask]
            )
            rank_rows[str(seed)][split_name] = {
                "examples": int(mask.sum()),
                "unique_recipients": int(np.unique(data["recipient_id"][mask]).size),
                "unique_donors": int(np.unique(data["donor_id"][mask]).size),
                "target_effective_rank": effective_rank(target_latent),
                "clean_target_effective_rank": effective_rank(clean_latent),
                "predicted_effective_rank": effective_rank(predicted_latent),
                "target_mean_dimension_std": float(target_latent.std(axis=0).mean()),
                "predicted_mean_dimension_std": float(predicted_latent.std(axis=0).mean()),
            }
            identity_rows[str(seed)][split_name] = {
                "target_latent_donor_eta_squared": eta_squared(
                    target_latent, data["donor_id"][mask]
                ),
                "target_latent_recipient_eta_squared": eta_squared(
                    target_latent, data["recipient_id"][mask]
                ),
                "latent_delta_donor_eta_squared": eta_squared(
                    target_latent - clean_latent, data["donor_id"][mask]
                ),
                "latent_delta_recipient_eta_squared": eta_squared(
                    target_latent - clean_latent, data["recipient_id"][mask]
                ),
            }

    raw_identity = {
        split_name: {
            "clean_target_donor_eta_squared": eta_squared(
                clean_target[mask], data["donor_id"][mask]
            ),
            "clean_target_recipient_eta_squared": eta_squared(
                clean_target[mask], data["recipient_id"][mask]
            ),
            "intervened_target_donor_eta_squared": eta_squared(
                intervened_target[mask], data["donor_id"][mask]
            ),
            "intervened_target_recipient_eta_squared": eta_squared(
                intervened_target[mask], data["recipient_id"][mask]
            ),
            "causal_delta_donor_eta_squared": eta_squared(
                causal_delta[mask], data["donor_id"][mask]
            ),
            "causal_delta_recipient_eta_squared": eta_squared(
                causal_delta[mask], data["recipient_id"][mask]
            ),
        }
        for split_name, mask in masks.items()
    }

    maximum_dimension = max(dimensions)
    state_projector = TrainOnlyPCA.fit(
        np.concatenate([clean_target[train], intervened_target[train]], axis=0),
        maximum_dimension,
    )
    delta_projector = TrainOnlyPCA.fit(causal_delta[train], maximum_dimension)
    pca_rows: dict[str, Any] = {}
    for dimension in dimensions:
        state_input = np.concatenate(
            [
                state_projector.transform(intervened_target, dimension),
                state_projector.transform(clean_target, dimension),
            ],
            axis=1,
        )
        delta_input = delta_projector.transform(causal_delta, dimension)
        state_decoder = StandardizedRidgeDecoder.fit(
            state_input[train], target_effect[train], ridge=decoder_ridge
        )
        delta_decoder = StandardizedRidgeDecoder.fit(
            delta_input[train], target_effect[train], ridge=decoder_ridge
        )
        pca_rows[str(dimension)] = {
            "state_pair_cumulative_explained_variance": float(
                state_projector.explained_variance_ratio[:dimension].sum()
            ),
            "causal_delta_cumulative_explained_variance": float(
                delta_projector.explained_variance_ratio[:dimension].sum()
            ),
            "state_pair": {
                name: _effect_scores(
                    state_decoder.predict(state_input[mask]), target_effect[mask], answer_count
                )
                for name, mask in masks.items()
            },
            "causal_delta": {
                name: _effect_scores(
                    delta_decoder.predict(delta_input[mask]), target_effect[mask], answer_count
                )
                for name, mask in masks.items()
            },
        }

    return {
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "pca_fit_split": "train",
        "pca_fit_examples": {
            "state_pair_states": int(2 * train.sum()),
            "causal_delta": int(train.sum()),
        },
        "effective_rank_by_seed_and_split": rank_rows,
        "raw_identity_eta_squared_by_split": raw_identity,
        "latent_identity_eta_squared_by_seed_and_split": identity_rows,
        "oracle_pca_ridge": pca_rows,
    }


def run_qwen_target_ijepa_diagnostic(config_path: str | Path) -> dict[str, Any]:
    """Run the explicitly post-hoc diagnostic and persist metrics plus provenance."""

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    hardware = require_free_disk(resource_profile)
    dataset_dir = Path(str(config["dataset_dir"]))
    registered_manifest = json.loads(
        Path(str(config["dataset_manifest"])).read_text(encoding="utf-8")
    )
    local_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_dataset_id = str(config.get("dataset_id", "LLM-CAPITAL-PATCH-001"))
    if registered_manifest.get("dataset_id") != expected_dataset_id:
        raise RuntimeError("registered dataset manifest has the wrong dataset identity")
    if local_manifest.get("dataset_id") != expected_dataset_id:
        raise RuntimeError("local activation shard has the wrong dataset identity")
    if registered_manifest.get("config_digest") != local_manifest.get("config_digest"):
        raise RuntimeError("registered and local dataset config digests differ")
    registered_shards = {
        (Path(str(row["path"])).name, str(row["sha256"]))
        for row in registered_manifest.get("shards", [])
    }
    local_shards = {
        (Path(str(row["path"])).name, str(row["sha256"]))
        for row in local_manifest.get("shards", [])
    }
    if registered_shards != local_shards:
        raise RuntimeError("registered and local activation shard checksums differ")
    data, records = read_hdf5_shards(dataset_dir)

    checkpoint_dir = Path(str(config["checkpoint_dir"]))
    seeds = tuple(int(value) for value in config.get("seeds", [311, 313, 317]))
    models: dict[int, TargetEncoderInterventionJEPA] = {}
    checkpoint_provenance: dict[str, Any] = {}
    for seed in seeds:
        checkpoint = checkpoint_dir / f"seed-{seed}.npz"
        if not checkpoint.is_file():
            raise RuntimeError(f"missing frozen target-encoder checkpoint: {checkpoint}")
        models[seed] = TargetEncoderInterventionJEPA.load(checkpoint)
        checkpoint_provenance[str(seed)] = {
            "path": checkpoint.as_posix(),
            "sha256": _sha256(checkpoint),
            "best_step": int(models[seed].training_metrics["best_step"]),
        }

    analysis = compute_target_ijepa_diagnostic(
        data,
        models,
        pca_dimensions=tuple(int(value) for value in config.get("pca_dimensions", [8, 16, 32, 64])),
        decoder_ridge=float(config.get("decoder_ridge", 1.0)),
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seeds[0] if seeds else None,
    )
    if provenance.git_dirty:
        raise RuntimeError(
            "repository must be clean before writing the post-hoc diagnostic result"
        )
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-TARGET-IJEPA-DIAGNOSTIC-001")),
        "status": "POSTHOC_DIAGNOSTIC",
        "evidence_level": "Availability",
        "dataset_id": expected_dataset_id,
        "dataset_examples": len(records),
        "dataset_config_digest": registered_manifest["config_digest"],
        "dataset_shards": [
            {"path": name, "sha256": digest} for name, digest in sorted(registered_shards)
        ],
        "checkpoint_provenance": checkpoint_provenance,
        "pca_dimensions": list(config.get("pca_dimensions", [8, 16, 32, 64])),
        "decoder_ridge": float(config.get("decoder_ridge", 1.0)),
        **analysis,
        "hypotheses_confirmed": [],
        "confirmatory_claims_allowed": False,
        "test_outcomes_used_posthoc": True,
        "scientific_boundary": (
            "This retrospective oracle diagnostic reuses already observed test outcomes. It can "
            "identify candidate representation, target, and capacity confounds, but confirms no "
            "hypothesis and provides no new learned-predictor or causal-mechanism evidence."
        ),
        "hardware": hardware.as_dict(),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_target_ijepa_diagnostic_v1.json"))
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output_metrics.as_posix(),
            "status": metrics["status"],
            "evidence_level": metrics["evidence_level"],
            "hypotheses_confirmed": [],
        },
    )
    return metrics


def _effect_scores(
    predicted: np.ndarray, observed: np.ndarray, answer_count: int
) -> dict[str, float]:
    power = max(float(np.mean(observed**2)), 1e-12)
    hidden_power = max(float(np.mean(observed[:, :-answer_count] ** 2)), 1e-12)
    logit_power = max(float(np.mean(observed[:, -answer_count:] ** 2)), 1e-12)
    return {
        "mse": float(np.mean((predicted - observed) ** 2)),
        "normalized_mse": float(np.mean((predicted - observed) ** 2) / power),
        "hidden_normalized_mse": float(
            np.mean((predicted[:, :-answer_count] - observed[:, :-answer_count]) ** 2)
            / hidden_power
        ),
        "logit_normalized_mse": float(
            np.mean((predicted[:, -answer_count:] - observed[:, -answer_count:]) ** 2)
            / logit_power
        ),
        "correlation": effect_correlation(predicted, observed),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "TrainOnlyPCA",
    "compute_target_ijepa_diagnostic",
    "effective_rank",
    "eta_squared",
    "run_qwen_target_ijepa_diagnostic",
]
