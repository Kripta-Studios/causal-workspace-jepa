"""Post-hoc format diagnosis for the task-ineligible Qwen binding capture."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.activation_store import read_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_capture import (
    capture_content_digest,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    BindingEpisode,
    binding_episodes_from_config,
)


RUNS = ("clean", "donor", "treated")


def compute_binding_format_diagnostic(
    arrays: Mapping[str, np.ndarray],
    episodes: Sequence[BindingEpisode],
    value_token_ids: Sequence[int],
    *,
    value_token_by_string: Mapping[str, int],
    decode_token: Callable[[int], str],
) -> dict[str, Any]:
    """Separate full-vocabulary failure from restricted binding readout."""

    rows = tuple(episodes)
    values = np.asarray(value_token_ids, dtype=np.int64)
    if not rows or values.ndim != 1 or values.size == 0 or len(set(values.tolist())) != values.size:
        raise ValueError("diagnostic requires episodes and unique registered value token IDs")
    required = {
        *(f"{run}_top_token" for run in RUNS),
        *(f"{run}_value_logits" for run in RUNS),
        "recipient_answer_id",
        "donor_answer_id",
    }
    if not required <= set(arrays):
        raise ValueError(f"format diagnostic arrays missing {sorted(required.difference(arrays))}")
    if any(np.asarray(arrays[name]).shape[0] != len(rows) for name in required):
        raise ValueError("format diagnostic arrays and episodes are misaligned")
    if any(not np.all(np.isfinite(np.asarray(arrays[name]))) for name in required):
        raise FloatingPointError("format diagnostic inputs contain nonfinite values")
    if any(
        np.asarray(arrays[f"{run}_value_logits"]).shape != (len(rows), values.size)
        for run in RUNS
    ):
        raise ValueError("format diagnostic value-logit widths differ from the value roster")
    lookup = {int(token): index for index, token in enumerate(values)}
    value_tokens = {str(key): int(value) for key, value in value_token_by_string.items()}
    required_values = {
        value for episode in rows for value in episode.recipient_values
    }
    if set(value_tokens) != required_values or set(value_tokens.values()) != set(values.tolist()):
        raise RuntimeError("value-token mapping differs from the registered value roster")
    by_split: dict[str, Any] = {}
    for split in dict.fromkeys(episode.split for episode in rows):
        indices = np.asarray(
            [index for index, episode in enumerate(rows) if episode.split == split],
            dtype=np.int64,
        )
        by_split[split] = {
            run: _summarize_run(
                arrays, rows, indices, run, values, lookup, value_tokens, decode_token
            )
            for run in RUNS
        }

    test_indices = [index for index, episode in enumerate(rows) if episode.split == "test"]
    paraphrase_indices = [
        index for index, episode in enumerate(rows) if episode.split == "paraphrase"
    ]
    if len(test_indices) == 0 or len(test_indices) != len(paraphrase_indices):
        raise ValueError("diagnostic requires equally sized test and paraphrase splits")
    for left, right in zip(test_indices, paraphrase_indices, strict=True):
        primary = rows[left]
        paraphrase = rows[right]
        if (
            primary.keys != paraphrase.keys
            or primary.recipient_values != paraphrase.recipient_values
            or primary.donor_values != paraphrase.donor_values
            or primary.query_index != paraphrase.query_index
            or primary.swapped_indices != paraphrase.swapped_indices
            or primary.template != "primary"
            or paraphrase.template != "paraphrase"
        ):
            raise ValueError("test/paraphrase factors are not exactly paired")
        for answer_field in ("recipient_answer_id", "donor_answer_id"):
            answer_ids = np.asarray(arrays[answer_field], dtype=np.int64)
            if int(answer_ids[left]) != int(answer_ids[right]):
                raise ValueError("test/paraphrase answer IDs are not exactly paired")

    paired = {
        run: _paired_summary(
            arrays,
            rows,
            np.asarray(test_indices, dtype=np.int64),
            np.asarray(paraphrase_indices, dtype=np.int64),
            run,
            lookup,
            value_tokens,
        )
        for run in RUNS
    }
    donor_treated_top_equal = bool(
        np.array_equal(arrays["donor_top_token"], arrays["treated_top_token"])
    )
    donor_treated_value_logit_error = float(
        np.max(
            np.abs(
                np.asarray(arrays["donor_value_logits"], dtype=np.float64)
                - np.asarray(arrays["treated_value_logits"], dtype=np.float64)
            )
        )
    )
    return {
        "by_split": by_split,
        "paired_test_to_paraphrase": paired,
        "donor_treated_top_tokens_identical": donor_treated_top_equal,
        "donor_treated_value_logits_max_abs_error": donor_treated_value_logit_error,
    }


def run_qwen_binding_format_diagnostic(config_path: str | Path) -> dict[str, Any]:
    """Run the outcome-disclosed diagnostic with clean provenance."""

    started = time.perf_counter()
    path = Path(config_path)
    config = load_config(path)
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {path.as_posix()}",
        resource_profile=str(config["resource_profile"]),
        seed=int(config["seed"]),
    )
    if provenance.git_dirty:
        raise RuntimeError("binding format diagnostic requires a clean committed worktree")
    hardware = require_free_disk(str(config["resource_profile"]))
    parent_config = load_config(config["parent_config"])
    capture_metrics = _read_json(Path(config["capture_metrics"]))
    manifest = _read_json(Path(config["capture_manifest"]))
    if capture_metrics.get("status") != "INELIGIBLE_TASK":
        raise RuntimeError("format diagnostic is restricted to the frozen ineligible capture")
    if capture_metrics.get("parent_experiment_id") != config["parent_experiment_id"]:
        raise RuntimeError("format diagnostic parent experiment mismatch")
    if manifest.get("content_sha256") != capture_metrics.get("storage", {}).get(
        "content_sha256"
    ):
        raise RuntimeError("format diagnostic manifest/metrics content mismatch")
    arrays, records = read_hdf5_shards(Path(config["capture_dir"]))
    if capture_content_digest(arrays, records) != manifest["content_sha256"]:
        raise RuntimeError("format diagnostic capture content checksum mismatch")
    episodes = binding_episodes_from_config(parent_config)
    if [record.get("example_id") for record in records] != [
        episode.episode_id for episode in episodes
    ]:
        raise RuntimeError("format diagnostic episode order differs from capture")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(parent_config["model"]),
        revision=str(parent_config["revision"]),
        local_files_only=bool(parent_config["local_files_only"]),
        trust_remote_code=False,
        token=False,
    )
    value_token_by_string: dict[str, int] = {}
    for split in ("calibration", "train", "validation", "test"):
        for value in parent_config["token_pools"]["values"][split]:
            token_ids = tokenizer.encode(f" {value}", add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(f"registered value {value!r} is no longer one token")
            value_token_by_string[str(value)] = int(token_ids[0])
    diagnostic = compute_binding_format_diagnostic(
        arrays,
        episodes,
        manifest["value_token_ids"],
        value_token_by_string=value_token_by_string,
        decode_token=lambda token: tokenizer.decode([int(token)]),
    )
    metrics = {
        "experiment_id": str(config["id"]),
        "parent_experiment_id": str(config["parent_experiment_id"]),
        "status": "POSTHOC_DIAGNOSTIC",
        "evidence_level": "Availability",
        "confirmatory_claims_allowed": False,
        "protected_outcomes_used_posthoc": True,
        "hypotheses_confirmed": [],
        "capture_status": str(capture_metrics["status"]),
        "capture_content_sha256": str(manifest["content_sha256"]),
        "episodes": len(episodes),
        **diagnostic,
        "runtime_seconds": float(time.perf_counter() - started),
        "hardware": hardware.as_dict(),
        "scientific_boundary": (
            "This outcome-disclosed analysis diagnoses prompt-format gating in an ineligible "
            "capture. It cannot rescue binding mediation v2, select a v3 prompt, decide "
            "H-LLM-15/16, localize a mediator, or establish a circuit or workspace."
        ),
    }
    output = Path(config["output_metrics"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output.as_posix(),
            "status": metrics["status"],
            "evidence_level": metrics["evidence_level"],
            "hypotheses_confirmed": [],
        },
    )
    return metrics


def _summarize_run(
    arrays: Mapping[str, np.ndarray],
    episodes: Sequence[BindingEpisode],
    indices: np.ndarray,
    run: str,
    value_ids: np.ndarray,
    lookup: Mapping[int, int],
    value_tokens: Mapping[str, int],
    decode_token: Callable[[int], str],
) -> dict[str, Any]:
    expected = np.asarray(
        arrays["recipient_answer_id" if run == "clean" else "donor_answer_id"]
    )[indices].astype(np.int64)
    top = np.asarray(arrays[f"{run}_top_token"])[indices].astype(np.int64)
    logits = np.asarray(arrays[f"{run}_value_logits"], dtype=np.float64)[indices]
    registered_prediction = value_ids[np.argmax(logits, axis=1)]
    episode_correct: list[bool] = []
    ranks: list[int] = []
    margins: list[float] = []
    for local, row_index in enumerate(indices):
        episode_ids = tuple(
            int(value_tokens[value]) for value in episodes[int(row_index)].recipient_values
        )
        if len(set(episode_ids)) != 4 or not set(episode_ids) <= set(lookup):
            raise RuntimeError("episode value tokens differ from the registered value roster")
        scores = np.asarray([logits[local, lookup[token]] for token in episode_ids])
        order = np.argsort(-scores)
        target = int(expected[local])
        if target not in episode_ids:
            raise RuntimeError("recorded answer token is absent from the episode value set")
        rank = 1 + int(np.flatnonzero(np.asarray(episode_ids)[order] == target)[0])
        best_other = max(
            logits[local, lookup[token]] for token in episode_ids if token != target
        )
        ranks.append(rank)
        margins.append(float(logits[local, lookup[target]] - best_other))
        episode_correct.append(episode_ids[int(np.argmax(scores))] == target)
    counts = Counter(int(value) for value in top)
    dominant_id, dominant_count = counts.most_common(1)[0]
    return {
        "episodes": int(indices.size),
        "full_vocabulary_accuracy": float(np.mean(top == expected)),
        "registered_value_accuracy": float(np.mean(registered_prediction == expected)),
        "episode_four_value_accuracy": float(np.mean(episode_correct)),
        "mean_episode_answer_rank": float(np.mean(ranks)),
        "mean_episode_answer_margin": float(np.mean(margins)),
        "positive_episode_answer_margin_fraction": float(np.mean(np.asarray(margins) > 0.0)),
        "unique_top_tokens": len(counts),
        "dominant_top_token_id": dominant_id,
        "dominant_top_token_decoded": decode_token(dominant_id),
        "dominant_top_token_fraction": dominant_count / int(indices.size),
    }


def _paired_summary(
    arrays: Mapping[str, np.ndarray],
    episodes: Sequence[BindingEpisode],
    primary: np.ndarray,
    paraphrase: np.ndarray,
    run: str,
    lookup: Mapping[int, int],
    value_tokens: Mapping[str, int],
) -> dict[str, Any]:
    answer_field = "recipient_answer_id" if run == "clean" else "donor_answer_id"
    expected = np.asarray(arrays[answer_field])[primary].astype(np.int64)
    primary_full = np.asarray(arrays[f"{run}_top_token"])[primary] == expected
    paraphrase_full = np.asarray(arrays[f"{run}_top_token"])[paraphrase] == expected
    margins: dict[str, list[float]] = {"primary": [], "paraphrase": []}
    four_correct: dict[str, list[bool]] = {"primary": [], "paraphrase": []}
    for label, indices in (("primary", primary), ("paraphrase", paraphrase)):
        logits = np.asarray(arrays[f"{run}_value_logits"], dtype=np.float64)[indices]
        for local, row_index in enumerate(indices):
            tokens = tuple(
                int(value_tokens[value])
                for value in episodes[int(row_index)].recipient_values
            )
            target = int(expected[local])
            if target not in tokens:
                raise RuntimeError("recorded answer token is absent from the paired value set")
            scores = np.asarray([logits[local, lookup[token]] for token in tokens])
            margins[label].append(
                float(
                    logits[local, lookup[target]]
                    - max(logits[local, lookup[token]] for token in tokens if token != target)
                )
            )
            four_correct[label].append(tokens[int(np.argmax(scores))] == target)
    primary_four = np.asarray(four_correct["primary"])
    paraphrase_four = np.asarray(four_correct["paraphrase"])
    return {
        "pairs": int(primary.size),
        "full_accuracy_primary": float(np.mean(primary_full)),
        "full_accuracy_paraphrase": float(np.mean(paraphrase_full)),
        "full_gained_under_paraphrase": int(np.count_nonzero(~primary_full & paraphrase_full)),
        "full_lost_under_paraphrase": int(np.count_nonzero(primary_full & ~paraphrase_full)),
        "episode_four_accuracy_primary": float(np.mean(primary_four)),
        "episode_four_accuracy_paraphrase": float(np.mean(paraphrase_four)),
        "episode_four_gained_under_paraphrase": int(
            np.count_nonzero(~primary_four & paraphrase_four)
        ),
        "episode_four_lost_under_paraphrase": int(
            np.count_nonzero(primary_four & ~paraphrase_four)
        ),
        "mean_episode_margin_primary": float(np.mean(margins["primary"])),
        "mean_episode_margin_paraphrase": float(np.mean(margins["paraphrase"])),
        "mean_paired_margin_change": float(
            np.mean(np.asarray(margins["paraphrase"]) - np.asarray(margins["primary"]))
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value
