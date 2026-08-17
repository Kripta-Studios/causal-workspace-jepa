"""Development-only CRCT diagnostics on the already-open Qwen capital patch shard.

This module deliberately reuses a disclosed dataset.  It is not a fresh confirmation set and
cannot create a circuit, J-space, workspace, or novelty claim.  Its role is to debug endpoint
normalization and direct-delta versus Taylor-residual prediction before a new protected study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SHARD = ROOT / "data/activations/qwen_capital_patches_v1/shard-00000-of-00001.h5"
DEFAULT_EXPECTED_SHA256 = "b02340368836a00b4ecada84dba3484bf0c46f59bfe6d78bdc58db3fc7e0b951"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.square(target - prediction, dtype=np.float64).sum())
    denominator = float(np.square(target, dtype=np.float64).sum())
    return numerator / max(denominator, 1e-18)


def residual_power(target: np.ndarray, baseline: np.ndarray) -> float:
    return nmse(target, baseline)


def _split_metrics(
    target: np.ndarray,
    jvp: np.ndarray,
    quadratic: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    y = target[indices]
    first = jvp[indices]
    second = quadratic[indices]
    first_row = np.square(y - first, dtype=np.float64).sum(axis=1)
    second_row = np.square(y - second, dtype=np.float64).sum(axis=1)
    target_row = np.square(y, dtype=np.float64).sum(axis=1)
    rel_first = first_row / np.maximum(target_row, 1e-18)
    rel_second = second_row / np.maximum(target_row, 1e-18)
    return {
        "row_count": int(len(indices)),
        "first_order_nmse": nmse(y, first),
        "quadratic_nmse": nmse(y, second),
        "quadratic_residual_power_fraction": residual_power(y, second),
        "quadratic_better_row_fraction": float(np.mean(second_row < first_row)),
        "quadratic_worse_by_10pct_row_fraction": float(
            np.mean(second_row > 1.10 * np.maximum(first_row, 1e-18))
        ),
        "median_row_first_order_relative_mse": float(np.median(rel_first)),
        "median_row_quadratic_relative_mse": float(np.median(rel_second)),
    }


def _block_metrics(
    target: np.ndarray,
    jvp: np.ndarray,
    quadratic: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    hidden = slice(0, 1024)
    logits = slice(1024, target.shape[1])
    return {
        "full_endpoint": _split_metrics(target, jvp, quadratic, indices),
        "hidden_1024": _split_metrics(
            target[:, hidden], jvp[:, hidden], quadratic[:, hidden], indices
        ),
        "answer_logit_block": _split_metrics(
            target[:, logits], jvp[:, logits], quadratic[:, logits], indices
        ),
    }


def _ridge_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    lambdas: Iterable[float],
) -> dict[float, np.ndarray]:
    """Dual-safe SVD ridge predictions; one SVD is reused for the lambda grid."""

    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    xc = np.asarray(x_train - x_mean, dtype=np.float64)
    yc = np.asarray(y_train - y_mean, dtype=np.float64)
    xe = np.asarray(x_eval - x_mean, dtype=np.float64)
    u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    projected_y = u.T @ yc
    projected_x = xe @ vt.T
    predictions: dict[float, np.ndarray] = {}
    for value in lambdas:
        lam = float(value)
        filt = singular / (singular * singular + lam)
        predictions[lam] = (projected_x * filt[None, :]) @ projected_y + y_mean
    return predictions


def _fit_ridge_family(
    x: np.ndarray,
    target: np.ndarray,
    split_id: np.ndarray,
    lambdas: list[float],
) -> dict[str, Any]:
    train = np.flatnonzero(split_id == 0)
    validation = np.flatnonzero(split_id == 1)
    test = np.flatnonzero(split_id == 2)
    pred_val = _ridge_predictions(x[train], target[train], x[validation], lambdas)
    scored = {lam: nmse(target[validation], pred) for lam, pred in pred_val.items()}
    chosen = min(scored, key=lambda value: (scored[value], value))
    pred_test = _ridge_predictions(x[train], target[train], x[test], [chosen])[chosen]
    return {
        "selected_lambda_train_to_validation_only": float(chosen),
        "validation_nmse_by_lambda": {str(k): float(v) for k, v in scored.items()},
        "validation_nmse": float(scored[chosen]),
        "test_nmse": nmse(target[test], pred_test),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }


def analyze_arrays(
    *,
    source_delta: np.ndarray,
    target_effect: np.ndarray,
    exact_jvp: np.ndarray,
    quadratic_taylor: np.ndarray,
    split_id: np.ndarray,
    lambdas: list[float] | None = None,
) -> dict[str, Any]:
    """Analyze disclosed arrays without any new Qwen forward pass."""

    source_delta = np.asarray(source_delta, dtype=np.float64)
    target_effect = np.asarray(target_effect, dtype=np.float64)
    exact_jvp = np.asarray(exact_jvp, dtype=np.float64)
    quadratic_taylor = np.asarray(quadratic_taylor, dtype=np.float64)
    split_id = np.asarray(split_id, dtype=np.int64).reshape(-1)
    if target_effect.shape != exact_jvp.shape or target_effect.shape != quadratic_taylor.shape:
        raise ValueError("target/JVP/quadratic arrays must have identical shapes")
    if (
        target_effect.shape[0] != source_delta.shape[0]
        or target_effect.shape[0] != split_id.shape[0]
    ):
        raise ValueError("all arrays must have identical row counts")
    if target_effect.shape[1] < 1025:
        raise ValueError("capital endpoint must contain hidden coordinates plus answer logits")

    grid = lambdas or [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    split_names = {0: "train", 1: "validation", 2: "test"}
    split_metrics = {
        name: _block_metrics(
            target_effect,
            exact_jvp,
            quadratic_taylor,
            np.flatnonzero(split_id == split_value),
        )
        for split_value, name in split_names.items()
    }

    quadratic_residual = target_effect - quadratic_taylor
    direct_family = _fit_ridge_family(source_delta, target_effect, split_id, grid)
    residual_family_raw = _fit_ridge_family(source_delta, quadratic_residual, split_id, grid)

    train = np.flatnonzero(split_id == 0)
    validation = np.flatnonzero(split_id == 1)
    test = np.flatnonzero(split_id == 2)
    chosen_residual_lambda = residual_family_raw["selected_lambda_train_to_validation_only"]
    residual_val = _ridge_predictions(
        source_delta[train],
        quadratic_residual[train],
        source_delta[validation],
        [chosen_residual_lambda],
    )[chosen_residual_lambda]
    residual_test = _ridge_predictions(
        source_delta[train],
        quadratic_residual[train],
        source_delta[test],
        [chosen_residual_lambda],
    )[chosen_residual_lambda]
    residual_family = {
        **residual_family_raw,
        "validation_reconstructed_effect_nmse": nmse(
            target_effect[validation], quadratic_taylor[validation] + residual_val
        ),
        "test_reconstructed_effect_nmse": nmse(
            target_effect[test], quadratic_taylor[test] + residual_test
        ),
    }

    return {
        "schema_version": "qwen_capital_crct_dev_v1",
        "status": "DEVELOPMENT_ANALYSIS_COMPLETE",
        "row_count": int(target_effect.shape[0]),
        "source_dim": int(source_delta.shape[1]),
        "endpoint_dim": int(target_effect.shape[1]),
        "split_counts": {
            name: int(np.sum(split_id == value)) for value, name in split_names.items()
        },
        "differential_diagnostics": split_metrics,
        "equal_family_ridge": {
            "direct_delta": direct_family,
            "quadratic_residual": residual_family,
            "interpretation": (
                "Both predictors use the same centered SVD-ridge family. Lambda is selected on "
                "the already-open validation split only. The test split is already disclosed and "
                "therefore remains development evidence, not fresh confirmation."
            ),
        },
        "scientific_boundary": {
            "qwen_forward_executed": False,
            "all_capital_splits_already_open": True,
            "fresh_confirmation_claim_permitted": False,
            "circuit_claim_permitted": False,
            "workspace_claim_permitted": False,
        },
    }


def analyze_shard(path: Path, *, expected_sha256: str = DEFAULT_EXPECTED_SHA256) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "schema_version": "qwen_capital_crct_dev_v1",
            "status": "MISSING_RAW_DEV_DATA",
            "path": str(path),
            "expected_sha256": expected_sha256,
            "scientific_boundary": {
                "qwen_forward_executed": False,
                "fresh_confirmation_claim_permitted": False,
            },
        }
    actual = _sha256(path)
    if expected_sha256 and actual != expected_sha256:
        return {
            "schema_version": "qwen_capital_crct_dev_v1",
            "status": "RAW_DEV_DATA_HASH_MISMATCH",
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual,
        }

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        return {
            "schema_version": "qwen_capital_crct_dev_v1",
            "status": "MISSING_H5PY",
            "error": str(exc),
        }

    with h5py.File(path, "r") as handle:
        required = ["source_delta", "target_effect", "exact_jvp", "quadratic_taylor", "split_id"]
        if "arrays" not in handle:
            raise KeyError("capital shard missing required HDF5 group: arrays")
        group = handle["arrays"]
        missing = [name for name in required if name not in group]
        if missing:
            raise KeyError(f"capital shard missing arrays under arrays/: {missing}")
        arrays = {name: group[name][...] for name in required}
    payload = analyze_arrays(**arrays)
    payload["source_shard"] = {
        "path": str(path),
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "hash_matches": True,
        "bytes": path.stat().st_size,
    }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, default=DEFAULT_SHARD)
    parser.add_argument("--expected-sha256", default=DEFAULT_EXPECTED_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = analyze_shard(args.shard, expected_sha256=args.expected_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
