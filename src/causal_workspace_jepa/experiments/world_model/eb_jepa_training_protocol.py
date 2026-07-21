"""Frozen source-config contract for the official EB-JEPA Two Rooms training portfolio."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


EXPECTED_SOURCE_CONFIG: dict[str, Any] = {
    "data.batch_size": 384,
    "data.num_workers": 16,
    "data.pin_mem": True,
    "data.persistent_workers": True,
    "training.use_amp": True,
    "training.dtype": "bfloat16",
    "model.compile": True,
    "model.dobs": 2,
    "model.henc": 32,
    "model.hpre": 32,
    "model.dstc": 32,
    "model.nsteps": 8,
    "model.encoder_architecture": "impala",
    "model.train_rollout": "last",
    "model.regularizer.cov_coeff": 8,
    "model.regularizer.std_coeff": 16,
    "model.regularizer.sim_coeff_t": 12,
    "model.regularizer.idm_coeff": 1,
    "model.regularizer.use_proj": False,
    "optim.epochs": 12,
    "optim.lr": 0.001,
    "optim.grad_clip_enc": 2.0,
    "optim.grad_clip_pred": 2.0,
    "optim.weight_decay": 1e-5,
}


def _nested_get(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        value = value[part]
    return value


def validate_source_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact frozen values, raising if the pinned upstream config drifts."""

    observed = {path: _nested_get(config, path) for path in EXPECTED_SOURCE_CONFIG}
    mismatches = {
        path: {"expected": expected, "observed": observed[path]}
        for path, expected in EXPECTED_SOURCE_CONFIG.items()
        if observed[path] != expected
    }
    if mismatches:
        raise ValueError(f"upstream training config drifted: {mismatches}")
    return observed


def required_checkpoint_names(epochs: int = 12) -> list[str]:
    return [f"e-{epoch}.pth.tar" for epoch in range(epochs)] + ["latest.pth.tar"]
