"""Stable activation site names."""

from __future__ import annotations


def transformer_site(layer: int, kind: str) -> str:
    allowed = {"resid_pre", "attn_out", "mlp_out", "resid_post"}
    if kind not in allowed:
        raise ValueError(f"unknown transformer site kind: {kind}")
    return f"blocks.{layer}.{kind}"


def world_model_site(module: str, name: str) -> str:
    if not module or not name:
        raise ValueError("module and name must be non-empty")
    return f"{module}.{name}"
