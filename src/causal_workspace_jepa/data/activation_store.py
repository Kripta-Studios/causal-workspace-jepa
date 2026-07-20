"""Activation storage estimates and lightweight manifests."""

from __future__ import annotations

from dataclasses import dataclass

from causal_workspace_jepa.common.resources import estimate_activation_bytes


@dataclass(frozen=True)
class ActivationStorageEstimate:
    examples: int
    layers: int
    positions: int
    hidden_size: int
    bytes_per_value: int
    estimated_bytes: int


def estimate_storage(
    examples: int,
    layers: int,
    positions: int,
    hidden_size: int,
    bytes_per_value: int = 2,
) -> ActivationStorageEstimate:
    return ActivationStorageEstimate(
        examples=examples,
        layers=layers,
        positions=positions,
        hidden_size=hidden_size,
        bytes_per_value=bytes_per_value,
        estimated_bytes=estimate_activation_bytes(
            examples=examples,
            layers=layers,
            positions=positions,
            hidden_size=hidden_size,
            bytes_per_value=bytes_per_value,
        ),
    )
