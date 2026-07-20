"""Workspace-candidate score helpers."""

from __future__ import annotations


def selective_necessity_ratio(multistep_damage: float, one_step_damage: float, epsilon: float = 1e-12) -> float:
    return float(multistep_damage / (one_step_damage + epsilon))
