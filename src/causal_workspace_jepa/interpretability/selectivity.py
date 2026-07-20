"""Selectivity metrics."""

from __future__ import annotations


def specificity_ratio(target_effect: float, control_effect: float, epsilon: float = 1e-12) -> float:
    return float(target_effect / (control_effect + epsilon))
