"""Frozen arms and aggregation for EB-JEPA Two Rooms competence evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import math


REQUIRED_ARMS = (
    "official_mppi_as_executed",
    "bound_corrected_mppi_as_executed",
)
PRIMARY_ELIGIBILITY_ARM = "bound_corrected_mppi_as_executed"


def planner_arm_contract(arm: str) -> dict[str, Any]:
    contracts = {
        "official_mppi_as_executed": {
            "planner_implementation": "official",
            "max_std": 2.0,
            "effective_max_norm": None,
        },
        "bound_corrected_mppi_as_executed": {
            "planner_implementation": "constraint_corrected",
            "max_std": 2.0,
            "effective_max_norm": 2.45,
        },
        "bound_and_keyword_corrected_mppi": {
            "planner_implementation": "constraint_corrected",
            "max_std": 1.5,
            "effective_max_norm": 2.45,
        },
    }
    if arm not in contracts:
        raise ValueError(f"unknown planner arm: {arm}")
    return dict(contracts[arm])


def summarize_action_norms(
    action_norms: Sequence[float], *, action_max_norm: float
) -> dict[str, float | int]:
    """Reject non-finite planner actions before computing contract metrics."""

    values = [float(value) for value in action_norms]
    if not values:
        raise ValueError("at least one executed action norm is required")
    if not math.isfinite(action_max_norm) or action_max_norm <= 0.0:
        raise ValueError("action_max_norm must be finite and positive")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError("planner produced a non-finite or negative action norm")
    return {
        "executed_action_count": len(values),
        "executed_action_violation_count": sum(
            value > action_max_norm + 1e-6 for value in values
        ),
        "max_executed_action_norm": max(values),
    }


def aggregate_competence(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    required_arms: Sequence[str] = REQUIRED_ARMS,
    primary_eligibility_arm: str = PRIMARY_ELIGIBILITY_ARM,
    overall_threshold: float = 0.80,
    per_seed_threshold: float = 0.70,
) -> dict[str, Any]:
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    arm_summaries: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        per_seed = {}
        for seed in seeds:
            seed_rows = [row for row in arm_rows if int(row["training_seed"]) == int(seed)]
            per_seed[str(seed)] = (
                sum(bool(row["success"]) for row in seed_rows) / len(seed_rows)
                if seed_rows
                else None
            )
        overall = sum(bool(row["success"]) for row in arm_rows) / len(arm_rows)
        eligible = arm in required_arms and overall >= overall_threshold and all(
            per_seed[str(seed)] is not None
            and float(per_seed[str(seed)]) >= per_seed_threshold
            for seed in seeds
        )
        arm_summaries[arm] = {
            "episodes": len(arm_rows),
            "overall_success_rate": overall,
            "per_seed_success_rate": per_seed,
            "competence_eligible": eligible,
            "executed_action_violation_count": sum(
                int(row.get("executed_action_violation_count", 0)) for row in arm_rows
            ),
            "max_executed_action_norm": max(
                float(row.get("max_executed_action_norm", 0.0)) for row in arm_rows
            ),
        }
    required_complete = all(arm in arm_summaries for arm in required_arms)
    if primary_eligibility_arm not in required_arms:
        raise ValueError("primary eligibility arm must be one of the required arms")
    primary_eligible = bool(
        required_complete
        and arm_summaries[primary_eligibility_arm]["competence_eligible"]
    )
    return {
        "overall_threshold": overall_threshold,
        "per_seed_threshold": per_seed_threshold,
        "arm_summaries": arm_summaries,
        "required_arms_complete": required_complete,
        "primary_eligibility_arm": primary_eligibility_arm,
        "primary_arm_competent": primary_eligible,
        "all_required_arms_competent": bool(
            required_complete
            and all(arm_summaries[arm]["competence_eligible"] for arm in required_arms)
        ),
    }
