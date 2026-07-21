"""Frozen arms and aggregation for EB-JEPA Two Rooms competence evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_ARMS = (
    "official_mppi_as_executed",
    "bound_corrected_mppi_as_executed",
)


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


def aggregate_competence(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    required_arms: Sequence[str] = REQUIRED_ARMS,
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
    all_required_eligible = required_complete and all(
        arm_summaries[arm]["competence_eligible"] for arm in required_arms
    )
    return {
        "overall_threshold": overall_threshold,
        "per_seed_threshold": per_seed_threshold,
        "arm_summaries": arm_summaries,
        "required_arms_complete": required_complete,
        "all_required_arms_competent": all_required_eligible,
    }
