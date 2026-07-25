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


def validate_competence_job_payload(
    payload: Mapping[str, Any],
    *,
    experiment_id: str,
    repo_commit: str,
    source_revision: str,
    training_seed: int,
    checkpoint_epoch: int,
    checkpoint_sha256: str,
    arm: str,
    arm_contract: Mapping[str, Any],
    analysis_seed: int,
    environment_seed: int,
    num_episodes: int,
) -> None:
    """Reject stale, incomplete, or relabeled ignored competence jobs."""

    expected_identity = {
        "status": "COMPLETED",
        "experiment_id": experiment_id,
        "repo_commit": repo_commit,
        "repo_dirty_at_start": False,
        "source_revision": source_revision,
        "source_clean": True,
        "training_seed": int(training_seed),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_recorded_epoch": int(checkpoint_epoch),
        "checkpoint_sha256": checkpoint_sha256,
        "arm": arm,
        "arm_contract": dict(arm_contract),
        "analysis_seed": int(analysis_seed),
        "environment_seed": int(environment_seed),
    }
    observed = {key: payload.get(key) for key in expected_identity}
    if observed != expected_identity:
        raise RuntimeError("existing competence job identity differs from the frozen job")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != num_episodes:
        raise RuntimeError("existing competence job has the wrong episode count")
    for index, row in enumerate(episodes):
        if not isinstance(row, Mapping):
            raise RuntimeError("existing competence job contains a malformed episode")
        row_identity = (
            row.get("arm"),
            row.get("training_seed"),
            row.get("checkpoint_epoch"),
            row.get("episode"),
        )
        if row_identity != (arm, training_seed, checkpoint_epoch, index):
            raise RuntimeError("existing competence episode identity/order differs")
        numeric = (
            row.get("final_state_distance"),
            row.get("episode_seconds"),
            row.get("max_executed_action_norm"),
        )
        if type(row.get("success")) is not bool or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0
            for value in numeric
        ):
            raise RuntimeError("existing competence episode has invalid outcomes")
        if not isinstance(row.get("executed_action_count"), int) or row[
            "executed_action_count"
        ] <= 0:
            raise RuntimeError("existing competence episode has no executed actions")
        if not isinstance(row.get("executed_action_violation_count"), int) or row[
            "executed_action_violation_count"
        ] < 0:
            raise RuntimeError("existing competence episode has invalid violation count")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("existing competence job lacks a summary")
    expected_summary = {
        "success_rate": sum(row["success"] for row in episodes) / num_episodes,
        "mean_final_state_distance": sum(row["final_state_distance"] for row in episodes)
        / num_episodes,
        "executed_action_violation_count": sum(
            row["executed_action_violation_count"] for row in episodes
        ),
        "max_executed_action_norm": max(
            row["max_executed_action_norm"] for row in episodes
        ),
    }
    for key, expected in expected_summary.items():
        observed_value = summary.get(key)
        if not isinstance(observed_value, (int, float)) or not math.isclose(
            float(observed_value), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"existing competence summary differs at {key}")


def aggregate_competence(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    checkpoint_epochs: Sequence[int] | None = None,
    episodes_per_job: int | None = None,
    required_arms: Sequence[str] = REQUIRED_ARMS,
    primary_eligibility_arm: str = PRIMARY_ELIGIBILITY_ARM,
    overall_threshold: float = 0.80,
    per_seed_threshold: float = 0.70,
) -> dict[str, Any]:
    if (checkpoint_epochs is None) != (episodes_per_job is None):
        raise ValueError("checkpoint_epochs and episodes_per_job must be provided together")
    if checkpoint_epochs is not None:
        if not checkpoint_epochs or episodes_per_job is None or episodes_per_job <= 0:
            raise ValueError("the frozen checkpoint roster and episode count must be nonempty")
        expected = {
            (int(seed), int(epoch), str(arm), episode)
            for seed in seeds
            for epoch in checkpoint_epochs
            for arm in required_arms
            for episode in range(episodes_per_job)
        }
        observed = {
            (
                int(row["training_seed"]),
                int(row["checkpoint_epoch"]),
                str(row["arm"]),
                int(row["episode"]),
            )
            for row in rows
        }
        if len(rows) != len(expected) or observed != expected:
            raise RuntimeError("competence rows do not match the exact frozen job roster")
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
