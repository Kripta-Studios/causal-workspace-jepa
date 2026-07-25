from causal_workspace_jepa.experiments.world_model.eb_jepa_competence_protocol import (
    aggregate_competence,
    planner_arm_contract,
    summarize_action_norms,
)

import pytest


def test_planner_arms_isolate_bound_from_keyword_correction() -> None:
    official = planner_arm_contract("official_mppi_as_executed")
    bounded = planner_arm_contract("bound_corrected_mppi_as_executed")
    intention = planner_arm_contract("bound_and_keyword_corrected_mppi")
    assert official["max_std"] == bounded["max_std"] == 2.0
    assert official["effective_max_norm"] is None
    assert bounded["effective_max_norm"] == 2.45
    assert intention["max_std"] == 1.5


def test_competence_requires_every_seed_in_both_required_arms() -> None:
    rows = []
    for arm in ("official_mppi_as_executed", "bound_corrected_mppi_as_executed"):
        for seed in (1, 1000, 10000):
            for episode in range(10):
                rows.append(
                    {
                        "arm": arm,
                        "training_seed": seed,
                        "success": episode < 8,
                        "executed_action_violation_count": 0,
                        "max_executed_action_norm": 2.0,
                    }
                )
    summary = aggregate_competence(rows, seeds=(1, 1000, 10000))
    assert summary["all_required_arms_competent"]
    assert summary["primary_arm_competent"]
    rows[0]["success"] = False
    rows[1]["success"] = False
    summary = aggregate_competence(rows, seeds=(1, 1000, 10000))
    assert not summary["all_required_arms_competent"]
    assert summary["primary_arm_competent"]


def test_bounded_primary_failure_blocks_eligibility_even_if_official_passes() -> None:
    rows = []
    for arm in ("official_mppi_as_executed", "bound_corrected_mppi_as_executed"):
        for seed in (1, 1000, 10000):
            for episode in range(10):
                rows.append(
                    {
                        "arm": arm,
                        "training_seed": seed,
                        "success": arm == "official_mppi_as_executed" or episode < 6,
                        "executed_action_violation_count": 0,
                        "max_executed_action_norm": 2.0,
                    }
                )
    summary = aggregate_competence(rows, seeds=(1, 1000, 10000))
    assert not summary["primary_arm_competent"]
    assert summary["primary_eligibility_arm"] == "bound_corrected_mppi_as_executed"


def test_action_norm_summary_fails_closed_on_nan() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        summarize_action_norms([1.0, float("nan")], action_max_norm=2.45)
    summary = summarize_action_norms([2.0, 2.5], action_max_norm=2.45)
    assert summary["executed_action_violation_count"] == 1
