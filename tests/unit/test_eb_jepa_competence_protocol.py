from causal_workspace_jepa.experiments.world_model.eb_jepa_competence_protocol import (
    aggregate_competence,
    planner_arm_contract,
)


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
    rows[0]["success"] = False
    rows[1]["success"] = False
    summary = aggregate_competence(rows, seeds=(1, 1000, 10000))
    assert not summary["all_required_arms_competent"]
