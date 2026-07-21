from causal_workspace_jepa.experiments.world_model.eb_jepa_planner_config import (
    evaluate_planner_config,
)


def test_planner_config_detects_silent_scale_mismatch() -> None:
    payload = {
        "yaml": {"mppi_var_scale": 1.5, "mppi_has_max_std": False, "cem_var_scale": 1.5},
        "signatures": {
            "mppi_parameters": ["self", "unroll", "max_std", "kwargs"],
            "mppi_accepts_kwargs": True,
        },
        "instances": {
            "official_mppi_max_std": 2.0,
            "official_cem_var_scale": 1.5,
            "keyword_corrected_mppi_max_std": 1.5,
        },
        "interpretation": {
            "var_scale_is_silently_ignored_by_mppi": True,
            "actual_proposal_scales_differ": True,
        },
    }
    assert all(evaluate_planner_config(payload).values())
