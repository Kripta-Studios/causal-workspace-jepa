from __future__ import annotations

from pathlib import Path

from causal_workspace_jepa.experiments.cross_domain.crct_stage0 import (
    PROFILES,
    Profile,
    run_benchmark,
)


def test_crct_stage0_recovers_planted_residual_circuit_and_rejects_magnitude(
    tmp_path: Path,
) -> None:
    profile = Profile(
        name="test",
        train_samples=768,
        validation_samples=192,
        test_samples=256,
        diagnostic_samples=128,
        state_dim=8,
        action_dim=5,
        route_width=8,
        bypass_width=4,
        nuisance_width=8,
        route_active=2,
        bypass_active=1,
        student_width=64,
        student_depth=2,
        student_steps=45,
        batch_size=128,
        learning_rate=3e-3,
        random_control_count=24,
    )
    PROFILES["test"] = profile
    try:
        result = run_benchmark(
            profile_name="test",
            seed=19,
            device_name="cpu",
            output=tmp_path / "result.json",
        )
    finally:
        PROFILES.pop("test", None)

    assert result["status"] == "SMOKE_VALIDATED"
    residual = result["rankings"]["residual_causal_fraction"]
    magnitude = result["rankings"]["activation_rms"]
    assert residual["average_precision"] >= 0.90
    assert residual["precision_at_truth_k"] >= 0.80
    assert magnitude["average_precision"] < residual["average_precision"]
    assert result["gauge_diagnostics"]["function_max_abs_error"] <= 2e-5
    assert result["gauge_diagnostics"]["route_residual_causal_rank_spearman"] > 0.99
    assert result["residual_student"]["test_improvement_fraction"] >= 0.50
