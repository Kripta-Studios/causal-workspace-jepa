import pytest

from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_phase0 import (
    ALLOWED_SPLITS,
    FORBIDDEN_SPLITS,
    assert_phase0_split_allowed,
    phase0_scientific_decision,
)


def test_phase0_has_no_protected_split_in_allowed_roster() -> None:
    assert ALLOWED_SPLITS == ("calibration", "train", "validation")
    assert set(ALLOWED_SPLITS).isdisjoint(FORBIDDEN_SPLITS)
    for split in ALLOWED_SPLITS:
        assert_phase0_split_allowed(split)
    for split in FORBIDDEN_SPLITS:
        with pytest.raises(PermissionError):
            assert_phase0_split_allowed(split)


def test_b0_failure_stops_before_derivative_phase() -> None:
    assert (
        phase0_scientific_decision(
            b0_pass=False,
            derivative_available=True,
            interaction_power=1.0,
            quadratic_nmse=1.0,
        )
        == "INELIGIBLE_TASK_PHASE0"
    )


def test_derivative_unavailability_is_not_replaced_by_finite_difference() -> None:
    assert (
        phase0_scientific_decision(
            b0_pass=True,
            derivative_available=False,
            interaction_power=None,
            quadratic_nmse=None,
        )
        == "DERIVATIVE_UNAVAILABLE_PHASE0"
    )


def test_phase0_nonlinearity_gate_is_conjunctive() -> None:
    assert (
        phase0_scientific_decision(
            b0_pass=True,
            derivative_available=True,
            interaction_power=0.11,
            quadratic_nmse=0.11,
        )
        == "PHASE0_B1_ELIGIBLE_FOR_LATER_B2"
    )
    assert (
        phase0_scientific_decision(
            b0_pass=True,
            derivative_available=True,
            interaction_power=0.09,
            quadratic_nmse=0.50,
        )
        == "COMPLETED_NEGATIVE_LOCALLY_DIFFERENTIAL"
    )


def test_bridge_authorization_is_scoped_and_parent_flags_remain_unchanged() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/experiments/crct_qwen_bridge_v1.json").read_text(encoding="utf-8")
    )
    authorization = config["scoped_authorization"]
    assert authorization["execution_authorized"] is True
    assert authorization["authorization_scope"] == "B0_B1_only_on_calibration_train_validation"
    assert authorization["allowed_splits"] == ["calibration", "train", "validation"]
    assert authorization["forbidden_splits"] == ["test", "paraphrase"]
    assert authorization["b2_b3_b4_forbidden"] is True
    assert authorization["protected_evaluation_forbidden"] is True
    assert authorization["parent_execution_authorized_flag_remains_unchanged"] is True


def test_exact_directional_derivatives_have_no_finite_difference_fallback() -> None:
    import torch

    from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_phase0 import (
        _exact_directional_jvp,
        _exact_directional_jvp_hvp,
    )

    def function(alpha: torch.Tensor) -> torch.Tensor:
        return torch.stack((alpha**2, alpha**3), dim=-1)

    alpha0 = torch.tensor([0.3, -0.2], dtype=torch.float64)
    tangent = torch.ones_like(alpha0)
    _y0, first, first_backend, _first_note = _exact_directional_jvp(
        function, alpha0, tangent
    )
    _y0b, first_b, second, second_backend, _second_note = _exact_directional_jvp_hvp(
        function, alpha0, tangent
    )
    expected_first = torch.stack((2.0 * alpha0, 3.0 * alpha0**2), dim=-1)
    expected_second = torch.stack((torch.full_like(alpha0, 2.0), 6.0 * alpha0), dim=-1)
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(first_b, expected_first)
    torch.testing.assert_close(second, expected_second)
    assert "jvp" in first_backend
    assert "jvp" in second_backend
