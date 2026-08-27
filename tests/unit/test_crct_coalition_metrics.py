from __future__ import annotations

import torch

from causal_workspace_jepa.interpretability.crct_coalition import (
    epsilon_functional_error,
    is_epsilon_sufficient,
    minimal_sufficient_sets,
    signed_contributions,
)
from causal_workspace_jepa.interpretability.ijepa_target_policy import (
    assert_no_privileged_residual,
    learned_residual_claim_eligible,
)


def test_signed_contributions_keep_negative_mean() -> None:
    contrib = {
        "pos": torch.ones(4, 2),
        "neg": -torch.ones(4, 2),
    }
    payload = signed_contributions(contrib)
    assert payload["pos"]["sign"] == 1.0
    assert payload["neg"]["sign"] == -1.0


def test_minimal_sets_exclude_supersets() -> None:
    target = torch.ones(3, 2)
    contrib = {
        "a": torch.ones(3, 2),
        "b": torch.zeros(3, 2),
    }
    sets = minimal_sufficient_sets(contrib, target, epsilon=1e-6)
    assert sets == [("a",)]


def test_epsilon_helper_threshold() -> None:
    error = epsilon_functional_error(torch.ones(2, 2), torch.ones(2, 2))
    assert is_epsilon_sufficient(error, epsilon=0.02)


def test_learned_residual_requires_replay_and_baselines() -> None:
    record = learned_residual_claim_eligible(
        residual_power=0.2,
        differential_plus_residual_heldout_nmse=0.05,
        direct_delta_heldout_nmse=0.08,
        present_baselines=[],
        original_model_replay_passed=False,
        residual_stable_across_seeds=True,
    )
    assert record["eligible"] is False
    assert record["missing_baselines"]


def test_assert_no_privileged_residual_when_direct_delta_wins() -> None:
    try:
        assert_no_privileged_residual(
            {"residual_mlp_nmse": 0.4, "direct_delta_mlp_nmse": 0.1}
        )
    except ValueError:
        return
    raise AssertionError("direct-delta win must block residual privilege")
