from __future__ import annotations

import torch

from causal_workspace_jepa.interpretability.crct_coalition import (
    is_epsilon_sufficient,
    nmse,
    restoration_error,
)


def test_restoration_error_is_nmse_not_identity() -> None:
    full = torch.ones(4, 3)
    restored = torch.zeros(4, 3)
    error = restoration_error(restored, full)
    assert error == nmse(restored, full)
    assert error > 0.5
    assert is_epsilon_sufficient(0.01, epsilon=0.02)
    assert not is_epsilon_sufficient(0.03, epsilon=0.02)
