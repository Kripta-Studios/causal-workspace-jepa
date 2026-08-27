from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from causal_workspace_jepa.experiments.cross_domain.crct_jepa_action_delta import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    ENCODER_SITES_EXCLUDED,
    FORBIDDEN_SEEDS,
    FROZEN_THRESHOLDS,
    SITE_NAMES,
    ActionDeltaPredictor,
    _orthogonal,
    greedy_restore,
    prune_inclusion_minimal,
    run_seed,
    threshold_digest,
)


def test_forbidden_historical_seeds_are_rejected() -> None:
    for seed in sorted(FORBIDDEN_SEEDS):
        try:
            ActionDeltaPredictor(seed)
        except ValueError:
            continue
        raise AssertionError(f"forbidden seed {seed} was accepted")


def test_confirmation_seeds_are_not_opened_by_unit_tests() -> None:
    assert CONFIRMATION_SEEDS == (1013, 1019, 1021)
    assert DEVELOPMENT_SEEDS == (43, 47, 53)
    assert not set(CONFIRMATION_SEEDS) & FORBIDDEN_SEEDS
    assert not set(DEVELOPMENT_SEEDS) & FORBIDDEN_SEEDS
    assert threshold_digest()


def test_search_set_excludes_encoder_and_has_eighteen_sites() -> None:
    assert len(SITE_NAMES) == 18
    assert all(not name.startswith("enc_") for name in SITE_NAMES)
    assert ENCODER_SITES_EXCLUDED == tuple(f"enc_{i}" for i in range(6))


def test_selector_signature_has_no_physics_labels() -> None:
    source = inspect.getsource(greedy_restore)
    assert "generate_pointmass2d" not in source
    assert "force_scale" not in source
    assert "planted" not in source


def test_reforward_ablation_changes_downstream_sites() -> None:
    model = ActionDeltaPredictor(2)
    state = torch.randn(16, 4)
    action = torch.randn(16, 2)
    _, original = model.forward_intervene(state, action, None)
    means = {name: original[name].mean().detach() for name in SITE_NAMES}
    overrides = {f"act_{i}": means[f"act_{i}"] for i in range(6)}
    _, after = model.forward_intervene(state, action, overrides)
    shifted = sum(
        float((after[name] - original[name]).abs().mean().detach())
        for name in SITE_NAMES
        if name.startswith("b1_")
    )
    assert shifted > 0.0


def test_gauge_preserves_full_map() -> None:
    model = ActionDeltaPredictor(2)
    state = torch.randn(32, 4)
    action = torch.randn(32, 2)
    with torch.no_grad():
        before = model(state, action)
        model.apply_hidden_gauge(_orthogonal(2, 7), _orthogonal(2, 11), _orthogonal(2, 13))
        after = model(state, action)
        mse = float(torch.mean((before - after).square()).item())
    assert mse <= 1e-8


def test_greedy_returns_searchable_sites_on_untrained_model() -> None:
    model = ActionDeltaPredictor(2)
    state = torch.randn(24, 4)
    action = torch.randn(24, 2)
    means = {name: tensor.mean().detach() for name, tensor in model.collect_sites(state, action).items()}
    coalition, error = greedy_restore(model, state, action, means, target="dvx")
    assert set(coalition) <= set(SITE_NAMES)
    assert error >= 0.0


def test_run_seed_rejects_confirmation_seed_on_development() -> None:
    try:
        run_seed(1013, "development")
    except ValueError:
        return
    raise AssertionError("confirmation seed was accepted in development")


def test_config_thresholds_match_code() -> None:
    config = json.loads(
        Path("configs/experiments/crct_jepa_action_delta_v1.json").read_text(encoding="utf-8")
    )
    assert config["thresholds"] == FROZEN_THRESHOLDS
    assert FROZEN_THRESHOLDS["max_coalition"] == 4
    assert FROZEN_THRESHOLDS["min_step_nmse"] == 0.02
    assert FROZEN_THRESHOLDS["random_control_sufficient_max"] == 0


def test_min_step_uses_absolute_nmse_not_shrinking_current_error() -> None:
    source = inspect.getsource(greedy_restore)
    assert 'FROZEN_THRESHOLDS["min_step_nmse"]' in source
    assert "current_error *" not in source


def test_prune_drops_redundant_member_on_untrained_model() -> None:
    model = ActionDeltaPredictor(2)
    state = torch.randn(16, 4)
    action = torch.randn(16, 2)
    means = {name: tensor.mean().detach() for name, tensor in model.collect_sites(state, action).items()}
    pruned = prune_inclusion_minimal(
        model, state, action, means, list(SITE_NAMES), target="dvx"
    )
    assert len(pruned) <= FROZEN_THRESHOLDS["max_coalition"] or len(pruned) < len(SITE_NAMES)
    assert set(pruned) <= set(SITE_NAMES)

