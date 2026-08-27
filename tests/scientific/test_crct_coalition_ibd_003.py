from __future__ import annotations

import torch

from causal_workspace_jepa.experiments.cross_domain.crct_coalition_ibd_003 import (
    C_EQUIV,
    C_FORWARD,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    FORBIDDEN_SEEDS,
    CoalitionRecoveryPlant,
    _rms_replacements,
    discover_minimal_restore_sets,
    restore_only,
    run_coalition_recovery,
    threshold_digest,
)
from causal_workspace_jepa.interpretability.crct_coalition import restoration_error


def _plant(seed: int = 31) -> CoalitionRecoveryPlant:
    return CoalitionRecoveryPlant(seed, device=torch.device("cpu"))


def test_forbidden_historical_seeds_are_rejected() -> None:
    for seed in sorted(FORBIDDEN_SEEDS):
        try:
            CoalitionRecoveryPlant(seed, device=torch.device("cpu"))
        except ValueError:
            continue
        raise AssertionError(f"forbidden seed {seed} was accepted")


def test_equivalent_paths_are_distinct_implementations() -> None:
    model = _plant()
    state = torch.randn(64, 8)
    action = torch.randn(64, 4)
    acts = model.activations(state, action)
    path_a = acts["h2"] + acts["h7"]
    path_b = acts["h4"] + acts["h9"]
    assert restoration_error(path_a, path_b) < 1e-5
    assert restoration_error(acts["h2"], acts["h4"]) > 0.05
    assert restoration_error(acts["h7"], acts["h9"]) > 0.05


def test_both_class_members_are_restore_sufficient() -> None:
    model = _plant()
    state = torch.randn(96, 8)
    action = torch.randn(96, 4)
    full = model.target_output(state, action)
    restore_forward = restore_only(model, state, action, tuple(C_FORWARD))
    restore_equiv = restore_only(model, state, action, tuple(C_EQUIV))
    assert restoration_error(restore_forward, full) <= 0.02
    assert restoration_error(restore_equiv, full) <= 0.02


def test_selector_recovers_a_class_member_without_labels() -> None:
    model = _plant()
    state = torch.randn(96, 8)
    action = torch.randn(96, 4)
    found = {frozenset(item) for item in discover_minimal_restore_sets(
        model, state, action, epsilon=0.02, max_size=6
    )}
    assert C_FORWARD in found
    assert C_EQUIV in found


def test_selector_does_not_hardcode_c_forward() -> None:
    model = _plant()
    model.known_w.zero_()
    state = torch.randn(64, 8)
    action = torch.randn(64, 4)
    found = {frozenset(item) for item in discover_minimal_restore_sets(
        model, state, action, epsilon=0.02, max_size=6
    )}
    assert C_FORWARD not in found


def test_decoy_activation_is_measured_and_magnitude_misses() -> None:
    result = run_coalition_recovery(seed=31, stage="development", samples=32)
    assert result["decoy_activation_energy"] > 1.0
    assert set(result["gauge"]["magnitude_topk"]) != set(result["recovered_circuit"])
    assert result["steer_executed_nmse"] > 0.0


def test_cancellation_is_interventional() -> None:
    result = run_coalition_recovery(seed=37, stage="development", samples=32)
    cancel = result["cancellation"]
    assert cancel["h0"] >= 1e-4
    assert cancel["h1"] >= 1e-4
    assert result["signed"]["h0"]["sign"] == -result["signed"]["h1"]["sign"]


def test_gauge_activation_rank_moves() -> None:
    result = run_coalition_recovery(seed=41, stage="development", samples=32)
    assert result["gauge"]["activation_spearman"] <= 0.95
    assert result["gates"]["magnitude_selector_misses"] is True


def test_rms_controls_are_distinct() -> None:
    model = _plant()
    state = torch.randn(32, 8)
    action = torch.randn(32, 4)
    acts = model.activations(state, action)
    rows = _rms_replacements(["h2", "h7", "h3", "h5"], acts, count=8)
    keys = {tuple(sorted(row)) for row in rows}
    assert len(rows) >= 2
    assert len(keys) == len(rows)


def test_confirmation_seeds_are_not_opened_by_unit_tests() -> None:
    assert CONFIRMATION_SEEDS == (971, 977, 983)
    assert DEVELOPMENT_SEEDS == (31, 37, 41)
    assert threshold_digest()


def test_out_of_support_steer_is_executed() -> None:
    result = run_coalition_recovery(seed=31, stage="development", samples=32)
    steer = next(item for item in result["interventions"] if item["operation"] == "steer")
    assert steer["in_support"] is False
    assert steer["executed_nmse"] > 0.0
    restore = next(item for item in result["interventions"] if item["operation"] == "restore_only")
    assert restore["in_support"] is True
