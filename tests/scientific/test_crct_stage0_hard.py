from __future__ import annotations

import torch

from causal_workspace_jepa.experiments.cross_domain.crct_stage0_hard import (
    HardSyntheticCircuit,
    _component_stats,
    _gauge_audit,
    _make_split,
    _sum_contributions,
    _full_targets,
    run_hard_benchmark,
)


def test_null_pairs_cancel_but_are_individually_action_sensitive() -> None:
    device = torch.device("cpu")
    model = HardSyntheticCircuit(17, device=device)
    split = _make_split(256, seed=99, device=device, name="test")
    metas = model.candidate_meta()
    stats = _component_stats(model, split, metas)
    left = stats["null:0"]["finite"]
    right = stats["null:1"]["finite"]
    assert float(torch.mean(left.square())) > 1e-7
    assert torch.max(torch.abs(left + right)).item() < 2e-6


def test_state_decoy_is_causally_active_in_absolute_output_but_not_action_mediator() -> None:
    device = torch.device("cpu")
    model = HardSyntheticCircuit(19, device=device)
    split = _make_split(128, seed=100, device=device, name="test")
    zero = torch.zeros_like(split.action)
    absolute = model.component_output("state_decoy:0", split.state, zero)
    effect = model.component_output("state_decoy:0", split.state, split.action) - absolute
    assert float(torch.mean(absolute.square())) > 1e-5
    assert torch.max(torch.abs(effect)).item() == 0.0


def test_component_residuals_reconstruct_full_residual() -> None:
    device = torch.device("cpu")
    model = HardSyntheticCircuit(23, device=device)
    split = _make_split(256, seed=101, device=device, name="test")
    metas = model.candidate_meta()
    stats = _component_stats(model, split, metas)
    reconstructed = _sum_contributions([meta.name for meta in metas], stats, "residual")
    target = _full_targets(model, split)["residual"]
    assert torch.max(torch.abs(reconstructed - target)).item() < 2e-5


def test_gauge_preserves_function_and_causal_ranking() -> None:
    device = torch.device("cpu")
    model = HardSyntheticCircuit(29, device=device)
    split = _make_split(256, seed=102, device=device, name="test")
    metas = model.candidate_meta()
    stats = _component_stats(model, split, metas)
    audit = _gauge_audit(model, split, metas, stats)
    assert audit["function_max_abs_error"] < 2e-5
    assert audit["causal_rank_spearman"] == 1.0
    assert audit["gauge_scale_max"] / audit["gauge_scale_min"] > 20.0


def test_smoke_preflight_executes_and_freezes_before_confirmation() -> None:
    result = run_hard_benchmark(profile_name="smoke", seed=31, device_name="cpu")
    assert result["dataset_provenance"]["plan_frozen_before_confirmation_generation"] is True
    assert result["frozen_discovery_plan"]["selected"]
    assert result["students"]["same_architecture_and_capacity"] is True
    assert result["gauge_diagnostics"]["function_max_abs_error"] <= 1e-5
    # Scientific HARD gates are intentionally not asserted here: a negative result is valid output.
