from __future__ import annotations

import torch

from causal_workspace_jepa.experiments.cross_domain.crct_coalition_ibd import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    HARD002_PRIMARY_SEEDS,
    InterpretableBottleneckCircuit,
    run_coalition_benchmark,
    run_stage,
    threshold_digest,
)
from causal_workspace_jepa.interpretability.crct_coalition import (
    cancellation_groups,
    intervention_support_label,
    literal_recall,
)
from causal_workspace_jepa.interpretability.ijepa_target_policy import (
    REQUIRED_IJEPA_BASELINES,
    learned_residual_claim_eligible,
    missing_required_baselines,
)


def test_hard002_primary_seeds_are_rejected() -> None:
    for seed in HARD002_PRIMARY_SEEDS:
        try:
            InterpretableBottleneckCircuit(seed, device=torch.device("cpu"))
        except ValueError:
            continue
        raise AssertionError(f"HARD-002 seed {seed} was reused")


def test_equivalent_minimal_circuits_are_not_the_full_graph() -> None:
    result = run_coalition_benchmark(seed=11, stage="development", device_name="cpu", samples=96)
    minimal = [set(item) for item in result["iid"]["minimal_sets"]]
    assert {"known_a", "unknown", "residual"} in minimal
    assert {"known_b", "unknown", "residual"} in minimal
    assert set(result["planted_nodes"]) not in minimal
    assert result["iid"]["sufficient"] is True
    assert result["iid"]["literal"]["recall"] < 1.0
    assert result["ontology_distinguished"] is True
    assert result["matched_control_report"]["sufficient"] is False


def test_cancellation_is_signed_and_visible() -> None:
    model = InterpretableBottleneckCircuit(13, device=torch.device("cpu"))
    state = torch.randn(64, 8)
    action = torch.randn(64, 4)
    contrib = {name: model.component_output(name, state, action) for name in ("cancel_pos", "cancel_neg")}
    groups = cancellation_groups(
        contrib,
        sum_energy_ratio_max=0.02,
        min_member_energy=1e-4,
    )
    assert groups
    assert groups[0]["signed"] is True
    assert set(groups[0]["members"]) == {"cancel_pos", "cancel_neg"}


def test_decoy_has_high_activation_but_zero_causal_use() -> None:
    result = run_coalition_benchmark(seed=17, stage="development", device_name="cpu", samples=64)
    assert result["decoy_activation_energy"] > 1.0
    assert result["decoy_causal_energy"] == 0.0


def test_ood_and_iid_both_keep_functional_equivalence() -> None:
    result = run_coalition_benchmark(seed=13, stage="development", device_name="cpu", samples=80)
    assert result["iid"]["sufficient"] is True
    assert result["ood"]["sufficient"] is True
    assert any(item["equivalent"] for item in result["iid"]["equivalence"])


def test_out_of_support_intervention_is_labeled() -> None:
    label = intervention_support_label(
        operation="steer",
        site="known_a",
        magnitude=8.0,
        combination=("known_a", "decoy"),
        trained_operations=("zero",),
        trained_sites=("known_a",),
        magnitude_max=1.0,
        trained_combination_size_max=1,
    )
    assert label["in_support"] is False
    assert label["support_status"] == "out_of_support"


def test_literal_recall_differs_from_sufficiency() -> None:
    recall = literal_recall(["known_a", "unknown", "residual"], ["known_a", "known_b", "unknown", "residual"])
    assert recall["recall"] == 0.75
    assert recall["precision"] == 1.0


def test_development_stage_runs_frozen_seeds() -> None:
    payload = run_stage("development")
    assert payload["seeds"] == list(DEVELOPMENT_SEEDS)
    assert payload["threshold_digest"] == threshold_digest()
    assert payload["all_seeds_distinguish_ontology"] is True
    assert CONFIRMATION_SEEDS == (811, 823, 829)


def test_ijepa_residual_is_not_privileged_without_fair_baselines() -> None:
    assert "direct_delta_capacity_matched" in REQUIRED_IJEPA_BASELINES
    assert missing_required_baselines(["jvp_first_order"]) 
    record = learned_residual_claim_eligible(
        residual_power=0.04,
        differential_plus_residual_heldout_nmse=0.2,
        direct_delta_heldout_nmse=0.1,
        present_baselines=REQUIRED_IJEPA_BASELINES,
        original_model_replay_passed=True,
        residual_stable_across_seeds=True,
    )
    assert record["eligible"] is False
    assert record["hard002_status_preserved"] == "NEGATIVE_RESULT"
    assert record["beats_direct_delta"] is False
