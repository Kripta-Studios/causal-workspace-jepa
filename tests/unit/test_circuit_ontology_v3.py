from causal_workspace_jepa.interpretability.circuit_ontology_v3 import (
    NOT_MEASURED,
    conservative_v3_record,
    epsilon_functional_sufficiency,
    group_coverage,
    registered_gate_failures,
)


def test_epsilon_sufficiency_is_descriptive() -> None:
    score = epsilon_functional_sufficiency(0.992, threshold=0.95)
    assert score.passes_threshold
    assert abs(score.epsilon - 0.008) < 1e-12


def test_registered_failure_is_not_rescued_by_functional_recovery() -> None:
    record = conservative_v3_record(
        registered_status="NEGATIVE_RESULT",
        gates={"node_recall_ge_0_60": False, "iid_recovery": True},
        iid_confirmation={
            "circuit_recovery_fraction": 0.993,
            "node_precision": 1.0,
            "node_recall": 0.4,
            "edge_precision": 1.0,
            "edge_recall": 1.0,
        },
        ood_confirmation={
            "circuit_recovery_fraction": 0.991,
            "node_precision": 1.0,
            "node_recall": 0.4,
            "edge_precision": 1.0,
            "edge_recall": 1.0,
        },
    )
    assert record["registered_status_preserved"] == "NEGATIVE_RESULT"
    assert record["registered_gate_failures"] == ["node_recall_ge_0_60"]
    assert record["iid"]["functional_sufficiency"]["passes_threshold"] is True
    assert record["redundancy_group_coverage"]["status"] == NOT_MEASURED


def test_group_coverage_supports_future_prospective_equivalence_rules() -> None:
    groups = {"redundant_pair": ["a", "b"], "mandatory_pair": ["c", "d"]}
    any_result = group_coverage(["a", "c"], groups, any_member_suffices=True)
    all_result = group_coverage(["a", "c"], groups, any_member_suffices=False)
    assert any_result["redundant_pair"]["covered"] is True
    assert all_result["redundant_pair"]["covered"] is False


def test_registered_gate_failures_only_returns_false_booleans() -> None:
    assert registered_gate_failures({"a": True, "b": False, "c": False}) == ["b", "c"]
