\
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/qwen_competence_recovery_v1.json"
MODULE = (
    ROOT
    / "src/causal_workspace_jepa/experiments/llm/qwen_binding_competence_recovery.py"
)


def test_config_is_calibration_only_and_thresholds_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["allowed_model_forward_splits"] == ["calibration"]
    assert config["forbidden_model_forward_splits"] == [
        "train",
        "validation",
        "test",
        "paraphrase",
    ]
    assert config["selection"]["clean_full_vocab_accuracy_min"] == 0.90
    assert config["selection"]["direct_permuted_full_vocab_accuracy_min"] == 0.90
    assert config["selection"]["candidate_only_accuracy_is_diagnostic_only"] is True
    assert config["scientific_boundary"]["no_b1"] is True
    assert config["scientific_boundary"]["no_validation_access"] is True
    assert config["scientific_boundary"]["no_test_access"] is True


def test_prompt_roster_and_selection_order_are_fixed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = config["prompt_variants"]
    assert [(item["id"], item["priority"]) for item in rows] == [
        ("legacy_v3_control", 0),
        ("explicit_plain_v1", 1),
        ("qwen_chat_prefill_v1", 2),
        ("qwen_chat_prefill_fewshot_v1", 3),
    ]
    assert config["selection"]["primary_score"] == "min_clean_direct_full_vocab_accuracy"
    assert config["selection"]["secondary_score"] == "mean_clean_direct_full_vocab_accuracy"
    assert config["selection"]["tie_break"] == "lower_priority_integer"


def test_runtime_module_has_no_code_path_naming_forbidden_split_generation() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # The names may appear in the explicit deny-list and result fields, but no call
    # may pass one of them as a split argument.
    forbidden = {"train", "validation", "test", "paraphrase"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "split" and isinstance(keyword.value, ast.Constant):
                assert keyword.value.value not in forbidden


def test_result_contract_marks_all_non_calibration_scopes_false() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert '"train_executed": False' in source
    assert '"validation_executed": False' in source
    assert '"test_executed": False' in source
    assert '"paraphrase_executed": False' in source
    assert '"protected_splits_executed": []' in source


def test_qwen_chat_variant_is_non_thinking_assistant_prefill() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "continue_final_message=True" in source
    assert "add_generation_prompt=False" in source
    assert "enable_thinking=False" in source
    assert '{"role": "assistant", "content": "Answer:"}' in source
