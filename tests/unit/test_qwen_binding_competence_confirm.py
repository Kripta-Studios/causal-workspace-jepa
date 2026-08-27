from __future__ import annotations

import ast
import json
from pathlib import Path

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.llm.qwen_binding_competence_confirm import (
    CONFIRM_SPLIT,
    FORBIDDEN_SPLITS,
    FROZEN_RENDERER,
    assert_confirmation_disjoint,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/qwen_competence_confirm_v1.json"
MODULE = (
    ROOT / "src/causal_workspace_jepa/experiments/llm/qwen_binding_competence_confirm.py"
)
PARENT = ROOT / "configs/experiments/qwen_binding_algebra_v3.yaml"


def test_confirmation_is_fresh_and_thresholds_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "QWEN-BINDING-COMPETENCE-CONFIRM-001"
    assert config["allowed_model_forward_splits"] == ["confirmation"]
    assert config["forbidden_model_forward_splits"] == [
        "train",
        "validation",
        "test",
        "paraphrase",
        "calibration",
    ]
    assert list(FORBIDDEN_SPLITS) == config["forbidden_model_forward_splits"]
    assert config["renderer"]["id"] == FROZEN_RENDERER
    assert config["gates"]["clean_full_vocab_accuracy_min"] == 0.9
    assert config["gates"]["direct_permuted_full_vocab_accuracy_min"] == 0.9
    assert config["gates"]["candidate_only_accuracy_is_diagnostic_only"] is True
    assert config["scientific_boundary"]["no_v3_rescue"] is True
    assert config["scientific_boundary"]["no_prompt_search"] is True
    assert config["split"]["name"] == CONFIRM_SPLIT
    assert config["split"]["seed"] == 701


def test_confirmation_tokens_are_disjoint_from_parent_pools() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = load_config(PARENT)
    for role in ("keys", "values"):
        for split, values in parent["token_pools"][role].items():
            assert list(values) == list(config["parent_token_pools"][role][split])
    assert_confirmation_disjoint(
        config["parent_token_pools"],
        config["token_pools"]["keys"][CONFIRM_SPLIT],
        config["token_pools"]["values"][CONFIRM_SPLIT],
    )


def test_confirmation_module_does_not_call_protected_splits() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"train", "validation", "test", "paraphrase", "calibration"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "split" and isinstance(keyword.value, ast.Constant):
                assert keyword.value.value not in forbidden


def test_confirmation_result_contract_keeps_protected_false() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert '"train_executed": False' in source
    assert '"validation_executed": False' in source
    assert '"test_executed": False' in source
    assert '"paraphrase_executed": False' in source
    assert '"calibration_executed": False' in source
    assert '"protected_splits_executed": []' in source
    assert '"does_not_rescue_v3": True' in source
