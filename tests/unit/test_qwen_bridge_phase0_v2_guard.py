from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_phase0_v2 import (
    _strict_spaced_token_id,
)

ROOT = Path(__file__).resolve().parents[2]


class _FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        if text == "alpha":
            return {"input_ids": [11]}
        if text == " alpha":
            return {"input_ids": [12, 13]}
        if text == " beta":
            return {"input_ids": [21]}
        return {"input_ids": [99, 100]}

    def decode(self, ids, **kwargs):
        del kwargs
        if list(ids) == [21]:
            return " beta"
        if list(ids) == [11]:
            return "alpha"
        return "<?>"


def test_bare_only_value_is_rejected_but_spaced_exact_value_is_accepted() -> None:
    tokenizer = _FakeTokenizer()
    with pytest.raises(ValueError):
        _strict_spaced_token_id(tokenizer, "alpha")
    assert _strict_spaced_token_id(tokenizer, "beta") == 21


def test_bridge_v2_keeps_bridge_v1_phase0_thresholds_exactly() -> None:
    old = json.loads(
        (ROOT / "configs/experiments/crct_qwen_bridge_v1.json").read_text(encoding="utf-8")
    )
    new = json.loads(
        (ROOT / "configs/experiments/crct_qwen_bridge_v2.json").read_text(encoding="utf-8")
    )
    assert new["phase0_thresholds"] == old["phase0_thresholds"]
    assert new["scoped_authorization"]["allowed_splits"] == [
        "calibration",
        "train",
        "validation",
    ]
    assert new["scoped_authorization"]["forbidden_splits"] == ["test", "paraphrase"]
    assert new["scoped_authorization"]["b2_b3_b4_forbidden"] is True


def test_phase0_v2_has_no_cli_for_protected_splits_and_correct_forward_telemetry() -> None:
    source = (
        ROOT
        / "src/causal_workspace_jepa/experiments/llm/qwen_binding_algebra_phase0_v2.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("--test", "--paraphrase", "--protected", "--split"):
        assert forbidden not in source
    assert "SPLIT_MATERIALIZED_NO_FORWARD" in source
    assert "TOKEN_CONTRACT_VERIFIED_PRE_MODEL" in source
    assert "B0_MODEL_FORWARD_EXECUTION_STARTED" in source
    assert "B0_MODEL_FORWARD_EXECUTION_COMPLETE" in source
    assert 'status_payload["model_forward_splits_completed"] = list(ALLOWED_SPLITS)' in source


def test_token_amendment_preparer_cannot_import_or_load_model_weights() -> None:
    source = (
        ROOT / "scripts/prepare_qwen_binding_algebra_v3_token_amendment.py"
    ).read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert "forward(" not in source
    assert "generate(" not in source
    assert "AutoTokenizer" in source
