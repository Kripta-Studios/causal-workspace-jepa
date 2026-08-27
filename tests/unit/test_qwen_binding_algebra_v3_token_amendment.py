from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from causal_workspace_jepa.common.config import load_config

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/experiments/qwen_binding_algebra_v3_token_amendment.json"
V2 = ROOT / "configs/experiments/qwen_binding_algebra_v2.yaml"
CR_V1 = ROOT / "configs/experiments/qwen_binding_algebra_cr_v1.yaml"
V3 = ROOT / "configs/experiments/qwen_binding_algebra_v3.yaml"
CR_V2 = ROOT / "configs/experiments/qwen_binding_algebra_cr_v2.yaml"
CONTRACT = ROOT / "configs/experiments/qwen_binding_algebra_v3_token_contract.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_generated_v3_contract_is_complete_and_source_configs_remain_frozen() -> None:
    for path in (V3, CR_V2, CONTRACT):
        assert path.is_file(), (
            f"missing generated tokenizer amendment artifact {path}; run "
            "python scripts/prepare_qwen_binding_algebra_v3_token_amendment.py first"
        )
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert (
        _sha(V2)
        == spec["source_files"]["configs/experiments/qwen_binding_algebra_v2.yaml"][
            "sha256"
        ]
    )
    assert (
        _sha(CR_V1)
        == spec["source_files"]["configs/experiments/qwen_binding_algebra_cr_v1.yaml"][
            "sha256"
        ]
    )
    copy_contract = dict(contract)
    expected_self = copy_contract.pop("self_sha256")
    assert _canonical(copy_contract) == expected_self
    assert contract["source_invalid_total"] == 23
    assert contract["replacement_count"] == 23
    assert contract["observed_source_invalid_values"] == spec["expected_source_invalid_values"]
    assert contract["selection_depends_only_on_tokenizer_metadata"] is True
    assert contract["model_outputs_or_logits_used"] is False
    assert contract["protected_prompts_materialized"] is False
    assert contract["all_resolved_values_strict_spaced_single_token"] is True
    assert _sha(V3) == contract["resolved_v3_sha256"]
    assert _sha(CR_V2) == contract["resolved_cr_v2_sha256"]


def test_v3_changes_only_failed_values_and_preserves_scientific_protocol_fields() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    old = _load_yaml(V2)
    new = _load_yaml(V3)
    assert new["id"] == "LLM-QWEN-BINDING-ALGEBRA-003"
    assert new["execution_authorized"] is False
    assert new["model"] == old["model"]
    assert new["revision"] == old["revision"]
    for field in (
        "splits",
        "permutation_convention",
        "action_partition",
        "treatment",
        "capture",
        "meta_model",
        "loss",
        "baselines",
        "controls",
        "gates",
    ):
        assert new[field] == old[field]
    assert new["token_pools"]["keys"] == old["token_pools"]["keys"]
    expected_invalid = spec["expected_source_invalid_values"]
    for split in ("calibration", "train", "validation", "test"):
        old_values = old["token_pools"]["values"][split]
        new_values = new["token_pools"]["values"][split]
        assert len(old_values) == len(new_values)
        for original, replacement in zip(old_values, new_values):
            if original in expected_invalid[split]:
                assert replacement != original
            else:
                assert replacement == original


def test_cr_v2_is_same_resolved_alphabet_and_remains_unauthorized() -> None:
    v3 = _load_yaml(V3)
    cr = _load_yaml(CR_V2)
    assert cr["experiment_id"] == "QWEN-BINDING-ALGEBRA-CR-V2"
    assert cr["parent_linkage"]["parent_experiment_id"] == "LLM-QWEN-BINDING-ALGEBRA-003"
    assert (
        cr["parent_linkage"]["parent_config"]
        == "configs/experiments/qwen_binding_algebra_v3.yaml"
    )
    assert cr["token_pools"]["values"] == v3["token_pools"]["values"]
    assert cr["authorization"]["execution_authorized"] is False
    assert cr["authorization"]["protected_execution_authorized"] is False
    copy_cr = dict(cr)
    expected = copy_cr.pop("canonical_digest")
    assert _canonical(copy_cr) == expected


@pytest.mark.gpu
def test_resolved_values_match_pinned_tokenizer_strict_answer_context() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            contract["model"],
            revision=contract["revision"],
            local_files_only=True,
        )
    except OSError:
        pytest.skip("SKIPPED_RESOURCE: pinned Qwen tokenizer is not in the local cache")
    rows = []
    for split in ("calibration", "train", "validation", "test"):
        for slot, value in enumerate(contract["resolved_values"][split]):
            ids = tokenizer(" " + value, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1
            decoded = tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            assert decoded == " " + value
            rows.append((split, slot, value, int(ids[0])))
    expected = [
        (row["split"], int(row["slot"]), row["value"], int(row["token_id"]))
        for row in contract["resolved_value_token_rows"]
    ]
    assert rows == expected

def test_generated_configs_match_repository_cpu_yaml_subset_loader() -> None:
    """Generated preregistrations must be consumable by the runtime loader."""

    rich_v3 = _load_yaml(V3)
    cpu_v3 = load_config(V3)
    assert cpu_v3 == rich_v3
    assert cpu_v3["id"] == "LLM-QWEN-BINDING-ALGEBRA-003"

    rich_cr_v2 = _load_yaml(CR_V2)
    cpu_cr_v2 = load_config(CR_V2)
    assert cpu_cr_v2 == rich_cr_v2
    assert cpu_cr_v2["experiment_id"] == "QWEN-BINDING-ALGEBRA-CR-V2"

    for path in (V3, CR_V2):
        block_list_lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("- ")
        ]
        assert block_list_lines == []
