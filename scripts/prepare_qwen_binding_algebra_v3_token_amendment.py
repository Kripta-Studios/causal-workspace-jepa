from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/experiments/qwen_binding_algebra_v3_token_amendment.json"
DEFAULT_V2 = ROOT / "configs/experiments/qwen_binding_algebra_v2.yaml"
DEFAULT_CR_V1 = ROOT / "configs/experiments/qwen_binding_algebra_cr_v1.yaml"
DEFAULT_V3 = ROOT / "configs/experiments/qwen_binding_algebra_v3.yaml"
DEFAULT_CR_V2 = ROOT / "configs/experiments/qwen_binding_algebra_cr_v2.yaml"
DEFAULT_CONTRACT = ROOT / "configs/experiments/qwen_binding_algebra_v3_token_contract.json"
SPLIT_ORDER = ("calibration", "train", "validation", "test")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _self_hash(payload: Mapping[str, Any], field: str = "self_sha256") -> str:
    copy_payload = dict(payload)
    copy_payload.pop(field, None)
    return _canonical_sha(copy_payload)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


class _CpuSubsetDumper(yaml.SafeDumper):
    """Emit sequence values in the inline form accepted by common.config.load_config."""


def _represent_flow_sequence(
    dumper: yaml.SafeDumper, data: list[Any]
) -> yaml.nodes.SequenceNode:
    node = dumper.represent_list(data)
    node.flow_style = True
    return node


_CpuSubsetDumper.add_representer(list, _represent_flow_sequence)


def _dump_yaml(payload: Mapping[str, Any]) -> bytes:
    """Serialize the preregistration in the repository's deterministic YAML subset.

    ``common.config.load_config`` intentionally rejects block-list syntax. PyYAML's
    default safe dumper emits lists as ``- item`` blocks, so generated configs must
    force every simple sequence into flow style: ``[item, ...]``.
    """

    text = yaml.dump(
        dict(payload),
        Dumper=_CpuSubsetDumper,
        sort_keys=False,
        allow_unicode=True,
        width=100000,
        default_flow_style=False,
    )
    return text.encode("utf-8")


def _strict_spaced_token(tokenizer: Any, value: str) -> tuple[int, str] | None:
    text = " " + value
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        return None
    token_id = int(ids[0])
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        return None
    return token_id, decoded


def _vocab_sha(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    rows = sorted((str(token), int(token_id)) for token, token_id in vocab.items())
    return _canonical_sha(rows)


def _source_invalid_values(
    tokenizer: Any, values: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in SPLIT_ORDER:
        result[split] = [
            str(value)
            for value in values[split]
            if _strict_spaced_token(tokenizer, str(value)) is None
        ]
    return result


def _all_casefold_strings(parent: Mapping[str, Any]) -> set[str]:
    pools = parent["token_pools"]
    strings = {
        str(value).casefold()
        for role in ("keys", "values")
        for split_values in pools[role].values()
        for value in split_values
    }
    return strings


def _candidate_is_well_formed(candidate: str, *, calibration: bool) -> bool:
    if calibration:
        return bool(re.fullmatch(r"[a-z]{3,16}", candidate))
    return bool(re.fullmatch(r"[A-Z][a-z]{2,20}", candidate))


def resolve_value_pools(
    *,
    tokenizer: Any,
    source_parent: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, Any]]:
    """Resolve only tokenizer-ineligible value slots using committed deterministic rosters."""

    source_values = source_parent["token_pools"]["values"]
    observed_invalid = _source_invalid_values(tokenizer, source_values)
    expected_invalid = {
        split: [str(value) for value in spec["expected_source_invalid_values"][split]]
        for split in SPLIT_ORDER
    }
    if observed_invalid != expected_invalid:
        raise RuntimeError(
            "pinned tokenizer invalid-value roster differs from the preregistered amendment: "
            f"observed={observed_invalid!r} expected={expected_invalid!r}"
        )
    invalid_total = sum(len(values) for values in observed_invalid.values())
    if invalid_total != int(spec["expected_source_invalid_total"]):
        raise RuntimeError("source invalid-value count differs from amendment contract")

    used = _all_casefold_strings(source_parent)
    resolved: dict[str, list[str]] = {}
    replacements: list[dict[str, Any]] = []
    rosters = spec["candidate_rosters"]

    for split in SPLIT_ORDER:
        resolved[split] = []
        calibration = split == "calibration"
        roster = rosters["calibration_colors" if calibration else "cities"]
        for slot, original in enumerate(source_values[split]):
            original = str(original)
            strict = _strict_spaced_token(tokenizer, original)
            if strict is not None:
                resolved[split].append(original)
                continue

            replacement: str | None = None
            replacement_id: int | None = None
            roster_index: int | None = None
            for index, candidate_raw in enumerate(roster):
                candidate = str(candidate_raw)
                if not _candidate_is_well_formed(candidate, calibration=calibration):
                    continue
                if candidate.casefold() in used:
                    continue
                strict_candidate = _strict_spaced_token(tokenizer, candidate)
                if strict_candidate is None:
                    continue
                replacement = candidate
                replacement_id = int(strict_candidate[0])
                roster_index = index
                break
            if replacement is None or replacement_id is None or roster_index is None:
                raise RuntimeError(
                    f"candidate roster exhausted while repairing {split}[{slot}]={original!r}"
                )
            used.add(replacement.casefold())
            resolved[split].append(replacement)
            replacements.append(
                {
                    "split": split,
                    "slot": slot,
                    "original": original,
                    "replacement": replacement,
                    "replacement_token_id": replacement_id,
                    "candidate_roster": "calibration_colors" if calibration else "cities",
                    "candidate_roster_index": roster_index,
                }
            )

    final_rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        for slot, value in enumerate(resolved[split]):
            strict = _strict_spaced_token(tokenizer, value)
            if strict is None:
                raise RuntimeError(
                    f"resolved value violates strict answer-token contract: {value!r}"
                )
            final_rows.append(
                {
                    "split": split,
                    "slot": slot,
                    "value": value,
                    "token_id": int(strict[0]),
                    "encoded_text": strict[1],
                }
            )

    if len({row["value"].casefold() for row in final_rows}) != len(final_rows):
        raise RuntimeError("resolved value pools are not globally casefold-unique")
    if len(replacements) != int(spec["expected_source_invalid_total"]):
        raise RuntimeError("replacement count does not equal the frozen source-invalid count")

    audit = {
        "observed_source_invalid_values": observed_invalid,
        "source_invalid_total": invalid_total,
        "resolved_value_count": len(final_rows),
        "replacement_count": len(replacements),
        "all_resolved_values_strict_spaced_single_token": True,
    }
    return resolved, replacements, {"rows": final_rows, **audit}


def _replace_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, old, new) for key, item in value.items()}
    return value


def build_v3_parent(
    source: Mapping[str, Any], resolved_values: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(source))
    payload = _replace_strings(payload, "qwen_binding_algebra_v2", "qwen_binding_algebra_v3")
    payload["id"] = "LLM-QWEN-BINDING-ALGEBRA-003"
    payload["status"] = "PREREGISTERED_TOKENIZER_AMENDMENT_OUTCOME_BLIND_NOT_RUN"
    payload["registered_date"] = "2026-08-17"
    payload["supersedes"] = "LLM-QWEN-BINDING-ALGEBRA-002"
    payload["amendment_reason"] = (
        "tokenizer-only repair of answer-value pools after V2 was found ineligible before the "
        "first competence forward; all scientific thresholds/actions/splits/seeds remain frozen"
    )
    payload["claim_boundary"] = (
        "causal permutation-operator test after tokenizer-only eligibility amendment; not a "
        "circuit, J-space, workspace, SOTA, or consciousness claim"
    )
    payload["token_pools"]["values"] = {
        split: list(resolved_values[split]) for split in SPLIT_ORDER
    }
    payload["token_contract"] = {
        "contract_path": "configs/experiments/qwen_binding_algebra_v3_token_contract.json",
        "answer_encoding": "single token for literal leading-space-plus-value",
        "bare_only_encoding_forbidden": True,
        "keys_unchanged_from_v2": True,
        "values_changed_only_where_v2_failed_pinned_tokenizer_contract": True,
        "tokenizer_only_selection_before_model_forward": True,
    }
    return payload


def build_cr_v2(
    source: Mapping[str, Any],
    *,
    resolved_values: Mapping[str, Sequence[str]],
    v3_sha256: str,
    v3_semantic_sha256: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(source))
    payload = _replace_strings(payload, "qwen_binding_algebra_cr_v1", "qwen_binding_algebra_cr_v2")
    payload["id"] = "QWEN-BINDING-ALGEBRA-CR-V2"
    payload["experiment_id"] = "QWEN-BINDING-ALGEBRA-CR-V2"
    payload["status"] = "PREREGISTERED_TOKENIZER_AMENDMENT_OUTCOME_BLIND_NOT_AUTHORIZED"
    payload["registered_date"] = "2026-08-17"
    payload["amendment_reason"] = (
        "tokenizer-only answer-value amendment inherited from LLM-QWEN-BINDING-ALGEBRA-003; "
        "no model output was used to choose replacements"
    )
    payload["parent_linkage"] = dict(payload["parent_linkage"])
    payload["parent_linkage"].update(
        {
            "parent_experiment_id": "LLM-QWEN-BINDING-ALGEBRA-003",
            "parent_config": "configs/experiments/qwen_binding_algebra_v3.yaml",
            "parent_config_sha256": v3_sha256,
            "parent_semantic_sha256": v3_semantic_sha256,
            "relationship": "tokenizer_contract_amendment",
            "exact_parent_required": True,
            "parent_config_may_be_modified": False,
        }
    )
    payload["token_pools"]["values"] = {
        split: list(resolved_values[split]) for split in SPLIT_ORDER
    }
    payload["token_contract"] = {
        "contract_path": "configs/experiments/qwen_binding_algebra_v3_token_contract.json",
        "answer_encoding": "single token for literal leading-space-plus-value",
        "bare_only_encoding_forbidden": True,
        "keys_unchanged_from_v2": True,
        "values_changed_only_where_v2_failed_pinned_tokenizer_contract": True,
        "tokenizer_only_selection_before_model_forward": True,
        "protected_prompts_or_model_outputs_used_for_selection": False,
    }
    payload.pop("canonical_digest", None)
    payload["canonical_digest"] = _canonical_sha(payload)
    return payload


def prepare_amendment(
    *,
    spec_path: Path,
    source_v2_path: Path,
    source_cr_v1_path: Path,
    output_v3_path: Path,
    output_cr_v2_path: Path,
    output_contract_path: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_v2 = _load_yaml(source_v2_path)
    source_cr_v1 = _load_yaml(source_cr_v1_path)

    for relative, record in spec["source_files"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen source file missing: {relative}")
        observed = _sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(
                f"frozen source sha256 mismatch for {relative}: {observed} != {record['sha256']}"
            )

    if source_v2.get("id") != spec["source_parent_experiment_id"]:
        raise RuntimeError("source V2 experiment id differs from amendment spec")
    if source_cr_v1.get("experiment_id") != spec["source_cr_experiment_id"]:
        raise RuntimeError("source CR-V1 experiment id differs from amendment spec")
    if source_v2.get("model") != spec["model"] or source_v2.get("revision") != spec["revision"]:
        raise RuntimeError("source V2 model/revision differs from tokenizer amendment")

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        local_files_only=True,
    )
    resolved_values, replacements, audit = resolve_value_pools(
        tokenizer=tokenizer,
        source_parent=source_v2,
        spec=spec,
    )

    v3 = build_v3_parent(source_v2, resolved_values)
    v3_bytes = _dump_yaml(v3)
    v3_sha = _sha256_bytes(v3_bytes)
    v3_semantic = _canonical_sha(v3)
    cr_v2 = build_cr_v2(
        source_cr_v1,
        resolved_values=resolved_values,
        v3_sha256=v3_sha,
        v3_semantic_sha256=v3_semantic,
    )
    cr_v2_bytes = _dump_yaml(cr_v2)
    cr_v2_sha = _sha256_bytes(cr_v2_bytes)

    output_v3_path.parent.mkdir(parents=True, exist_ok=True)
    output_v3_path.write_bytes(v3_bytes)
    output_cr_v2_path.write_bytes(cr_v2_bytes)

    contract: dict[str, Any] = {
        "schema_version": "qwen_binding_algebra_v3_token_contract_v1",
        "amendment_id": spec["amendment_id"],
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_depends_only_on_tokenizer_metadata": True,
        "model_weights_loaded": False,
        "model_outputs_or_logits_used": False,
        "protected_prompts_materialized": False,
        "model": spec["model"],
        "revision": spec["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": _vocab_sha(tokenizer),
        "amendment_spec_sha256": _sha256_file(spec_path),
        "source_v2_sha256": _sha256_file(source_v2_path),
        "source_cr_v1_sha256": _sha256_file(source_cr_v1_path),
        "resolved_v3_sha256": v3_sha,
        "resolved_v3_semantic_sha256": v3_semantic,
        "resolved_cr_v2_sha256": cr_v2_sha,
        "strict_answer_contract": spec["answer_token_contract"],
        "expected_source_invalid_values": spec["expected_source_invalid_values"],
        "observed_source_invalid_values": audit["observed_source_invalid_values"],
        "source_invalid_total": audit["source_invalid_total"],
        "replacement_count": audit["replacement_count"],
        "replacements": replacements,
        "resolved_values": resolved_values,
        "resolved_value_token_rows": audit["rows"],
        "all_resolved_values_strict_spaced_single_token": audit[
            "all_resolved_values_strict_spaced_single_token"
        ],
        "old_v2_and_cr_v1_must_remain_unchanged": True,
        "scientific_thresholds_actions_splits_seeds_changed": False,
    }
    contract["self_sha256"] = _self_hash(contract)
    output_contract_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--source-cr-v1", type=Path, default=DEFAULT_CR_V1)
    parser.add_argument("--output-v3", type=Path, default=DEFAULT_V3)
    parser.add_argument("--output-cr-v2", type=Path, default=DEFAULT_CR_V2)
    parser.add_argument("--output-contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = prepare_amendment(
        spec_path=args.spec,
        source_v2_path=args.source_v2,
        source_cr_v1_path=args.source_cr_v1,
        output_v3_path=args.output_v3,
        output_cr_v2_path=args.output_cr_v2,
        output_contract_path=args.output_contract,
    )
    print(
        json.dumps(
            {
                "status": "TOKENIZER_ONLY_AMENDMENT_RESOLVED",
                "source_invalid_total": payload["source_invalid_total"],
                "replacement_count": payload["replacement_count"],
                "contract_self_sha256": payload["self_sha256"],
                "v3_sha256": payload["resolved_v3_sha256"],
                "cr_v2_sha256": payload["resolved_cr_v2_sha256"],
                "generated_files": [
                    str(args.output_v3),
                    str(args.output_cr_v2),
                    str(args.output_contract),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
