"""Tokenizer-amended, fail-closed Qwen binding-algebra Phase-0.

This module executes only calibration/train/validation.  It verifies the generated V3 token
contract with the pinned tokenizer *before model weights are loaded*.  Test/paraphrase episodes
are never materialized here and B2/B3/B4 remain unauthorized.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from causal_workspace_jepa.experiments.llm import qwen_binding_algebra_phase0 as v1

ROOT = Path(__file__).resolve().parents[4]
ALLOWED_SPLITS = v1.ALLOWED_SPLITS
FORBIDDEN_SPLITS = v1.FORBIDDEN_SPLITS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _contract_self_hash(payload: Mapping[str, Any]) -> str:
    copy_payload = dict(payload)
    copy_payload.pop("self_sha256", None)
    return _canonical_sha(copy_payload)


def _vocab_sha(tokenizer: Any) -> str:
    rows = sorted(
        (str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()
    )
    return _canonical_sha(rows)


def _strict_spaced_token_id(tokenizer: Any, value: str) -> int:
    text = " " + value
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"V3 answer value is not one token in the registered answer context: {value!r}"
        )
    token_id = int(ids[0])
    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise ValueError(
            f"V3 answer value fails exact leading-space decode roundtrip: {value!r} -> {decoded!r}"
        )
    return token_id


def _verify_generated_contract(
    *,
    bridge: Mapping[str, Any],
    parent_path: Path,
    parent: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    amendment = bridge["token_amendment"]
    contract_path = ROOT / amendment["resolved_contract"]
    cr_v2_path = ROOT / amendment["resolved_cr_extension"]
    source_v2_path = ROOT / "configs/experiments/qwen_binding_algebra_v2.yaml"
    if not contract_path.is_file() or not cr_v2_path.is_file():
        raise FileNotFoundError("resolved V3 token amendment artifacts are missing")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_self = str(contract.get("self_sha256", ""))
    observed_self = _contract_self_hash(contract)
    if expected_self != observed_self:
        raise RuntimeError("V3 token-contract self hash mismatch")
    if _sha256_file(parent_path) != contract["resolved_v3_sha256"]:
        raise RuntimeError("committed V3 YAML differs from token-contract hash")
    if _sha256_file(cr_v2_path) != contract["resolved_cr_v2_sha256"]:
        raise RuntimeError("committed CR-V2 YAML differs from token-contract hash")
    if _sha256_file(source_v2_path) != contract["source_v2_sha256"]:
        raise RuntimeError("source V2 YAML changed after tokenizer amendment")
    if parent.get("id") != "LLM-QWEN-BINDING-ALGEBRA-003":
        raise RuntimeError("Phase-0 V2 requires LLM-QWEN-BINDING-ALGEBRA-003")
    if parent.get("model") != contract["model"] or parent.get("revision") != contract["revision"]:
        raise RuntimeError("V3 model/revision differs from resolved token contract")
    if _vocab_sha(tokenizer) != contract["tokenizer_vocab_sha256"]:
        raise RuntimeError("pinned tokenizer vocabulary differs from resolved token contract")

    resolved_values = {
        split: [str(value) for value in parent["token_pools"]["values"][split]]
        for split in ("calibration", "train", "validation", "test")
    }
    if resolved_values != contract["resolved_values"]:
        raise RuntimeError("V3 value pools differ from resolved token contract")

    rows = []
    for split, values in resolved_values.items():
        for slot, value in enumerate(values):
            token_id = _strict_spaced_token_id(tokenizer, value)
            rows.append(
                {
                    "split": split,
                    "slot": slot,
                    "value": value,
                    "token_id": token_id,
                }
            )
    expected_rows = [
        {
            "split": row["split"],
            "slot": int(row["slot"]),
            "value": row["value"],
            "token_id": int(row["token_id"]),
        }
        for row in contract["resolved_value_token_rows"]
    ]
    if rows != expected_rows:
        raise RuntimeError("runtime V3 answer token ids differ from frozen token contract")
    return {
        "contract_path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
        "contract_self_sha256": expected_self,
        "tokenizer_vocab_sha256": contract["tokenizer_vocab_sha256"],
        "source_invalid_total": int(contract["source_invalid_total"]),
        "replacement_count": int(contract["replacement_count"]),
        "resolved_value_count": len(rows),
        "strict_spaced_single_token_pass": True,
        "protected_prompts_materialized": False,
        "model_weights_loaded_during_resolution": False,
    }


def _allowed_treatment_token_audit(
    tokenizer: Any, episodes: Sequence[Any], cases: Sequence[Any]
) -> dict[str, Any]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    failures: list[str] = []
    histogram: dict[str, int] = {}
    for case in cases:
        episode = episode_by_id[case.episode_id]
        clean = tokenizer(v1._clean_prompt(episode), add_special_tokens=True)["input_ids"]
        target = tokenizer(
            episode.prompt_after(case.target_permutation), add_special_tokens=True
        )["input_ids"]
        if len(clean) != len(target):
            failures.append(case.case_id + ":length")
            continue
        changed = sum(int(left != right) for left, right in zip(clean, target))
        support = sum(
            int(case.target_permutation[index] != index) for index in range(4)
        )
        histogram[str(changed)] = histogram.get(str(changed), 0) + 1
        if changed != support:
            failures.append(case.case_id + f":changed={changed}:support={support}")
    return {
        "case_count": len(cases),
        "changed_position_histogram": histogram,
        "failure_count": len(failures),
        "failures": failures[:100],
        "pass": not failures,
    }


def _write_output(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def run_phase0_v2(
    *,
    bridge_path: Path,
    parent_path: Path,
    output: Path,
    run_dir: Path,
    device_name: str,
    forward_batch: int,
    replay_batch: int,
    derivative_batch: int,
) -> dict[str, Any]:
    import torch

    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    parent = v1._load_parent_config(parent_path)
    authorization = bridge["scoped_authorization"]
    if not bool(authorization.get("execution_authorized")):
        raise RuntimeError("Bridge-002 Phase-0 execution authorization is disabled")
    if authorization.get("authorization_scope") != "B0_B1_only_on_calibration_train_validation":
        raise RuntimeError("Bridge-002 authorization scope changed")
    if authorization["allowed_splits"] != list(ALLOWED_SPLITS):
        raise RuntimeError("Bridge-002 allowed split roster changed")
    if authorization["forbidden_splits"] != list(FORBIDDEN_SPLITS):
        raise RuntimeError("Bridge-002 forbidden split roster changed")
    if not bool(authorization.get("phase0_only")):
        raise RuntimeError("Bridge-002 is not Phase-0-only")

    frozen_blob_checks: dict[str, Any] = {}
    for relative, expected_blob in bridge["frozen_parent_git_blobs"].items():
        actual_blob = v1._git("rev-parse", f"HEAD:{relative}")
        frozen_blob_checks[relative] = {
            "expected_git_blob": expected_blob,
            "actual_git_blob": actual_blob,
            "matches": actual_blob == expected_blob,
        }
        if actual_blob != expected_blob:
            raise RuntimeError(f"frozen V2/CR-V1 source changed: {relative}")

    protocol_guard = v1._assert_committed_and_pushed_protocol(
        str(bridge["expected_base_commit"]), bridge["required_committed_files"]
    )
    plan = v1.build_phase0_plan(parent)
    plan["schema_version"] = "qwen_binding_algebra_phase0_plan_v2"
    plan["parent_experiment_id"] = "LLM-QWEN-BINDING-ALGEBRA-003"
    plan["frozen_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    plan["plan_sha256"] = v1._json_sha(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    plan_path = run_dir / "phase0_plan_v2.json"
    _write_output(plan_path, plan)
    ledger = run_dir / "ACCESS_LEDGER.jsonl"
    v1._write_ledger(
        ledger,
        "PLAN_FROZEN",
        count=len(plan["cases"]),
        plan_sha256=plan["plan_sha256"],
        bridge="CRCT-QWEN-BRIDGE-002",
    )
    for split in ALLOWED_SPLITS:
        v1._write_ledger(
            ledger,
            "SPLIT_MATERIALIZED_NO_FORWARD",
            split=split,
            count=int(plan["case_counts"][split]),
        )

    status_payload: dict[str, Any] = {
        "schema_version": "qwen_binding_algebra_phase0_bridge_v2",
        "experiment_id": bridge["experiment_id"],
        "parent_experiment_id": bridge["parent_experiment_id"],
        "parent_cr_extension_id": bridge["parent_cr_extension_id"],
        "protocol_guard": {**protocol_guard, "frozen_parent_git_blobs": frozen_blob_checks},
        "phase0_plan_sha256": plan["plan_sha256"],
        "allowed_splits_materialized": list(ALLOWED_SPLITS),
        "model_forward_execution_started": False,
        "model_forward_splits_completed": [],
        "protected_splits_executed": [],
        "scientific_boundary": {
            "phase0_only": True,
            "b2_b3_b4_executed": False,
            "test_executed": False,
            "paraphrase_executed": False,
            "qk_mechanism_claim_allowed": False,
            "capital_dev_is_confirmation": False,
            "workspace_claim_permitted": False,
        },
    }

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        return {
            **status_payload,
            "status": "AVAILABILITY_BLOCKED",
            "reason": f"transformers: {exc}",
        }

    identity = v1._model_identity(parent)
    v1._write_ledger(
        ledger,
        "TOKENIZER_LOAD_STARTED",
        model=identity["name"],
        revision=identity["revision"],
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            identity["name"], revision=identity["revision"], local_files_only=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as exc:
        return {
            **status_payload,
            "status": "AVAILABILITY_BLOCKED",
            "reason": f"pinned tokenizer load failed: {type(exc).__name__}: {exc}",
        }
    v1._write_ledger(ledger, "TOKENIZER_LOAD_COMPLETE")

    episodes, cases = v1._generate_allowed_protocol(parent)
    try:
        contract = _verify_generated_contract(
            bridge=bridge,
            parent_path=parent_path,
            parent=parent,
            tokenizer=tokenizer,
        )
        treatment = _allowed_treatment_token_audit(tokenizer, episodes, cases)
        if not treatment["pass"]:
            raise RuntimeError("V3 allowed-split treatment token audit failed")
    except Exception as exc:
        blocked = {
            **status_payload,
            "status": "TOKEN_CONTRACT_BLOCKED_PRE_MODEL",
            "token_contract_error_type": type(exc).__name__,
            "token_contract_error": str(exc),
            "model_weights_loaded": False,
        }
        _write_output(output, blocked)
        v1._write_ledger(
            ledger,
            "TOKEN_CONTRACT_BLOCKED_PRE_MODEL",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return blocked
    status_payload["token_contract"] = contract
    status_payload["allowed_treatment_token_audit"] = treatment
    v1._write_ledger(
        ledger,
        "TOKEN_CONTRACT_VERIFIED_PRE_MODEL",
        count=int(contract["resolved_value_count"]),
        contract_self_sha256=contract["contract_self_sha256"],
    )

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return {**status_payload, "status": "AVAILABILITY_BLOCKED", "reason": "CUDA unavailable"}
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(bridge["runtime_seed"]))
    np.random.seed(int(bridge["runtime_seed"]))

    v1._write_ledger(
        ledger,
        "MODEL_LOAD_STARTED",
        model=identity["name"],
        revision=identity["revision"],
    )
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                identity["name"],
                revision=identity["revision"],
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                identity["name"],
                revision=identity["revision"],
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
        model = model.to(device)
        model.eval()
    except Exception as exc:
        return {
            **status_payload,
            "status": "AVAILABILITY_BLOCKED",
            "reason": f"pinned local model load failed: {type(exc).__name__}: {exc}",
        }
    v1._write_ledger(
        ledger,
        "MODEL_LOAD_COMPLETE",
        model=identity["name"],
        revision=identity["revision"],
    )

    started = time.perf_counter()
    status_payload["model_forward_execution_started"] = True
    v1._write_ledger(
        ledger,
        "B0_MODEL_FORWARD_EXECUTION_STARTED",
        allowed_splits=list(ALLOWED_SPLITS),
    )
    b0 = v1._b0_metrics(
        model,
        tokenizer,
        episodes,
        cases,
        bridge=bridge,
        device=device,
        max_length=identity["max_sequence_length"],
        forward_batch=forward_batch,
        replay_batch=replay_batch,
    )
    status_payload["model_forward_splits_completed"] = list(ALLOWED_SPLITS)
    v1._write_ledger(
        ledger,
        "B0_MODEL_FORWARD_EXECUTION_COMPLETE",
        allowed_splits=list(ALLOWED_SPLITS),
    )
    status_payload["b0"] = b0

    if not b0["pass"]:
        status_payload["status"] = "INELIGIBLE_TASK_PHASE0"
        status_payload["b1"] = {"executed": False, "reason": "B0 competence/replay gate failed"}
    else:
        v1._write_ledger(ledger, "B0_PASSED_B1_AUTHORIZED", split="validation")
        b1 = v1._b1_metrics(
            model,
            tokenizer,
            episodes,
            cases,
            bridge=bridge,
            device=device,
            max_length=identity["max_sequence_length"],
            derivative_batch=derivative_batch,
            forward_batch=forward_batch,
            ledger=ledger,
        )
        status_payload["b1"] = b1
        status_payload["status"] = v1.phase0_scientific_decision(
            b0_pass=True,
            derivative_available=bool(b1["derivative_available"]),
            interaction_power=(
                b1.get("composition_interaction", {}).get("interaction_power_fraction")
                if b1.get("derivative_available")
                else None
            ),
            quadratic_nmse=(
                b1.get("quadratic_nmse") if b1.get("derivative_available") else None
            ),
            interaction_min=float(
                bridge["phase0_thresholds"]["composition_interaction_power_min"]
            ),
            quadratic_nmse_min=float(bridge["phase0_thresholds"]["quadratic_nmse_min"]),
        )

    status_payload["runtime"] = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if device.type == "cuda":
        status_payload["runtime"].update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    status_payload["protected_splits_executed"] = []
    status_payload["result_sha256"] = v1._json_sha(status_payload)
    _write_output(output, status_payload)
    v1._write_ledger(ledger, "PHASE0_V2_COMPLETE", status=status_payload["status"])
    return status_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-batch", type=int, default=16)
    parser.add_argument("--replay-batch", type=int, default=8)
    parser.add_argument("--derivative-batch", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = run_phase0_v2(
            bridge_path=args.bridge_config,
            parent_path=args.parent_config,
            output=args.output,
            run_dir=args.run_dir,
            device_name=args.device,
            forward_batch=args.forward_batch,
            replay_batch=args.replay_batch,
            derivative_batch=args.derivative_batch,
        )
        if not args.output.exists():
            _write_output(args.output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "output": str(args.output),
                    "model_forward_splits_completed": payload.get(
                        "model_forward_splits_completed", []
                    ),
                    "protected_splits_executed": payload.get(
                        "protected_splits_executed", []
                    ),
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "qwen_binding_algebra_phase0_bridge_v2",
            "status": "INFRASTRUCTURE_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "protected_splits_executed": [],
        }
        _write_output(args.output, failure)
        print(json.dumps(failure, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
