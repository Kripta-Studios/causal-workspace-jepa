from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ALLOWED_SPLIT = "calibration"
FORBIDDEN_SPLITS = ("train", "validation", "test", "paraphrase")


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def _json_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_ledger(path: Path, stage: str, **extra: Any) -> None:
    import datetime as dt

    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "split": ALLOWED_SPLIT,
        "protected_split": False,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _strict_spaced_id(tokenizer: Any, value: str) -> int:
    ids = tokenizer(" " + value, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"value is not strict spaced single-token: {value!r}")
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != " " + value:
        raise ValueError(f"value does not round-trip as strict spaced token: {value!r}")
    return int(ids[0])


def _legacy_prompt(keys: Sequence[str], values: Sequence[str], query_index: int) -> str:
    return (
        "Use the four mappings. Reply with only the value.\n"
        f"- {keys[0]} -> {values[0]}\n"
        f"- {keys[1]} -> {values[1]}\n"
        f"- {keys[2]} -> {values[2]}\n"
        f"- {keys[3]} -> {values[3]}\n"
        f"Query: {keys[query_index]} ->"
    )


def _plain_prompt(keys: Sequence[str], values: Sequence[str], query_index: int) -> str:
    return (
        "Lookup task. Return exactly one table value and nothing else.\n"
        "Table:\n"
        f"{keys[0]}: {values[0]}\n"
        f"{keys[1]}: {values[1]}\n"
        f"{keys[2]}: {values[2]}\n"
        f"{keys[3]}: {values[3]}\n"
        f"Question: What value is paired with {keys[query_index]}?\n"
        "Answer:"
    )


def _chat_user(keys: Sequence[str], values: Sequence[str], query_index: int) -> str:
    return (
        "Read the lookup table and answer the question.\n"
        f"{keys[0]} = {values[0]}\n"
        f"{keys[1]} = {values[1]}\n"
        f"{keys[2]} = {values[2]}\n"
        f"{keys[3]} = {values[3]}\n"
        f"Which value is paired with {keys[query_index]}?\n"
        "Return exactly one table value with no explanation."
    )


def _chat_prefill(
    tokenizer: Any,
    keys: Sequence[str],
    values: Sequence[str],
    query_index: int,
    *,
    fewshot: bool,
) -> str:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You solve literal lookup-table questions. "
                "Return only the requested value and do not explain."
            ),
        }
    ]
    if fewshot:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        "Read the lookup table and answer the question.\n"
                        "oak = red\npine = blue\nbirch = green\ncedar = gold\n"
                        "Which value is paired with birch?\n"
                        "Return exactly one table value with no explanation."
                    ),
                },
                {"role": "assistant", "content": "Answer: green"},
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": _chat_user(keys, values, query_index)},
            {"role": "assistant", "content": "Answer:"},
        ]
    )
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "pinned tokenizer/Transformers does not support the preregistered "
            "non-thinking assistant-prefill renderer"
        ) from exc
    if not rendered.endswith("Answer:"):
        raise RuntimeError("chat-prefill renderer did not preserve final assistant prefill")
    return rendered


def _render(
    variant: str,
    tokenizer: Any,
    keys: Sequence[str],
    values: Sequence[str],
    query_index: int,
) -> tuple[str, bool]:
    if variant == "legacy":
        return _legacy_prompt(keys, values, query_index), True
    if variant == "plain_explicit":
        return _plain_prompt(keys, values, query_index), True
    if variant == "chat_prefill":
        return _chat_prefill(tokenizer, keys, values, query_index, fewshot=False), False
    if variant == "chat_prefill_fewshot":
        return _chat_prefill(tokenizer, keys, values, query_index, fewshot=True), False
    raise ValueError(f"unknown prompt variant kind: {variant}")


@dataclass(frozen=True)
class Row:
    prompt: str
    expected_id: int
    candidate_ids: tuple[int, int, int, int]
    add_special_tokens: bool


def _build_calibration_rows(
    parent: Mapping[str, Any],
    tokenizer: Any,
    *,
    variant_kind: str,
) -> tuple[list[Row], list[Row], dict[str, Any]]:
    from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
        apply_permutation,
        generate_binding_algebra_episodes,
        permutation_changes_slot,
        permutations_in_classes,
    )

    split_cfg = parent["splits"][ALLOWED_SPLIT]
    pools = parent["token_pools"]
    episodes = generate_binding_algebra_episodes(
        split=ALLOWED_SPLIT,
        keys=pools["keys"][ALLOWED_SPLIT],
        values=pools["values"][ALLOWED_SPLIT],
        count=int(split_cfg["count"]),
        seed=int(split_cfg["seed"]),
        template=str(split_cfg.get("template", "primary")),
    )
    clean_rows: list[Row] = []
    direct_rows: list[Row] = []
    for episode in episodes:
        candidate_ids = tuple(_strict_spaced_id(tokenizer, value) for value in episode.base_values)
        clean_prompt, clean_add_special = _render(
            variant_kind,
            tokenizer,
            episode.keys,
            episode.base_values,
            episode.query_index,
        )
        clean_rows.append(
            Row(
                prompt=clean_prompt,
                expected_id=_strict_spaced_id(tokenizer, episode.clean_answer),
                candidate_ids=candidate_ids,  # type: ignore[arg-type]
                add_special_tokens=clean_add_special,
            )
        )
        actions = [
            action
            for action in permutations_in_classes(parent["action_partition"]["train_classes"])
            if permutation_changes_slot(action, episode.query_index)
        ]
        for action in actions:
            permuted = apply_permutation(episode.base_values, action)
            direct_prompt, direct_add_special = _render(
                variant_kind,
                tokenizer,
                episode.keys,
                permuted,
                episode.query_index,
            )
            direct_rows.append(
                Row(
                    prompt=direct_prompt,
                    expected_id=_strict_spaced_id(tokenizer, permuted[episode.query_index]),
                    candidate_ids=candidate_ids,  # type: ignore[arg-type]
                    add_special_tokens=direct_add_special,
                )
            )
    meta = {
        "episode_count": len(episodes),
        "clean_row_count": len(clean_rows),
        "direct_row_count": len(direct_rows),
    }
    return clean_rows, direct_rows, meta


def _evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Row],
    *,
    device: Any,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    import torch

    total = 0
    full_correct = 0
    candidate_correct = 0
    margins: list[float] = []
    top1_counts: dict[str, int] = {}
    prompt_token_lengths: list[int] = []
    for start in range(0, len(rows), batch_size):
        block = rows[start : start + batch_size]
        prompts = [row.prompt for row in block]
        special_flags = {row.add_special_tokens for row in block}
        if len(special_flags) != 1:
            raise RuntimeError("mixed chat/plain tokenization modes inside one evaluation block")
        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=special_flags.pop(),
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            logits = model(**batch, use_cache=False, return_dict=True).logits
        last = batch["attention_mask"].long().sum(dim=1) - 1
        idx = torch.arange(len(block), device=device)
        final_logits = logits[idx, last]
        full_pred = final_logits.argmax(dim=1)
        for row_index, row in enumerate(block):
            expected = int(row.expected_id)
            candidates = torch.tensor(row.candidate_ids, device=device, dtype=torch.long)
            candidate_logits = final_logits[row_index, candidates]
            best_pos = int(candidate_logits.argmax().item())
            candidate_pred = int(candidates[best_pos].item())
            wrong = candidates[candidates != expected]
            best_wrong = float(final_logits[row_index, wrong].max().item())
            expected_logit = float(final_logits[row_index, expected].item())
            margins.append(expected_logit - best_wrong)
            pred_id = int(full_pred[row_index].item())
            decoded = tokenizer.decode(
                [pred_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            top1_counts[decoded] = top1_counts.get(decoded, 0) + 1
            full_correct += int(pred_id == expected)
            candidate_correct += int(candidate_pred == expected)
        prompt_token_lengths.extend(int(x) for x in batch["attention_mask"].sum(dim=1).tolist())
        total += len(block)
    return {
        "count": total,
        "full_vocab_accuracy": full_correct / max(total, 1),
        "candidate_only_accuracy_diagnostic": candidate_correct / max(total, 1),
        "expected_vs_best_wrong_candidate_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "expected_vs_best_wrong_candidate_margin_median": (
            float(np.median(margins)) if margins else 0.0
        ),
        "prompt_tokens_min": min(prompt_token_lengths) if prompt_token_lengths else 0,
        "prompt_tokens_max": max(prompt_token_lengths) if prompt_token_lengths else 0,
        "top1_token_counts": dict(
            sorted(top1_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        ),
    }


def _assert_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    if config["allowed_model_forward_splits"] != [ALLOWED_SPLIT]:
        raise RuntimeError("competence recovery must be calibration-only")
    if tuple(config["forbidden_model_forward_splits"]) != FORBIDDEN_SPLITS:
        raise RuntimeError("forbidden split roster changed")
    parent_checks = {}
    for relative, spec in config["parent_files"].items():
        actual = _git("rev-parse", f"HEAD:{relative}")
        expected = str(spec["git_blob"])
        parent_checks[relative] = {
            "actual_git_blob": actual,
            "expected_git_blob": expected,
            "matches": actual == expected,
        }
        if actual != expected:
            raise RuntimeError(f"frozen parent changed: {relative}")
    tracked_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    refs = [
        line.strip()
        for line in _git("branch", "-r", "--contains", "HEAD").splitlines()
        if line.strip().startswith("origin/")
    ]
    if tracked_dirty:
        raise RuntimeError("tracked worktree must be clean before calibration forwards")
    if not refs:
        raise RuntimeError("competence-recovery HEAD must be pushed to origin before forwards")
    return {
        "parent_checks": parent_checks,
        "tracked_worktree_clean": True,
        "origin_refs_containing_head": refs,
        "head_is_pushed_to_origin": True,
    }


def run(
    *,
    config_path: Path,
    output: Path,
    run_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from causal_workspace_jepa.common.config import load_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(config_path.read_text(encoding="utf-8"))
    guard = _assert_protocol(config)
    parent = load_config(ROOT / "configs/experiments/qwen_binding_algebra_v3.yaml")
    model_cfg = config["model"]
    ledger = run_dir / "ACCESS_LEDGER.jsonl"
    _write_ledger(
        ledger,
        "RECOVERY_PLAN_FROZEN",
        prompt_variant_ids=[item["id"] for item in config["prompt_variants"]],
        allowed_split=ALLOWED_SPLIT,
        forbidden_splits=list(FORBIDDEN_SPLITS),
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return {
            "status": "AVAILABILITY_BLOCKED",
            "reason": "CUDA unavailable",
            "protected_splits_executed": [],
        }

    _write_ledger(ledger, "TOKENIZER_LOAD_STARTED")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=model_cfg["revision"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _write_ledger(ledger, "TOKENIZER_LOAD_COMPLETE")

    # Build every prompt variant from calibration only before loading model weights.
    variant_rows: dict[str, tuple[list[Row], list[Row], dict[str, Any]]] = {}
    unavailable: dict[str, str] = {}
    for item in config["prompt_variants"]:
        try:
            variant_rows[item["id"]] = _build_calibration_rows(
                parent,
                tokenizer,
                variant_kind=item["kind"],
            )
        except Exception as exc:
            unavailable[item["id"]] = f"{type(exc).__name__}: {exc}"
    _write_ledger(
        ledger,
        "CALIBRATION_PROMPTS_MATERIALIZED_NO_FORWARD",
        available_variants=sorted(variant_rows),
        unavailable_variants=unavailable,
    )

    _write_ledger(ledger, "MODEL_LOAD_STARTED")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            revision=model_cfg["revision"],
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation=model_cfg["attn_implementation"],
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            revision=model_cfg["revision"],
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation=model_cfg["attn_implementation"],
        )
    model = model.to(device)
    model.eval()
    _write_ledger(ledger, "MODEL_LOAD_COMPLETE")

    max_length = int(parent.get("max_sequence_length", 96))
    results: dict[str, Any] = {}
    for item in config["prompt_variants"]:
        variant_id = item["id"]
        if variant_id not in variant_rows:
            results[variant_id] = {
                "status": "UNAVAILABLE",
                "reason": unavailable[variant_id],
                "priority": int(item["priority"]),
            }
            continue
        clean_rows, direct_rows, meta = variant_rows[variant_id]
        _write_ledger(
            ledger,
            "CALIBRATION_FORWARD_STARTED",
            prompt_variant_id=variant_id,
            clean_count=len(clean_rows),
            direct_count=len(direct_rows),
        )
        clean = _evaluate_rows(
            model,
            tokenizer,
            clean_rows,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        direct = _evaluate_rows(
            model,
            tokenizer,
            direct_rows,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        _write_ledger(
            ledger,
            "CALIBRATION_FORWARD_COMPLETE",
            prompt_variant_id=variant_id,
            clean_count=len(clean_rows),
            direct_count=len(direct_rows),
        )
        results[variant_id] = {
            "status": "EVALUATED",
            "priority": int(item["priority"]),
            "kind": item["kind"],
            "rows": meta,
            "clean": clean,
            "direct_permuted": direct,
        }

    selection_cfg = config["selection"]
    clean_floor = float(selection_cfg["clean_full_vocab_accuracy_min"])
    direct_floor = float(selection_cfg["direct_permuted_full_vocab_accuracy_min"])
    eligible = []
    for variant_id, result in results.items():
        if result.get("status") != "EVALUATED":
            continue
        clean_acc = float(result["clean"]["full_vocab_accuracy"])
        direct_acc = float(result["direct_permuted"]["full_vocab_accuracy"])
        result["eligible"] = bool(clean_acc >= clean_floor and direct_acc >= direct_floor)
        result["primary_score"] = min(clean_acc, direct_acc)
        result["secondary_score"] = 0.5 * (clean_acc + direct_acc)
        if result["eligible"]:
            eligible.append(
                (
                    -result["primary_score"],
                    -result["secondary_score"],
                    int(result["priority"]),
                    variant_id,
                )
            )

    selected = None
    if eligible:
        eligible.sort()
        selected = eligible[0][3]
        status = "COMPETENCE_RECOVERY_PROMPT_SELECTED"
    else:
        status = "COMPETENCE_RECOVERY_FAILED"

    payload = {
        "schema_version": "qwen_binding_competence_recovery_result_v1",
        "experiment_id": config["experiment_id"],
        "status": status,
        "protocol_guard": guard,
        "allowed_model_forward_splits": [ALLOWED_SPLIT],
        "model_forward_splits_executed": [ALLOWED_SPLIT],
        "forbidden_model_forward_splits": list(FORBIDDEN_SPLITS),
        "protected_splits_executed": [],
        "train_executed": False,
        "validation_executed": False,
        "test_executed": False,
        "paraphrase_executed": False,
        "selection_rule": selection_cfg,
        "prompt_variants": results,
        "selected_prompt_variant_id": selected,
        "scientific_boundary": config["scientific_boundary"],
        "model": model_cfg,
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
    }
    payload["result_sha256"] = _json_sha(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_ledger(ledger, "COMPETENCE_RECOVERY_COMPLETE", status=status, selected=selected)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = run(
            config_path=args.config,
            output=args.output,
            run_dir=args.run_dir,
            device_name=args.device,
            batch_size=args.batch_size,
        )
        print(json.dumps(
            {
                "status": payload["status"],
                "selected_prompt_variant_id": payload.get("selected_prompt_variant_id"),
                "protected_splits_executed": payload.get("protected_splits_executed", []),
            },
            indent=2,
        ))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "qwen_binding_competence_recovery_result_v1",
            "status": "INFRASTRUCTURE_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "protected_splits_executed": [],
            "train_executed": False,
            "validation_executed": False,
            "test_executed": False,
            "paraphrase_executed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": failure["status"], "error": failure["error"]}, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
