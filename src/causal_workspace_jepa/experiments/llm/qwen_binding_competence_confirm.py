"""Fresh Qwen competence confirmation for the frozen chat-prefill renderer.

This experiment is not V3 and cannot rescue V3. It uses a new confirmation split
that is disjoint from calibration/train/validation/test/paraphrase. Protected
splits remain closed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import hashlib
import traceback
from typing import Any, Mapping, Sequence

from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
    apply_permutation,
    assert_globally_disjoint_token_pools,
    generate_binding_algebra_episodes,
    permutation_changes_slot,
    permutations_in_classes,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_competence_recovery import (
    ALLOWED_SPLIT as RECOVERY_SPLIT,
    FORBIDDEN_SPLITS as PARENT_FORBIDDEN,
    Row,
    _evaluate_rows,
    _render,
    _strict_spaced_id,
    _write_ledger,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIRM_SPLIT = "confirmation"
FORBIDDEN_SPLITS = PARENT_FORBIDDEN + (RECOVERY_SPLIT,)
FROZEN_RENDERER = "qwen_chat_prefill_v1"
FROZEN_RENDERER_KIND = "chat_prefill"


def _json_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_parent_pools(parent_pools: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    return {
        role: {split: list(values) for split, values in role_pools.items()}
        for role, role_pools in parent_pools.items()
    }


def assert_confirmation_disjoint(
    parent_pools: Mapping[str, Any],
    confirmation_keys: Sequence[str],
    confirmation_values: Sequence[str],
) -> None:
    merged = load_parent_pools(parent_pools)
    if CONFIRM_SPLIT in merged.get("keys", {}) or CONFIRM_SPLIT in merged.get("values", {}):
        raise RuntimeError("parent protocol must not already own a confirmation split")
    merged.setdefault("keys", {})[CONFIRM_SPLIT] = list(confirmation_keys)
    merged.setdefault("values", {})[CONFIRM_SPLIT] = list(confirmation_values)
    assert_globally_disjoint_token_pools(merged)


def build_confirmation_rows(
    parent_pools: Mapping[str, Any],
    tokenizer: Any,
    *,
    keys: Sequence[str],
    values: Sequence[str],
    count: int,
    seed: int,
    train_classes: Sequence[str],
) -> tuple[list[Row], list[Row], dict[str, Any]]:
    assert_confirmation_disjoint(parent_pools, keys, values)
    episodes = generate_binding_algebra_episodes(
        split=CONFIRM_SPLIT,
        keys=keys,
        values=values,
        count=int(count),
        seed=int(seed),
        template="primary",
    )
    clean_rows: list[Row] = []
    direct_rows: list[Row] = []
    identities: list[dict[str, Any]] = []
    for episode in episodes:
        candidate_ids = tuple(_strict_spaced_id(tokenizer, value) for value in episode.base_values)
        clean_prompt, clean_add_special = _render(
            FROZEN_RENDERER_KIND,
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
        identities.append(
            {
                "episode_id": episode.episode_id,
                "keys": list(episode.keys),
                "values": list(episode.base_values),
                "query_index": episode.query_index,
            }
        )
        actions = [
            action
            for action in permutations_in_classes(train_classes)
            if permutation_changes_slot(action, episode.query_index)
        ]
        for action in actions:
            permuted = apply_permutation(episode.base_values, action)
            direct_prompt, direct_add_special = _render(
                FROZEN_RENDERER_KIND,
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
        "episode_identities": identities,
        "split": CONFIRM_SPLIT,
        "renderer": FROZEN_RENDERER,
    }
    return clean_rows, direct_rows, meta


def _assert_protocol(config: Mapping[str, Any]) -> None:
    if config["allowed_model_forward_splits"] != [CONFIRM_SPLIT]:
        raise RuntimeError("confirmation must be confirmation-split only")
    if list(config["forbidden_model_forward_splits"]) != list(FORBIDDEN_SPLITS):
        raise RuntimeError("confirmation forbidden-split list is frozen")
    if config["renderer"]["id"] != FROZEN_RENDERER:
        raise RuntimeError("confirmation renderer is frozen to qwen_chat_prefill_v1")
    if float(config["gates"]["clean_full_vocab_accuracy_min"]) != 0.9:
        raise RuntimeError("clean full-vocabulary gate is frozen at 0.90")
    if float(config["gates"]["direct_permuted_full_vocab_accuracy_min"]) != 0.9:
        raise RuntimeError("direct-permuted full-vocabulary gate is frozen at 0.90")
    if config["gates"]["candidate_only_accuracy_is_diagnostic_only"] is not True:
        raise RuntimeError("candidate-only accuracy cannot be an eligibility metric")
    if CONFIRM_SPLIT in set(config["forbidden_model_forward_splits"]):
        raise RuntimeError("confirmation split cannot also be forbidden")
    if (
        "test" in config["allowed_model_forward_splits"]
        or "paraphrase" in config["allowed_model_forward_splits"]
    ):
        raise RuntimeError("protected splits cannot be confirmation forwards")


def run_confirmation(config: Mapping[str, Any], *, ledger_path: Path) -> dict[str, Any]:
    _assert_protocol(config)
    parent_pools = config["parent_token_pools"]
    keys = list(config["token_pools"]["keys"][CONFIRM_SPLIT])
    values = list(config["token_pools"]["values"][CONFIRM_SPLIT])
    _write_ledger(
        ledger_path,
        "CONFIRMATION_PLAN_FROZEN",
        allowed_split=CONFIRM_SPLIT,
        forbidden_splits=list(FORBIDDEN_SPLITS),
        renderer=FROZEN_RENDERER,
        protected_split=False,
        split=CONFIRM_SPLIT,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=model_cfg["revision"],
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    clean_rows, direct_rows, meta = build_confirmation_rows(
        parent_pools,
        tokenizer,
        keys=keys,
        values=values,
        count=int(config["split"]["count"]),
        seed=int(config["split"]["seed"]),
        train_classes=list(config["action_partition"]["train_classes"]),
    )
    _write_ledger(
        ledger_path,
        "CONFIRMATION_PROMPTS_MATERIALIZED_NO_FORWARD",
        split=CONFIRM_SPLIT,
        protected_split=False,
        episode_count=meta["episode_count"],
    )
    device = torch.device(str(config["execution"]["device"]))
    _write_ledger(ledger_path, "MODEL_LOAD_STARTED", split=CONFIRM_SPLIT, protected_split=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg["revision"],
        torch_dtype=torch.float32,
        attn_implementation=str(model_cfg.get("attn_implementation", "eager")),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    model.to(device)
    model.eval()
    _write_ledger(ledger_path, "MODEL_LOAD_COMPLETE", split=CONFIRM_SPLIT, protected_split=False)
    _write_ledger(
        ledger_path,
        "CONFIRMATION_FORWARD_STARTED",
        split=CONFIRM_SPLIT,
        protected_split=False,
        clean_count=len(clean_rows),
        direct_count=len(direct_rows),
        prompt_variant_id=FROZEN_RENDERER,
    )
    clean = _evaluate_rows(
        model,
        tokenizer,
        clean_rows,
        device=device,
        batch_size=int(config["execution"]["batch_size"]),
        max_length=int(config["execution"].get("max_length", 128)),
    )
    direct = _evaluate_rows(
        model,
        tokenizer,
        direct_rows,
        device=device,
        batch_size=int(config["execution"]["batch_size"]),
        max_length=int(config["execution"].get("max_length", 128)),
    )
    _write_ledger(
        ledger_path,
        "CONFIRMATION_FORWARD_COMPLETE",
        split=CONFIRM_SPLIT,
        protected_split=False,
        prompt_variant_id=FROZEN_RENDERER,
    )
    clean_acc = float(clean["full_vocab_accuracy"])
    direct_acc = float(direct["full_vocab_accuracy"])
    passed = clean_acc >= 0.9 and direct_acc >= 0.9
    status = "COMPETENCE_CONFIRMATION_PASSED" if passed else "COMPETENCE_CONFIRMATION_FAILED"
    result = {
        "experiment_id": config["experiment_id"],
        "status": status,
        "evidence_level": "Availability",
        "renderer": FROZEN_RENDERER,
        "model": model_cfg,
        "allowed_model_forward_splits": [CONFIRM_SPLIT],
        "forbidden_model_forward_splits": list(FORBIDDEN_SPLITS),
        "model_forward_splits_executed": [CONFIRM_SPLIT],
        "protected_splits_executed": [],
        "train_executed": False,
        "validation_executed": False,
        "test_executed": False,
        "paraphrase_executed": False,
        "calibration_executed": False,
        "clean": clean,
        "direct_permuted": direct,
        "candidate_only_accuracy_is_diagnostic_only": True,
        "gates": {
            "clean_full_vocab_accuracy_min": 0.9,
            "direct_permuted_full_vocab_accuracy_min": 0.9,
            "clean_passed": clean_acc >= 0.9,
            "direct_passed": direct_acc >= 0.9,
        },
        "episode_meta": {key: value for key, value in meta.items() if key != "episode_identities"},
        "episode_manifest_sha256": _json_sha(meta["episode_identities"]),
        "scientific_boundary": dict(config["scientific_boundary"]),
        "does_not_rescue_v3": True,
        "v3_status_preserved": "INELIGIBLE_TASK_PHASE0",
        "claim_boundary": (
            "fresh competence confirmation of the frozen renderer only; no circuit, CRCT, "
            "workspace, or V3 rescue"
        ),
    }
    _write_ledger(
        ledger_path,
        "COMPETENCE_CONFIRMATION_COMPLETE",
        split=CONFIRM_SPLIT,
        protected_split=False,
        status=status,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ledger = Path(args.ledger)
    try:
        result = run_confirmation(config, ledger_path=ledger)
    except Exception as exc:
        Path(args.output).write_text(
            json.dumps(
                {
                    "status": "INFRASTRUCTURE_FAILURE",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "does_not_rescue_v3": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 2
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if result["status"] == "COMPETENCE_CONFIRMATION_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
