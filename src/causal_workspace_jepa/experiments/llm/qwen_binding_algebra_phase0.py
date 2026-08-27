"""Fail-closed Qwen binding-algebra Phase-0 bridge.

Only calibration/train/validation are executable.  Protected test/paraphrase episodes are never
materialized by this module, and there is intentionally no CLI switch that can enable them.

The parent preregistration remains authoritative.  This bridge supplies the previously missing
bounded execution stack for competence/layer-0 replay (B0) and, only if B0 passes, the finite
nonlinearity audit (B1).  It never trains B2/B3/B4 predictors and never opens protected outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ALLOWED_SPLITS = ("calibration", "train", "validation")
FORBIDDEN_SPLITS = ("test", "paraphrase")
TRAJECTORY_LAYERS = (14, 18, 21, 24, 27)
DIRECT_VERIFICATION_LAYER = 21


def _json_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assert_phase0_split_allowed(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise PermissionError(
            f"split {split!r} is protected or outside CRCT-QWEN-BRIDGE-001 Phase-0 scope"
        )


def phase0_scientific_decision(
    *,
    b0_pass: bool,
    derivative_available: bool,
    interaction_power: float | None,
    quadratic_nmse: float | None,
    interaction_min: float = 0.10,
    quadratic_nmse_min: float = 0.10,
) -> str:
    """Pure decision function used by tests and the runtime evaluator."""

    if not b0_pass:
        return "INELIGIBLE_TASK_PHASE0"
    if not derivative_available:
        return "DERIVATIVE_UNAVAILABLE_PHASE0"
    assert interaction_power is not None and quadratic_nmse is not None
    if interaction_power < interaction_min or quadratic_nmse < quadratic_nmse_min:
        return "COMPLETED_NEGATIVE_LOCALLY_DIFFERENTIAL"
    return "PHASE0_B1_ELIGIBLE_FOR_LATER_B2"


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def _assert_committed_and_pushed_protocol(
    expected_base: str, tracked_files: Sequence[str]
) -> dict[str, Any]:
    """Enforce the parent protocol's freeze/commit/push boundary before Qwen execution."""

    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_base, head], cwd=ROOT
    ).returncode == 0
    tracked_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    remote_contains = [
        line.strip()
        for line in _git("branch", "-r", "--contains", head).splitlines()
        if line.strip().startswith("origin/")
    ]
    file_checks: dict[str, Any] = {}
    for relative in tracked_files:
        path = ROOT / relative
        worktree_blob = _git("hash-object", str(path)) if path.exists() else None
        try:
            head_blob = _git("rev-parse", f"HEAD:{relative}")
        except RuntimeError:
            head_blob = None
        file_checks[relative] = {
            "exists": path.exists(),
            "worktree_blob": worktree_blob,
            "head_blob": head_blob,
            "committed_exactly": bool(worktree_blob and worktree_blob == head_blob),
        }
    payload = {
        "head": head,
        "expected_base_ancestor": ancestor,
        "tracked_worktree_clean": not tracked_dirty,
        "origin_remote_refs_containing_head": remote_contains,
        "head_is_pushed_to_origin": bool(remote_contains),
        "tracked_source_checks": file_checks,
        "all_tracked_sources_committed_exactly": all(
            item["committed_exactly"] for item in file_checks.values()
        ),
    }
    if not ancestor:
        raise RuntimeError("frozen base commit is not an ancestor of the protocol commit")
    if tracked_dirty:
        raise RuntimeError("tracked worktree/index must be clean before any Qwen Phase-0 forward")
    if not remote_contains:
        raise RuntimeError(
            "protocol HEAD is not present in an origin/* remote-tracking ref; push first"
        )
    if not payload["all_tracked_sources_committed_exactly"]:
        raise RuntimeError("Phase-0 source/config files are not committed exactly at HEAD")
    return payload


def _load_parent_config(path: Path) -> dict[str, Any]:
    from causal_workspace_jepa.common.config import load_config

    return load_config(path)


def _generate_allowed_protocol(parent: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    """Generate only allowed splits; never call the helper that materializes all five splits."""

    from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
        BindingAlgebraCase,
        decompose_into_transpositions,
        generate_binding_algebra_episodes,
        permutation_changes_slot,
        permutation_class,
        permutations_in_classes,
    )

    pools = parent["token_pools"]
    partition = parent["action_partition"]
    train_classes = tuple(str(value) for value in partition["train_classes"])
    heldout_classes = tuple(str(value) for value in partition["held_out_classes"])
    episodes: list[Any] = []
    cases: list[Any] = []
    split_case_counts: dict[str, int] = defaultdict(int)
    for split in ALLOWED_SPLITS:
        assert_phase0_split_allowed(split)
        split_cfg = parent["splits"][split]
        split_episodes = generate_binding_algebra_episodes(
            split=split,
            keys=pools["keys"][split],
            values=pools["values"][split],
            count=int(split_cfg["count"]),
            seed=int(split_cfg["seed"]),
            template=str(split_cfg.get("template", "primary")),
        )
        episodes.extend(split_episodes)
        classes = train_classes if split in {"calibration", "train"} else heldout_classes
        for episode in split_episodes:
            roster = [
                action
                for action in permutations_in_classes(classes)
                if permutation_changes_slot(action, episode.query_index)
            ]
            for action in roster:
                index = split_case_counts[split]
                split_case_counts[split] += 1
                cases.append(
                    BindingAlgebraCase(
                        case_id=f"{split}-case-{index:05d}",
                        split=split,
                        episode_id=episode.episode_id,
                        query_index=episode.query_index,
                        target_permutation=action,
                        permutation_class=permutation_class(action),
                        generator_rollout=decompose_into_transpositions(action),
                    )
                )
    if any(getattr(item, "split", None) in FORBIDDEN_SPLITS for item in [*episodes, *cases]):
        raise RuntimeError("protected split materialized despite fail-closed generator")
    return episodes, cases


def build_phase0_plan(parent: Mapping[str, Any]) -> dict[str, Any]:
    episodes, cases = _generate_allowed_protocol(parent)
    payload = {
        "schema_version": "qwen_binding_algebra_phase0_plan_v1",
        "allowed_splits": list(ALLOWED_SPLITS),
        "forbidden_splits_not_materialized": list(FORBIDDEN_SPLITS),
        "episodes": [episode.to_dict() for episode in episodes],
        "cases": [case.to_dict() for case in cases],
        "episode_counts": {
            split: sum(episode.split == split for episode in episodes) for split in ALLOWED_SPLITS
        },
        "case_counts": {
            split: sum(case.split == split for case in cases) for split in ALLOWED_SPLITS
        },
    }
    payload["plan_sha256"] = _json_sha(payload)
    return payload


def _model_identity(parent: Mapping[str, Any]) -> dict[str, Any]:
    model = parent.get("model", "Qwen/Qwen3-0.6B")
    if isinstance(model, Mapping):
        name = str(model.get("name") or model.get("model_name") or "Qwen/Qwen3-0.6B")
        revision = str(model.get("revision") or parent.get("revision") or "")
    else:
        name = str(model)
        revision = str(parent.get("revision") or "")
    return {
        "name": name,
        "revision": revision,
        "max_sequence_length": int(parent.get("max_sequence_length", 96)),
    }




def _clean_prompt(episode: Any) -> str:
    from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
        identity_permutation,
    )

    return episode.prompt_after(identity_permutation())

def _single_token_id(tokenizer: Any, value: str) -> int:
    candidates = (" " + value, value)
    for candidate in candidates:
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"registered value is not single-token under pinned tokenizer: {value!r}")


def _candidate_ids(tokenizer: Any, values: Sequence[str], device: Any) -> Any:
    import torch

    return torch.tensor(
        [_single_token_id(tokenizer, str(value)) for value in values], device=device
    )


def _tokenize(
    tokenizer: Any, prompts: Sequence[str], *, device: Any, max_length: int
) -> dict[str, Any]:
    batch = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in batch.items()}


def _last_positions(attention_mask: Any) -> Any:
    return attention_mask.long().sum(dim=1) - 1


def _take_last(hidden: Any, attention_mask: Any) -> Any:
    import torch

    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, _last_positions(attention_mask)]


def _forward_logits(model: Any, batch: Mapping[str, Any]) -> Any:
    return model(**batch, use_cache=False, return_dict=True).logits


def _capture_endpoint(
    model: Any,
    *,
    attention_mask: Any,
    candidate_ids: Any,
    input_ids: Any | None = None,
    inputs_embeds: Any | None = None,
    layers: Sequence[int] = TRAJECTORY_LAYERS,
) -> Any:
    """Capture exact decoder-block outputs plus four role-reindexed candidate logits."""

    import torch

    captures: dict[int, Any] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captures[index] = output[0] if isinstance(output, tuple) else output

        return hook

    decoder_layers = model.model.layers
    for index in layers:
        handles.append(decoder_layers[index].register_forward_hook(make_hook(index)))
    try:
        kwargs = {
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        else:
            kwargs["inputs_embeds"] = inputs_embeds
        output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    missing = [index for index in layers if index not in captures]
    if missing:
        raise RuntimeError(f"decoder hooks failed to capture layers: {missing}")
    parts = [_take_last(captures[index], attention_mask) for index in layers]
    final_logits = _take_last(output.logits, attention_mask)
    role_logits = torch.gather(final_logits, 1, candidate_ids)
    return torch.cat([*parts, role_logits], dim=1)


def _batch_accuracy(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    expected_ids: Sequence[int],
    candidate_rows: Sequence[Sequence[int]],
    *,
    device: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    full_correct = 0
    candidate_correct = 0
    total = 0
    for start in range(0, len(prompts), batch_size):
        stop = min(len(prompts), start + batch_size)
        batch = _tokenize(tokenizer, prompts[start:stop], device=device, max_length=max_length)
        with torch.no_grad():
            logits = _take_last(_forward_logits(model, batch), batch["attention_mask"])
        predicted = logits.argmax(dim=1).detach().cpu().tolist()
        expected = list(expected_ids[start:stop])
        candidates = torch.tensor(candidate_rows[start:stop], device=device, dtype=torch.long)
        candidate_logits = torch.gather(logits, 1, candidates)
        candidate_choice = candidates[
            torch.arange(candidates.shape[0], device=device), candidate_logits.argmax(dim=1)
        ].detach().cpu().tolist()
        full_correct += sum(int(a == b) for a, b in zip(predicted, expected))
        candidate_correct += sum(int(a == b) for a, b in zip(candidate_choice, expected))
        total += stop - start
    return {
        "count": total,
        "full_vocab_accuracy": full_correct / max(total, 1),
        "candidate_only_accuracy_diagnostic": candidate_correct / max(total, 1),
    }


def _token_contract(
    tokenizer: Any, episodes: Sequence[Any], cases: Sequence[Any]
) -> dict[str, Any]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    value_failures: list[str] = []
    values = sorted({value for episode in episodes for value in episode.base_values})
    for value in values:
        try:
            _single_token_id(tokenizer, value)
        except ValueError:
            value_failures.append(value)
    treatment_failures: list[str] = []
    changed_histogram: dict[str, int] = defaultdict(int)
    for case in cases:
        episode = episode_by_id[case.episode_id]
        clean = tokenizer(_clean_prompt(episode), add_special_tokens=True)["input_ids"]
        target = tokenizer(episode.prompt_after(case.target_permutation), add_special_tokens=True)[
            "input_ids"
        ]
        if len(clean) != len(target):
            treatment_failures.append(case.case_id + ":length")
            continue
        changed = sum(int(left != right) for left, right in zip(clean, target))
        support = sum(int(case.target_permutation[index] != index) for index in range(4))
        changed_histogram[str(changed)] += 1
        if changed != support:
            treatment_failures.append(case.case_id + f":changed={changed}:support={support}")
    return {
        "single_token_value_count": len(values),
        "single_token_failures": value_failures,
        "treatment_failures": treatment_failures[:100],
        "treatment_failure_count": len(treatment_failures),
        "changed_position_histogram": dict(changed_histogram),
        "pass": not value_failures and not treatment_failures,
    }


def _head_output_readiness(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[Any],
    *,
    device: Any,
    max_length: int,
) -> dict[str, Any]:
    """Exact o_proj decomposition only; explicitly not post-RoPE QK reconstruction."""

    import torch
    import torch.nn.functional as functional

    calibration = [episode for episode in episodes if episode.split == "calibration"][:8]
    batch = _tokenize(
        tokenizer,
        [_clean_prompt(episode) for episode in calibration],
        device=device,
        max_length=max_length,
    )
    layer = model.model.layers[DIRECT_VERIFICATION_LAYER]
    o_proj = layer.self_attn.o_proj
    captured: dict[str, Any] = {}

    def pre_hook(_module: Any, args: Any) -> None:
        captured["pre"] = args[0]

    def out_hook(_module: Any, _args: Any, output: Any) -> None:
        captured["out"] = output

    h1 = o_proj.register_forward_pre_hook(pre_hook)
    h2 = o_proj.register_forward_hook(out_hook)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False, return_dict=True)
    finally:
        h1.remove()
        h2.remove()
    pre = captured["pre"]
    observed = captured["out"]
    reconstruction = functional.linear(pre, o_proj.weight, o_proj.bias)
    exact_error = float((reconstruction - observed).abs().max().detach().cpu())
    heads = int(model.config.num_attention_heads)
    if pre.shape[-1] % heads != 0:
        raise RuntimeError("pre-o_proj width is not divisible by num_attention_heads")
    head_dim = int(pre.shape[-1] // heads)
    contributions = []
    for head in range(heads):
        start = head * head_dim
        stop = start + head_dim
        contribution = functional.linear(pre[..., start:stop], o_proj.weight[:, start:stop], None)
        contributions.append(contribution)
    summed = torch.stack(contributions, dim=0).sum(dim=0)
    if o_proj.bias is not None:
        summed = summed + o_proj.bias
    decomposition_error = float((summed - observed).abs().max().detach().cpu())
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    return {
        "layer": DIRECT_VERIFICATION_LAYER,
        "o_proj_exact_reconstruction_max_abs": exact_error,
        "sum_head_contributions_max_abs": decomposition_error,
        "num_attention_heads": heads,
        "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", heads)),
        "pre_o_proj_width": int(pre.shape[-1]),
        "head_slice_width": head_dim,
        "q_projection_width": int(q_proj.out_features),
        "k_projection_width": int(k_proj.out_features),
        "exact_qk_post_rope_reconstruction": False,
        "qk_mechanism_claim_allowed": False,
        "note": (
            "This proves only exact linear decomposition of the post-attention pre-o_proj tensor. "
            "RoPE/GQA query-key score reconstruction is deliberately deferred to a separately "
            "tested milestone before any QK-routing claim."
        ),
    }


def _layer0_replay(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[Any],
    cases: Sequence[Any],
    *,
    device: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    selected = [case for case in cases if case.split == "validation"]
    maximum = 0.0
    changed_total = 0
    rows = 0
    for start in range(0, len(selected), batch_size):
        block = selected[start : start + batch_size]
        clean_prompts = [_clean_prompt(episode_by_id[case.episode_id]) for case in block]
        target_prompts = [
            episode_by_id[case.episode_id].prompt_after(case.target_permutation) for case in block
        ]
        clean_batch = _tokenize(tokenizer, clean_prompts, device=device, max_length=max_length)
        target_batch = _tokenize(tokenizer, target_prompts, device=device, max_length=max_length)
        if clean_batch["input_ids"].shape != target_batch["input_ids"].shape:
            raise RuntimeError("clean/target token shapes differ during layer0 replay")
        if not torch.equal(clean_batch["attention_mask"], target_batch["attention_mask"]):
            raise RuntimeError("clean/target attention masks differ during layer0 replay")
        candidates = torch.stack(
            [
                _candidate_ids(tokenizer, episode_by_id[case.episode_id].base_values, device)
                for case in block
            ]
        )
        embed = model.get_input_embeddings()
        with torch.no_grad():
            clean_embed = embed(clean_batch["input_ids"])
            target_embed = embed(target_batch["input_ids"])
            changed = clean_batch["input_ids"] != target_batch["input_ids"]
            patched_embed = torch.where(changed.unsqueeze(-1), target_embed, clean_embed)
            direct = _capture_endpoint(
                model,
                input_ids=target_batch["input_ids"],
                attention_mask=target_batch["attention_mask"],
                candidate_ids=candidates,
            )
            replay = _capture_endpoint(
                model,
                inputs_embeds=patched_embed,
                attention_mask=clean_batch["attention_mask"],
                candidate_ids=candidates,
            )
        maximum = max(maximum, float((direct - replay).abs().max().detach().cpu()))
        changed_total += int(changed.sum().detach().cpu())
        rows += len(block)
    return {
        "validation_case_count": rows,
        "changed_token_count": changed_total,
        "max_abs_endpoint_error": maximum,
    }


def _observed_layer21_patch_replay(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[Any],
    cases: Sequence[Any],
    *,
    device: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Replay the directly observed full layer-21 state before any predicted-state patch.

    This is an execution-integrity control, not a localized circuit claim.  We replace the entire
    layer-21 residual sequence with the directly executed target sequence so downstream layers and
    role logits must replay the direct target up to numerical tolerance.
    """

    import torch

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    selected = [case for case in cases if case.split == "validation"]
    maximum = 0.0
    rows = 0
    target_layer = model.model.layers[DIRECT_VERIFICATION_LAYER]

    def capture_layer21(input_ids: Any, attention_mask: Any) -> Any:
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured["value"] = output[0] if isinstance(output, tuple) else output

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        return captured["value"].detach()

    for start in range(0, len(selected), batch_size):
        block = selected[start : start + batch_size]
        clean_prompts = [_clean_prompt(episode_by_id[case.episode_id]) for case in block]
        target_prompts = [
            episode_by_id[case.episode_id].prompt_after(case.target_permutation) for case in block
        ]
        clean_batch = _tokenize(tokenizer, clean_prompts, device=device, max_length=max_length)
        target_batch = _tokenize(tokenizer, target_prompts, device=device, max_length=max_length)
        if clean_batch["input_ids"].shape != target_batch["input_ids"].shape:
            raise RuntimeError("clean/target token shapes differ in observed layer21 replay")
        if not torch.equal(clean_batch["attention_mask"], target_batch["attention_mask"]):
            raise RuntimeError("clean/target masks differ in observed layer21 replay")
        candidates = torch.stack(
            [
                _candidate_ids(tokenizer, episode_by_id[case.episode_id].base_values, device)
                for case in block
            ]
        )
        target_state = capture_layer21(
            target_batch["input_ids"], target_batch["attention_mask"]
        )
        with torch.no_grad():
            direct = _capture_endpoint(
                model,
                input_ids=target_batch["input_ids"],
                attention_mask=target_batch["attention_mask"],
                candidate_ids=candidates,
                layers=(24, 27),
            )

        def replace_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                return (target_state,) + tuple(output[1:])
            return target_state

        handle = target_layer.register_forward_hook(replace_hook)
        try:
            with torch.no_grad():
                replay = _capture_endpoint(
                    model,
                    input_ids=clean_batch["input_ids"],
                    attention_mask=clean_batch["attention_mask"],
                    candidate_ids=candidates,
                    layers=(24, 27),
                )
        finally:
            handle.remove()
        maximum = max(maximum, float((direct - replay).abs().max().detach().cpu()))
        rows += len(block)
    return {
        "validation_case_count": rows,
        "patch_scope": "entire_observed_blocks.21.resid_post_sequence",
        "max_abs_downstream_endpoint_error": maximum,
        "claim_boundary": "execution-integrity replay only; not mediator localization",
    }


def _b0_metrics(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[Any],
    cases: Sequence[Any],
    *,
    bridge: Mapping[str, Any],
    device: Any,
    max_length: int,
    forward_batch: int,
    replay_batch: int,
) -> dict[str, Any]:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    token_contract = _token_contract(tokenizer, episodes, cases)
    clean_by_split: dict[str, Any] = {}
    direct_by_split: dict[str, Any] = {}
    for split in ALLOWED_SPLITS:
        split_episodes = [episode for episode in episodes if episode.split == split]
        prompts = [_clean_prompt(episode) for episode in split_episodes]
        expected = [
            _single_token_id(tokenizer, episode.base_values[episode.query_index])
            for episode in split_episodes
        ]
        candidates = [
            [_single_token_id(tokenizer, value) for value in episode.base_values]
            for episode in split_episodes
        ]
        clean_by_split[split] = _batch_accuracy(
            model,
            tokenizer,
            prompts,
            expected,
            candidates,
            device=device,
            max_length=max_length,
            batch_size=forward_batch,
        )

        split_cases = [case for case in cases if case.split == split]
        target_prompts = [
            episode_by_id[case.episode_id].prompt_after(case.target_permutation)
            for case in split_cases
        ]
        target_expected = [
            _single_token_id(
                tokenizer,
                episode_by_id[case.episode_id].answer_after(case.target_permutation),
            )
            for case in split_cases
        ]
        target_candidates = [
            [
                _single_token_id(tokenizer, value)
                for value in episode_by_id[case.episode_id].base_values
            ]
            for case in split_cases
        ]
        direct_by_split[split] = _batch_accuracy(
            model,
            tokenizer,
            target_prompts,
            target_expected,
            target_candidates,
            device=device,
            max_length=max_length,
            batch_size=forward_batch,
        )

    replay = _layer0_replay(
        model,
        tokenizer,
        episodes,
        cases,
        device=device,
        max_length=max_length,
        batch_size=replay_batch,
    )
    layer21_replay = _observed_layer21_patch_replay(
        model,
        tokenizer,
        episodes,
        cases,
        device=device,
        max_length=max_length,
        batch_size=replay_batch,
    )
    routing = _head_output_readiness(
        model, tokenizer, episodes, device=device, max_length=max_length
    )
    thresholds = bridge["phase0_thresholds"]
    clean_floor = float(thresholds["clean_accuracy_min"])
    direct_floor = float(thresholds["direct_permuted_accuracy_min"])
    replay_max = float(thresholds["layer0_replay_max_abs"])
    layer21_replay_max = float(thresholds["observed_layer21_patch_replay_max_abs"])
    oproj_max = float(thresholds["o_proj_reconstruction_max_abs"])
    gates = {
        "token_contract": bool(token_contract["pass"]),
        "clean_accuracy_each_allowed_split": all(
            metrics["full_vocab_accuracy"] >= clean_floor for metrics in clean_by_split.values()
        ),
        "direct_permuted_accuracy_each_allowed_split": all(
            metrics["full_vocab_accuracy"] >= direct_floor for metrics in direct_by_split.values()
        ),
        "layer0_replay": replay["max_abs_endpoint_error"] <= replay_max,
        "observed_layer21_patch_replay": (
            layer21_replay["max_abs_downstream_endpoint_error"] <= layer21_replay_max
        ),
        "o_proj_head_decomposition_readiness": (
            routing["o_proj_exact_reconstruction_max_abs"] <= oproj_max
            and routing["sum_head_contributions_max_abs"] <= oproj_max
        ),
    }
    return {
        "token_contract": token_contract,
        "clean_competence": clean_by_split,
        "direct_permuted_competence": direct_by_split,
        "layer0_replay": replay,
        "observed_layer21_patch_replay": layer21_replay,
        "head_output_routing_readiness": routing,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _relative_nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.square(target - prediction, dtype=np.float64).sum())
    denominator = float(np.square(target, dtype=np.float64).sum())
    return numerator / max(denominator, 1e-18)


def _clear_cuda_after_derivative_failure() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _exact_directional_jvp(
    function: Any, alpha0: Any, tangent: Any
) -> tuple[Any, Any, str, str | None]:
    """Exact JVP with an exact reverse-mode fallback; finite differences are forbidden."""

    import torch

    try:
        y0, first = torch.func.jvp(function, (alpha0,), (tangent,))
        return y0, first, "torch.func.jvp", None
    except Exception as first_error:
        _clear_cuda_after_derivative_failure()
        try:
            y0, first = torch.autograd.functional.jvp(
                function,
                alpha0.detach().requires_grad_(True),
                tangent,
                create_graph=False,
                strict=False,
            )
            return (
                y0,
                first,
                "torch.autograd.functional.jvp",
                f"torch.func.jvp failed: {type(first_error).__name__}: {first_error}",
            )
        except Exception as second_error:
            raise RuntimeError(
                "both exact JVP backends failed; finite-difference fallback is forbidden; "
                f"torch.func={type(first_error).__name__}: {first_error}; "
                f"autograd.functional={type(second_error).__name__}: {second_error}"
            ) from second_error


def _exact_directional_jvp_hvp(
    function: Any, alpha0: Any, tangent: Any
) -> tuple[Any, Any, Any, str, str | None]:
    """Exact first/second directional derivatives using two exact autograd routes."""

    import torch

    try:
        y0, first = torch.func.jvp(function, (alpha0,), (tangent,))

        def first_derivative(alpha: Any) -> Any:
            return torch.func.jvp(function, (alpha,), (tangent,))[1]

        _first_at_zero, second = torch.func.jvp(
            first_derivative, (alpha0,), (tangent,)
        )
        return y0, first, second, "torch.func.nested_jvp", None
    except Exception as first_error:
        _clear_cuda_after_derivative_failure()
        try:
            alpha = alpha0.detach().requires_grad_(True)
            y0, first = torch.autograd.functional.jvp(
                function, alpha, tangent, create_graph=True, strict=False
            )

            def first_derivative(alpha_inner: Any) -> Any:
                return torch.autograd.functional.jvp(
                    function,
                    alpha_inner,
                    tangent,
                    create_graph=True,
                    strict=False,
                )[1]

            _first_at_zero, second = torch.autograd.functional.jvp(
                first_derivative,
                alpha,
                tangent,
                create_graph=False,
                strict=False,
            )
            return (
                y0,
                first,
                second,
                "torch.autograd.functional.nested_jvp",
                f"torch.func nested JVP failed: {type(first_error).__name__}: {first_error}",
            )
        except Exception as second_error:
            raise RuntimeError(
                "both exact JVP/HVP backends failed; finite-difference fallback is forbidden; "
                f"torch.func={type(first_error).__name__}: {first_error}; "
                f"autograd.functional={type(second_error).__name__}: {second_error}"
            ) from second_error


def _derivative_batch(
    model: Any,
    tokenizer: Any,
    episodes: Mapping[str, Any],
    cases: Sequence[Any],
    *,
    device: Any,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, str | None]:
    import torch

    clean_prompts = [_clean_prompt(episodes[case.episode_id]) for case in cases]
    target_prompts = [
        episodes[case.episode_id].prompt_after(case.target_permutation) for case in cases
    ]
    clean_batch = _tokenize(tokenizer, clean_prompts, device=device, max_length=max_length)
    target_batch = _tokenize(tokenizer, target_prompts, device=device, max_length=max_length)
    if clean_batch["input_ids"].shape != target_batch["input_ids"].shape:
        raise RuntimeError("clean/target token shapes differ in derivative chord")
    if not torch.equal(clean_batch["attention_mask"], target_batch["attention_mask"]):
        raise RuntimeError("clean/target attention masks differ in derivative chord")
    candidates = torch.stack(
        [_candidate_ids(tokenizer, episodes[case.episode_id].base_values, device) for case in cases]
    )
    embed = model.get_input_embeddings()
    with torch.no_grad():
        clean_embed = embed(clean_batch["input_ids"])
        target_embed = embed(target_batch["input_ids"])
    delta = target_embed - clean_embed
    attention_mask = clean_batch["attention_mask"]

    def function(alpha: Any) -> Any:
        mixed = clean_embed + alpha[:, None, None] * delta
        return _capture_endpoint(
            model,
            inputs_embeds=mixed,
            attention_mask=attention_mask,
            candidate_ids=candidates,
        )

    alpha0 = torch.zeros(len(cases), device=device, dtype=clean_embed.dtype)
    tangent = torch.ones_like(alpha0)
    y0, first, second, derivative_backend, fallback_note = _exact_directional_jvp_hvp(
        function, alpha0, tangent
    )
    with torch.no_grad():
        y1 = function(torch.ones_like(alpha0))
        direct = _capture_endpoint(
            model,
            input_ids=target_batch["input_ids"],
            attention_mask=target_batch["attention_mask"],
            candidate_ids=candidates,
        )
    replay_error = float((y1 - direct).abs().max().detach().cpu())
    effect = y1 - y0
    quadratic = first + 0.5 * second
    return (
        effect.detach().float().cpu().numpy(),
        first.detach().float().cpu().numpy(),
        quadratic.detach().float().cpu().numpy(),
        y1.detach().float().cpu().numpy(),
        replay_error,
        derivative_backend,
        fallback_note,
    )


def _primitive_jvp_effects(
    model: Any,
    tokenizer: Any,
    validation_episodes: Sequence[Any],
    *,
    device: Any,
    max_length: int,
    batch_size: int,
    ledger: Path,
) -> tuple[dict[tuple[str, tuple[int, ...]], np.ndarray], dict[str, int], list[str]]:
    """Exact local primitive JVPs on validation; no finite primitive target is executed."""

    import torch

    from causal_workspace_jepa.experiments.llm.qwen_binding_algebra_protocol import (
        transposition_generators,
    )

    rows = [
        (episode, tuple(action))
        for episode in validation_episodes
        for action in transposition_generators()
    ]
    cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    backend_counts: dict[str, int] = defaultdict(int)
    fallback_notes: list[str] = []
    for start in range(0, len(rows), batch_size):
        block = rows[start : start + batch_size]
        clean_prompts = [_clean_prompt(episode) for episode, _action in block]
        target_prompts = [episode.prompt_after(action) for episode, action in block]
        clean_batch = _tokenize(tokenizer, clean_prompts, device=device, max_length=max_length)
        target_batch = _tokenize(tokenizer, target_prompts, device=device, max_length=max_length)
        if clean_batch["input_ids"].shape != target_batch["input_ids"].shape:
            raise RuntimeError("primitive JVP clean/target token shapes differ")
        if not torch.equal(clean_batch["attention_mask"], target_batch["attention_mask"]):
            raise RuntimeError("primitive JVP clean/target masks differ")
        candidates = torch.stack(
            [_candidate_ids(tokenizer, episode.base_values, device) for episode, _action in block]
        )
        embed = model.get_input_embeddings()
        with torch.no_grad():
            clean_embed = embed(clean_batch["input_ids"])
            target_embed = embed(target_batch["input_ids"])
        delta = target_embed - clean_embed
        attention_mask = clean_batch["attention_mask"]

        def function(alpha: Any) -> Any:
            mixed = clean_embed + alpha[:, None, None] * delta
            return _capture_endpoint(
                model,
                inputs_embeds=mixed,
                attention_mask=attention_mask,
                candidate_ids=candidates,
            )

        alpha0 = torch.zeros(len(block), device=device, dtype=clean_embed.dtype)
        tangent = torch.ones_like(alpha0)
        _y0, first, backend, fallback_note = _exact_directional_jvp(
            function, alpha0, tangent
        )
        backend_counts[backend] += len(block)
        if fallback_note is not None:
            fallback_notes.append(fallback_note)
        values = first.detach().float().cpu().numpy()
        for (episode, action), value in zip(block, values):
            cache[(episode.episode_id, action)] = value
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "stage": "B1_PRIMITIVE_JVP_BATCH",
                        "split": "validation",
                        "episode_action_count": len(block),
                        "finite_primitive_targets_executed": False,
                        "protected_split": False,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return cache, dict(backend_counts), fallback_notes


def _composition_interaction_power(
    model: Any,
    tokenizer: Any,
    validation_episodes: Sequence[Any],
    validation_cases: Sequence[Any],
    case_effects: Mapping[str, np.ndarray],
    *,
    device: Any,
    max_length: int,
    derivative_batch: int,
    ledger: Path,
) -> dict[str, Any]:
    """Residual to summed exact primitive JVPs, respecting train-only finite primitives."""

    primitive_jvp, primitive_backend_counts, primitive_fallback_notes = _primitive_jvp_effects(
        model,
        tokenizer,
        validation_episodes,
        device=device,
        max_length=max_length,
        batch_size=derivative_batch,
        ledger=ledger,
    )
    numerator = 0.0
    denominator = 0.0
    by_class: dict[str, list[float]] = defaultdict(list)
    for case in validation_cases:
        primitive_sum = np.zeros_like(case_effects[case.case_id])
        for action in case.generator_rollout:
            primitive_sum += primitive_jvp[(case.episode_id, tuple(action))]
        composed = case_effects[case.case_id]
        interaction = composed - primitive_sum
        interaction_energy = float(np.square(interaction, dtype=np.float64).sum())
        composed_energy = float(np.square(composed, dtype=np.float64).sum())
        numerator += interaction_energy
        denominator += composed_energy
        by_class[str(case.permutation_class)].append(
            interaction_energy / max(composed_energy, 1e-18)
        )
    return {
        "definition": "finite_composed_effect_minus_sum_exact_local_primitive_jvps",
        "finite_primitive_validation_targets_executed": False,
        "interaction_power_fraction": numerator / max(denominator, 1e-18),
        "case_count": len(validation_cases),
        "primitive_jvp_backend_counts": primitive_backend_counts,
        "primitive_jvp_fallback_notes": primitive_fallback_notes,
        "median_case_relative_interaction_by_class": {
            key: float(np.median(values)) for key, values in by_class.items()
        },
    }


def _b1_metrics(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[Any],
    cases: Sequence[Any],
    *,
    bridge: Mapping[str, Any],
    device: Any,
    max_length: int,
    derivative_batch: int,
    forward_batch: int,
    ledger: Path,
) -> dict[str, Any]:
    validation_cases = [case for case in cases if case.split == "validation"]
    validation_episodes = [episode for episode in episodes if episode.split == "validation"]
    episode_by_id = {episode.episode_id: episode for episode in validation_episodes}
    effects: list[np.ndarray] = []
    firsts: list[np.ndarray] = []
    quadratics: list[np.ndarray] = []
    case_effects: dict[str, np.ndarray] = {}
    replay_max = 0.0
    derivative_backend_counts: dict[str, int] = defaultdict(int)
    derivative_fallback_notes: list[str] = []
    started = time.perf_counter()
    try:
        for start in range(0, len(validation_cases), derivative_batch):
            block = validation_cases[start : start + derivative_batch]
            (
                effect,
                first,
                quadratic,
                _y1,
                replay_error,
                derivative_backend,
                fallback_note,
            ) = _derivative_batch(
                model,
                tokenizer,
                episode_by_id,
                block,
                device=device,
                max_length=max_length,
            )
            effects.append(effect)
            firsts.append(first)
            quadratics.append(quadratic)
            replay_max = max(replay_max, replay_error)
            derivative_backend_counts[derivative_backend] += len(block)
            if fallback_note is not None:
                derivative_fallback_notes.append(fallback_note)
            for case, value in zip(block, effect):
                case_effects[case.case_id] = value
            with ledger.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "stage": "B1_DERIVATIVE_BATCH",
                            "split": "validation",
                            "case_ids": [case.case_id for case in block],
                            "protected_split": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    except Exception as exc:
        return {
            "derivative_available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finite_difference_substitution_used": False,
            "elapsed_seconds": time.perf_counter() - started,
        }

    effect = np.concatenate(effects, axis=0)
    first = np.concatenate(firsts, axis=0)
    quadratic = np.concatenate(quadratics, axis=0)
    hidden_width = int(model.config.hidden_size) * len(TRAJECTORY_LAYERS)
    full_first = _relative_nmse(effect, first)
    full_quadratic = _relative_nmse(effect, quadratic)
    hidden_quadratic = _relative_nmse(effect[:, :hidden_width], quadratic[:, :hidden_width])
    logits_quadratic = _relative_nmse(effect[:, hidden_width:], quadratic[:, hidden_width:])
    interaction = _composition_interaction_power(
        model,
        tokenizer,
        validation_episodes,
        validation_cases,
        case_effects,
        device=device,
        max_length=max_length,
        derivative_batch=derivative_batch,
        ledger=ledger,
    )
    thresholds = bridge["phase0_thresholds"]
    return {
        "derivative_available": True,
        "finite_difference_substitution_used": False,
        "exact_derivative_backend_counts": dict(derivative_backend_counts),
        "exact_derivative_fallback_notes": derivative_fallback_notes,
        "validation_case_count": len(validation_cases),
        "trajectory_layers": list(TRAJECTORY_LAYERS),
        "endpoint_dim": int(effect.shape[1]),
        "first_order_nmse": full_first,
        "quadratic_nmse": full_quadratic,
        "quadratic_hidden_trajectory_nmse": hidden_quadratic,
        "quadratic_role_logit_nmse": logits_quadratic,
        "chord_target_replay_max_abs": replay_max,
        "composition_interaction": interaction,
        "gates": {
            "composition_interaction_power_ge_0_10": (
                interaction["interaction_power_fraction"]
                >= float(thresholds["composition_interaction_power_min"])
            ),
            "quadratic_nmse_ge_0_10": (
                full_quadratic >= float(thresholds["quadratic_nmse_min"])
            ),
            "chord_endpoint_matches_direct_target": (
                replay_max <= float(thresholds["layer0_replay_max_abs"])
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_ledger(
    path: Path,
    stage: str,
    *,
    split: str | None = None,
    count: int | None = None,
    **extra: Any,
) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "split": split,
        "count": count,
        "protected_split": bool(split in FORBIDDEN_SPLITS),
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_phase0(
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
    parent = _load_parent_config(parent_path)
    authorization = bridge["scoped_authorization"]
    if not bool(authorization.get("execution_authorized")):
        raise RuntimeError("bridge Phase-0 execution authorization is not enabled")
    if authorization.get("authorization_scope") != "B0_B1_only_on_calibration_train_validation":
        raise RuntimeError("bridge authorization scope changed")
    if not bool(authorization.get("authorization_is_separate_milestone")):
        raise RuntimeError("bridge authorization must remain a separate milestone")
    if not bool(authorization.get("parent_execution_authorized_flag_remains_unchanged")):
        raise RuntimeError("parent authorization flag may not be rewritten by the bridge")
    if not bool(authorization.get("requires_committed_pushed_bridge_commit")):
        raise RuntimeError("bridge must require a committed and pushed execution milestone")
    if authorization["allowed_splits"] != list(ALLOWED_SPLITS):
        raise RuntimeError("bridge authorization allowed-split roster changed")
    if authorization["forbidden_splits"] != list(FORBIDDEN_SPLITS):
        raise RuntimeError("bridge authorization forbidden-split roster changed")
    if not bool(authorization["phase0_only"]):
        raise RuntimeError("bridge authorization is not Phase-0-only")
    frozen_blob_checks: dict[str, Any] = {}
    for relative, expected_blob in bridge["frozen_parent_git_blobs"].items():
        actual_blob = _git("rev-parse", f"HEAD:{relative}")
        frozen_blob_checks[relative] = {
            "expected_git_blob": expected_blob,
            "actual_git_blob": actual_blob,
            "matches": actual_blob == expected_blob,
        }
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"frozen parent source changed before Phase-0: {relative} "
                f"{actual_blob} != {expected_blob}"
            )

    protocol_guard = _assert_committed_and_pushed_protocol(
        str(bridge["expected_base_commit"]),
        bridge["required_committed_files"],
    )
    plan = build_phase0_plan(parent)
    plan["frozen_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    plan_path = run_dir / "phase0_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    ledger = run_dir / "ACCESS_LEDGER.jsonl"
    _write_ledger(ledger, "PLAN_FROZEN", count=len(plan["cases"]), plan_sha256=plan["plan_sha256"])
    for split in ALLOWED_SPLITS:
        _write_ledger(
            ledger,
            "SPLIT_MATERIALIZED",
            split=split,
            count=int(plan["case_counts"][split]),
        )

    identity = _model_identity(parent)
    status_payload: dict[str, Any] = {
        "schema_version": "qwen_binding_algebra_phase0_bridge_v1",
        "experiment_id": bridge["experiment_id"],
        "parent_experiment_id": bridge["parent_experiment_id"],
        "protocol_guard": {**protocol_guard, "frozen_parent_git_blobs": frozen_blob_checks},
        "phase0_plan_sha256": plan["plan_sha256"],
        "allowed_splits_materialized": list(ALLOWED_SPLITS),
        "allowed_splits_executed": [],
        "model_forward_splits_executed": [],
        "protected_splits_executed": [],
        "model": identity,
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

    _write_ledger(
        ledger,
        "MODEL_LOAD_STARTED",
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
            "reason": f"pinned local model/tokenizer load failed: {type(exc).__name__}: {exc}",
        }
    _write_ledger(
        ledger,
        "MODEL_LOAD_COMPLETE",
        model=identity["name"],
        revision=identity["revision"],
    )

    episodes, cases = _generate_allowed_protocol(parent)
    status_payload["allowed_splits_executed"] = list(ALLOWED_SPLITS)
    status_payload["model_forward_splits_executed"] = list(ALLOWED_SPLITS)
    for split in ALLOWED_SPLITS:
        _write_ledger(
            ledger,
            "B0_MODEL_FORWARD_SCOPE",
            split=split,
            count=sum(case.split == split for case in cases),
        )
    started = time.perf_counter()
    b0 = _b0_metrics(
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
    status_payload["b0"] = b0
    if not b0["pass"]:
        status_payload["status"] = "INELIGIBLE_TASK_PHASE0"
        status_payload["b1"] = {"executed": False, "reason": "B0 competence/replay gate failed"}
    else:
        _write_ledger(ledger, "B0_PASSED_B1_AUTHORIZED", split="validation")
        b1 = _b1_metrics(
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
        status_payload["status"] = phase0_scientific_decision(
            b0_pass=True,
            derivative_available=bool(b1["derivative_available"]),
            interaction_power=(
                b1.get("composition_interaction", {}).get("interaction_power_fraction")
                if b1.get("derivative_available")
                else None
            ),
            quadratic_nmse=(b1.get("quadratic_nmse") if b1.get("derivative_available") else None),
            interaction_min=float(bridge["phase0_thresholds"]["composition_interaction_power_min"]),
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
    status_payload["scientific_boundary"]["test_executed"] = False
    status_payload["scientific_boundary"]["paraphrase_executed"] = False
    status_payload["result_sha256"] = _json_sha(status_payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_ledger(ledger, "PHASE0_COMPLETE", status=status_payload["status"])
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
        payload = run_phase0(
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "output": str(args.output),
                    "protected_splits_executed": payload.get("protected_splits_executed", []),
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "qwen_binding_algebra_phase0_bridge_v1",
            "status": "INFRASTRUCTURE_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "protected_splits_executed": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "INFRASTRUCTURE_FAILURE", "error": str(exc)}, indent=2))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
