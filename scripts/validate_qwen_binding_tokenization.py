#!/usr/bin/env python
"""Validate the frozen Qwen binding treatment without executing the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    assert_disjoint_pools,
    audit_token_pools,
    audit_tokenized_treatment,
    binding_episodes_from_config,
    tokenization_digest,
    tokenized_treatment_payload,
)


def validate_tokenization(config_path: str | Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    config_path = Path(config_path)
    config = load_config(config_path)
    provenance = collect_provenance(
        command=(
            "python scripts/validate_qwen_binding_tokenization.py "
            f"--config {config_path.as_posix()}"
        ),
        resource_profile=str(config["resource_profile"]),
        seed=int(config["seed"]),
    )
    if provenance.git_dirty:
        raise RuntimeError("binding tokenization audit requires a clean committed worktree")
    snapshot = snapshot_download(
        repo_id=str(config["model"]),
        revision=str(config["revision"]),
        local_files_only=bool(config.get("local_files_only", True)),
        token=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        token=False,
    )
    token_pools = config["token_pools"]
    assert_disjoint_pools(token_pools["keys"])
    assert_disjoint_pools(token_pools["values"])
    split_config = config["splits"]
    episodes = binding_episodes_from_config(config)
    audits: list[dict[str, Any]] = []
    for episode in episodes:
        audit = audit_tokenized_treatment(
            episode,
            encode_prompt=lambda prompt: tokenizer.encode(prompt, add_special_tokens=True),
            encode_answer=lambda answer: tokenizer.encode(
                f" {answer}", add_special_tokens=False
            ),
        )
        if len(audit.recipient_ids) > int(config["max_sequence_length"]):
            raise RuntimeError(
                f"{episode.episode_id} length {len(audit.recipient_ids)} exceeds budget"
            )
        audits.append(tokenized_treatment_payload(episode, audit))
    by_split: dict[str, dict[str, Any]] = {}
    for split in split_config:
        rows = [row for row in audits if row["episode"]["split"] == split]
        by_split[split] = {
            "episodes": len(rows),
            "sequence_lengths": sorted({row["sequence_length"] for row in rows}),
            "changed_position_pairs": len(
                {tuple(row["changed_positions"]) for row in rows}
            ),
            "query_position_counts": [
                sum(row["episode"]["query_index"] == index for row in rows)
                for index in range(4)
            ],
        }
    expected = sum(int(spec["count"]) for spec in split_config.values())
    _, pool_items_single_token, pool_ids_disjoint, pool_digest = (
        audit_token_pools(
            token_pools,
            encode_item=lambda item: tokenizer.encode(
                f" {item}", add_special_tokens=False
            ),
        )
    )
    paired_shift_isolated = _paired_shift_isolated(episodes, config)
    resolved_revision = Path(snapshot).name
    gates = {
        "all_expected_episodes_audited": len(audits) == expected,
        "exactly_two_changed_tokens": all(
            len(row["changed_positions"]) == 2 for row in audits
        ),
        "all_sequences_within_budget": all(
            row["sequence_length"] <= int(config["max_sequence_length"])
            for row in audits
        ),
        "query_positions_balanced": all(
            max(summary["query_position_counts"])
            - min(summary["query_position_counts"])
            <= 1
            for summary in by_split.values()
        ),
        "registered_pool_items_single_token": pool_items_single_token,
        "token_ids_disjoint_across_splits_and_roles": pool_ids_disjoint,
        "paired_paraphrase_changes_template_only": paired_shift_isolated,
        "resolved_revision_matches_pin": resolved_revision == str(config["revision"]),
    }
    metrics: dict[str, Any] = {
        "experiment_id": str(
            config.get("tokenization_audit_id", "LLM-QWEN-BINDING-TOKEN-AUDIT-001")
        ),
        "parent_experiment_id": str(config["id"]),
        "status": "SMOKE_VALIDATED" if all(gates.values()) else "FAILED",
        "evidence_level": "Availability",
        "model": str(config["model"]),
        "revision": str(config["revision"]),
        "resolved_revision": resolved_revision,
        "episode_count": len(audits),
        "episode_sha256": tokenization_digest(audits),
        "token_pool_sha256": pool_digest,
        "by_split": by_split,
        "gates": gates,
        "scientific_boundary": (
            "This audit validates deterministic token identity and treatment shape only. "
            "It does not execute Qwen, establish task competence, mediation, a circuit, "
            "an Intervention-JEPA advantage, or a workspace."
        ),
    }
    output = Path(str(config["tokenization_audit_metrics"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output.as_posix(), "gates": gates},
    )
    if not all(gates.values()):
        raise RuntimeError(f"binding tokenization audit failed: {gates}")
    return metrics


def _paired_shift_isolated(
    episodes: list[Any], config: dict[str, Any]
) -> bool:
    paired_specs = {
        split: str(spec["paired_with"])
        for split, spec in config["splits"].items()
        if "paired_with" in spec
    }
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in config["splits"]
    }
    for split, source_split in paired_specs.items():
        shifted = by_split[split]
        source = by_split[source_split]
        if len(shifted) != len(source):
            return False
        for left, right in zip(shifted, source, strict=True):
            if (
                left.keys != right.keys
                or left.recipient_values != right.recipient_values
                or left.donor_values != right.donor_values
                or left.query_index != right.query_index
                or left.swapped_indices != right.swapped_indices
                or left.template == right.template
            ):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = validate_tokenization(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
