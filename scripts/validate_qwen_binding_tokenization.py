#!/usr/bin/env python
"""Validate the frozen Qwen binding treatment without executing the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    assert_disjoint_pools,
    audit_tokenized_treatment,
    generate_binding_episodes,
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
    audits: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for split in ("calibration", "train", "validation", "test", "paraphrase"):
        pool = "test" if split == "paraphrase" else split
        spec = split_config[split]
        episodes = generate_binding_episodes(
            split=split,
            keys=token_pools["keys"][pool],
            values=token_pools["values"][pool],
            count=int(spec["count"]),
            seed=int(spec["seed"]),
            template=str(spec["template"]),
        )
        for episode in episodes:
            audit = audit_tokenized_treatment(
                episode,
                encode_prompt=lambda prompt: tokenizer.encode(
                    prompt, add_special_tokens=True
                ),
                encode_answer=lambda answer: tokenizer.encode(
                    f" {answer}", add_special_tokens=False
                ),
            )
            if len(audit.recipient_ids) > int(config["max_sequence_length"]):
                raise RuntimeError(
                    f"{episode.episode_id} length {len(audit.recipient_ids)} exceeds budget"
                )
            payload = {
                "episode": episode.to_dict(),
                "changed_positions": audit.changed_positions,
                "recipient_answer_id": audit.recipient_answer_id,
                "donor_answer_id": audit.donor_answer_id,
                "sequence_length": len(audit.recipient_ids),
            }
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            digest.update(serialized.encode("utf-8"))
            audits.append(payload)
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
    }
    metrics: dict[str, Any] = {
        "experiment_id": "LLM-QWEN-BINDING-TOKEN-AUDIT-001",
        "parent_experiment_id": str(config["id"]),
        "status": "SMOKE_VALIDATED" if all(gates.values()) else "FAILED",
        "evidence_level": "Availability",
        "model": str(config["model"]),
        "revision": str(config["revision"]),
        "episode_count": len(audits),
        "episode_sha256": digest.hexdigest(),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = validate_tokenization(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
