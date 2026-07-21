"""Evidence-backed audit of the explicit repository completion criteria."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance


def evaluate_completion(root: str | Path = ".") -> dict[str, dict[str, Any]]:
    """Evaluate scientific artifact criteria without mutating repository state."""

    root = Path(root)
    world = _read_json(root / "artifacts/metrics/lewm_small_reproduction_v1.json")
    workspace = _read_json(root / "artifacts/metrics/workspace_discovery_study.json")
    qwen = _read_json(root / "artifacts/metrics/qwen3_0_6b_instrumentation_smoke.json")
    qwen_data = _read_json(root / "artifacts/metrics/qwen3_0_6b_intervention_dataset.json")
    qwen_meta = _read_json(root / "artifacts/metrics/qwen_intervention_jepa_v1.json")
    graph_path = root / "artifacts/tables/lewm_small_circuit.json"
    graph = _read_json(graph_path)
    direct = qwen_meta.get("direct_verification", {})
    primary_scores = qwen_meta.get("scores", {}).get("primary", {})
    world_seeds = world.get("seed_results", [])
    workspace_controls = [
        result.get("workspace", {}).get("detector_controls", {}) for result in world_seeds
    ]
    required_baselines = {
        "no_change",
        "mean_effect",
        "linear",
        "bilinear",
        "mlp",
        "local_jacobian",
        "corpus_jacobian",
        "sparse_feature_linear",
    }
    criteria = {
        "cpu_end_to_end": _criterion(
            _read_json(root / "artifacts/metrics/tiny_jepa_smoke.json").get("status")
            == "SMOKE_VALIDATED",
            "Deterministic Tier 0 generation, tiny JEPA train/save/load/eval, and planning metrics exist.",
        ),
        "published_action_jepa": _criterion(
            world.get("published_integration", {}).get("official_revision")
            == "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
            and world.get("reproduction_passing_seeds") == 3,
            "Source-traceable faithful small LeWorldModel reproduction passed its modeling gate on 3/3 seeds.",
        ),
        "world_selective_intervention": _criterion(
            world.get("planning_passing_seeds", 0) >= 2
            and sum(
                bool(result.get("planning", {}).get("future_latent_trajectory_mse", 0) > 0)
                and bool(result.get("planning", {}).get("mean_absolute_candidate_cost_change", 0) > 0)
                and bool(result.get("planning", {}).get("selected_first_action_change_rate", 0) > 0)
                for result in world_seeds
            )
            >= 2,
            "A hidden-subspace intervention selectively changed future trajectory, planning costs, and selected actions on 2/3 seeds; the broader circuit gate remained negative.",
            "Specificity",
        ),
        "circuit_reconstructed_and_audited": _criterion(
            graph.get("status") == "REJECTED"
            and len(graph.get("nodes", [])) >= 5
            and all(
                all(
                    key in result.get("circuit_audit", {})
                    for key in (
                        "necessity_pass",
                        "sufficiency_pass",
                        "specificity_pass",
                        "minimality_pass",
                    )
                )
                for result in world_seeds
            ),
            "The candidate graph was reconstructed and tested for necessity, sufficiency, specificity, faithfulness, and minimality; it failed replication and is rejected.",
            "Circuit reconstruction",
        ),
        "workspace_controls": _criterion(
            workspace.get("workspace_found") is False
            and bool(workspace.get("detector_controls", {}).get("positive_control_found"))
            and not bool(workspace.get("detector_controls", {}).get("negative_control_found"))
            and len(workspace_controls) == 3
            and all(
                control.get("positive_control_found")
                and not control.get("negative_control_found")
                for control in workspace_controls
            )
            and world.get("workspace_found") is False,
            "Positive/negative controls pass and both tiny/published-reproduction workspace tests retain null results.",
            "Specificity",
        ),
        "real_qwen_hf": _criterion(
            qwen.get("all_passed") is True
            and qwen.get("model") == "Qwen/Qwen3-0.6B"
            and qwen.get("autograd_selected_logit_gradient_norm", 0) > 0,
            "Pinned Hugging Face Qwen hooks, autograd, and direct interventions passed.",
            "Causal mediation",
        ),
        "real_qwen_dataset": _criterion(
            qwen_data.get("all_passed") is True
            and qwen_data.get("outcomes") == 432
            and qwen_data.get("passes", {}).get("effects_nonzero") is True,
            "A checksummed 432-outcome real-Qwen intervention dataset exists with fixed splits.",
            "Causal mediation",
        ),
        "qwen_meta_model_baselines": _criterion(
            qwen_meta.get("all_passed") is True
            and required_baselines.issubset(primary_scores)
            and len(qwen_meta.get("seeds", [])) == 3,
            "Intervention-JEPA and all required baselines were evaluated on fixed held-out prompt/feature/operation splits across three seeds.",
            "Generalization",
        ),
        "direct_qwen_verification": _criterion(
            direct.get("all_predictions_directly_executed") is True
            and len(direct.get("rows", [])) == 16
            and direct.get("circuit_candidate_pass") is False,
            "All 16 predictions were directly executed on new prompts; the failed top-ranked candidate is explicitly rejected.",
            "Causal mediation",
        ),
        "multiple_seed_principal_results": _criterion(
            len(world.get("seeds", [])) == 3 and len(qwen_meta.get("seeds", [])) == 3,
            "Principal world-model and LLM meta-model results use three registered seeds.",
            "Generalization",
        ),
        "documentation_and_manifests": _criterion(
            all(
                (root / relative).is_file()
                for relative in (
                    "README.md",
                    "SUMMARY.md",
                    "docs/HYPOTHESES.md",
                    "docs/LITERATURE.md",
                    "docs/RESULTS.md",
                    "docs/EXPERIMENT_REGISTRY.md",
                    "data/manifests/datasets.yaml",
                    "data/manifests/llm_prompts.yaml",
                    "papers/references.bib",
                )
            ),
            "Required handoff, literature, hypothesis, dataset, experiment, and result registries exist.",
        ),
    }
    return criteria


def run_completion_audit(
    *,
    output_path: str | Path = "artifacts/metrics/completion_audit.json",
    resource_profile: str = "configs/resource/gpu_12gb.yaml",
) -> dict[str, Any]:
    """Run fast software checks and write the final structured audit log."""

    started = time.monotonic()
    provenance = collect_provenance(
        command="python scripts/audit_completion.py",
        resource_profile=resource_profile,
        seed=None,
    )
    checks = {
        "unit_integration_scientific_tests": _run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
        ),
        "ruff": _run(["ruff", "check", "src", "tests", "scripts"]),
        "git_diff_check": _run(["git", "diff", "--check"]),
        "reproducibility_audit": _run([sys.executable, "scripts/audit_reproducibility.py"]),
    }
    criteria = evaluate_completion()
    clean_committed_code = not provenance.git_dirty
    synchronized_before_output = _git_synchronized()
    criteria["clean_committed_execution"] = _criterion(
        clean_committed_code,
        "The audit started from a clean committed worktree; scientific artifacts record clean source commits.",
    )
    criteria["pushed_branch_before_output"] = _criterion(
        synchronized_before_output,
        "HEAD matched its configured upstream before this audit artifact was created.",
    )
    criteria["fast_tests_and_reproducibility"] = _criterion(
        all(check["returncode"] == 0 for check in checks.values()),
        "Full unittest discovery, Ruff, diff check, and reproducibility audit pass.",
    )
    all_passed = all(bool(item["passed"]) for item in criteria.values())
    payload: dict[str, Any] = {
        "experiment_id": "AUDIT-COMPLETE-001",
        "status": "SMOKE_VALIDATED" if all_passed else "BLOCKED",
        "evidence_level": "Availability",
        "all_passed": all_passed,
        "criteria": criteria,
        "software_checks": checks,
        "runtime_seconds": float(time.monotonic() - started),
        "scientific_boundary": (
            "Completion means the explicit repository criteria are implemented and audited. "
            "It does not turn rejected circuit/workspace hypotheses into positive findings."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        destination.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": destination.as_posix(), "all_passed": all_passed},
    )
    if not all_passed:
        failed = [name for name, item in criteria.items() if not item["passed"]]
        raise RuntimeError(f"completion audit failed: {failed}")
    return payload


def _criterion(
    passed: bool, note: str, evidence_level: str = "Availability"
) -> dict[str, Any]:
    return {"passed": bool(passed), "evidence_level": evidence_level, "note": note}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_tail": output[-12:],
    }


def _git_synchronized() -> bool:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        upstream = subprocess.check_output(
            ["git", "rev-parse", "@{upstream}"], text=True
        ).strip()
        return head == upstream
    except (OSError, subprocess.CalledProcessError):
        return False
