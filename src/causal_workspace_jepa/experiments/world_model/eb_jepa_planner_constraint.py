"""Confirm the pinned EB-JEPA MPPI action-constraint defect."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk


def evaluate_planner_constraint(payload: Mapping[str, Any]) -> dict[str, bool]:
    source = payload.get("source_contract", {})
    summary = payload.get("summary", {})
    design = payload.get("design", {})
    max_norm = float(design.get("max_norm", 0.0))
    return {
        "frozen_32_seed_design": design.get("num_seeds") == 32
        and design.get("seeds") == list(range(32)),
        "cem_source_contains_norm_enforcement": bool(
            source.get("cem_plan_mentions_max_norms", False)
        ),
        "mppi_source_omits_norm_enforcement": not bool(
            source.get("mppi_plan_mentions_max_norms", True)
        ),
        "environment_passes_action_without_bound_check": not bool(
            source.get("environment_step_checks_action_space", True)
        )
        and bool(source.get("environment_step_passes_action_to_transition", False)),
        "cem_never_violates": int(summary.get("cem_violation_count", -1)) == 0
        and float(summary.get("cem_max_observed_norm", max_norm + 1.0)) <= max_norm + 1e-6,
        "mppi_violation_replication": int(summary.get("mppi_violation_count", 0)) >= 31
        and float(summary.get("mppi_violation_fraction", 0.0)) >= 0.95,
        "mppi_violation_is_material": float(summary.get("mppi_median_max_norm", 0.0))
        >= max_norm + 1.0,
    }


def run_eb_jepa_planner_constraint(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resources = require_free_disk(resource_profile)
    source_root = Path(str(config["source_root"])).resolve()
    interpreter = Path(str(config["interpreter"])).resolve()
    probe_script = Path(str(config["probe_script"])).resolve()
    revision = str(config["revision"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(interpreter),
            str(probe_script),
            "--source-root",
            str(source_root),
            "--revision",
            revision,
            "--num-seeds",
            str(config.get("num_seeds", 32)),
            "--max-norm",
            str(config.get("max_norm", 2.45)),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=180,
    )
    payload = json.loads(result.stdout)
    passes = evaluate_planner_constraint(payload)
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-PLANNER-CONSTRAINT-001")),
        "status": "CONFIRMED_UPSTREAM_DEFECT"
        if all(passes.values())
        else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "post_discovery_confirmation": True,
        "passes": passes,
        "all_passed": bool(all(passes.values())),
        "probe": payload,
        "probe_stderr": result.stderr.strip(),
        "orchestration_wall_seconds": time.perf_counter() - started,
        "hardware": resources.as_dict(),
        "claim": (
            "At pinned revision 966e61e..., official CEM enforces the configured 2.45 action norm "
            "while official MPPI does not. Under an identical deterministic control objective, "
            "the frozen 32-seed confirmation distinguishes the implementations; DotWall.step "
            "does not independently enforce its action-space boundary."
        ),
        "scientific_boundary": (
            "This confirms a planner implementation/configuration defect, not model competence or "
            "a learned JEPA mechanism. Published success must be reproduced with both original and "
            "constraint-corrected MPPI before interpreting action or recurrent circuits."
        ),
    }
    output = Path(
        str(
            config.get(
                "output_metrics", "artifacts/metrics/eb_jepa_planner_constraint.json"
            )
        )
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {Path(config_path).as_posix()}",
        resource_profile=resource_profile,
        seed=None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output.as_posix(), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"EB-JEPA planner-constraint confirmation failed: {passes}")
    return metrics
