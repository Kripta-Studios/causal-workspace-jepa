"""Validate a separately labeled constraint-corrected EB-JEPA MPPI."""

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


def evaluate_mppi_correction(payload: Mapping[str, Any]) -> dict[str, bool]:
    design = payload.get("design", {})
    summary = payload.get("summary", {})
    max_norm = float(design.get("max_norm", 0.0))
    return {
        "frozen_32_seed_design": design.get("num_seeds") == 32
        and design.get("seeds") == list(range(32)),
        "unbounded_actions_match_official": float(
            summary.get("max_unbounded_action_abs_diff", 1.0)
        )
        <= 1e-6,
        "unbounded_losses_match_official": float(
            summary.get("max_unbounded_loss_abs_diff", 1.0)
        )
        <= 1e-6,
        "returned_actions_are_bounded": int(
            summary.get("corrected_returned_violation_count", -1)
        )
        == 0
        and float(summary.get("corrected_max_returned_norm", max_norm + 1.0))
        <= max_norm + 1e-6,
        "cost_inputs_are_bounded": int(
            summary.get("corrected_cost_input_violation_count", -1)
        )
        == 0
        and float(summary.get("corrected_max_cost_input_norm", max_norm + 1.0))
        <= max_norm + 1e-6,
    }


def run_eb_jepa_mppi_correction(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resources = require_free_disk(resource_profile)
    source_root = Path(str(config["source_root"])).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path("src").resolve()), str(source_root)]
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(Path(str(config["interpreter"])).resolve()),
            str(Path(str(config["probe_script"])).resolve()),
            "--source-root",
            str(source_root),
            "--revision",
            str(config["revision"]),
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
    passes = evaluate_mppi_correction(payload)
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-MPPI-CORRECTION-001")),
        "status": "SMOKE_VALIDATED" if all(passes.values()) else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "passes": passes,
        "all_passed": bool(all(passes.values())),
        "probe": payload,
        "probe_stderr": result.stderr.strip(),
        "orchestration_wall_seconds": time.perf_counter() - started,
        "hardware": resources.as_dict(),
        "claim": (
            "A separately named MPPI implementation matches pinned official MPPI when the action "
            "bound is disabled and projects both cost-evaluation candidates and returned actions "
            "when the bound is active. It does not modify the official reproduction baseline."
        ),
        "scientific_boundary": (
            "This validates a planner correction only. It provides no trained planning, model "
            "mechanism, circuit, or workspace evidence."
        ),
    }
    output = Path(
        str(config.get("output_metrics", "artifacts/metrics/eb_jepa_mppi_correction.json"))
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
        raise RuntimeError(f"EB-JEPA MPPI correction failed: {passes}")
    return metrics
