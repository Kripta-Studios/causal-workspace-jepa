"""Retain the EB-JEPA MPPI proposal-scale configuration diagnostic."""

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


def evaluate_planner_config(payload: Mapping[str, Any]) -> dict[str, bool]:
    yaml_config = payload.get("yaml", {})
    signatures = payload.get("signatures", {})
    instances = payload.get("instances", {})
    interpretation = payload.get("interpretation", {})
    return {
        "mppi_yaml_uses_var_scale_only": yaml_config.get("mppi_var_scale") == 1.5
        and not bool(yaml_config.get("mppi_has_max_std", True)),
        "mppi_signature_uses_max_std_and_kwargs": "max_std"
        in signatures.get("mppi_parameters", [])
        and "var_scale" not in signatures.get("mppi_parameters", [])
        and bool(signatures.get("mppi_accepts_kwargs", False)),
        "official_mppi_uses_default_two": instances.get("official_mppi_max_std") == 2.0,
        "official_cem_consumes_one_point_five": instances.get("official_cem_var_scale") == 1.5,
        "correct_keyword_recovers_one_point_five": instances.get(
            "keyword_corrected_mppi_max_std"
        )
        == 1.5,
        "runtime_confirms_silent_mismatch": bool(
            interpretation.get("var_scale_is_silently_ignored_by_mppi", False)
        )
        and bool(interpretation.get("actual_proposal_scales_differ", False)),
    }


def run_eb_jepa_planner_config(config_path: str | Path) -> dict[str, Any]:
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
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=180,
    )
    payload = json.loads(result.stdout)
    passes = evaluate_planner_config(payload)
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-PLANNER-CONFIG-001")),
        "status": "CONFIRMED_UPSTREAM_CONFIG_MISMATCH"
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
            "The pinned MPPI YAML supplies var_scale=1.5, but MPPIPlanner expects max_std and "
            "silently uses its 2.0 default through **kwargs. CEM consumes var_scale=1.5."
        ),
        "scientific_boundary": (
            "This is a planner configuration-contract result, not learned-model competence or "
            "mechanistic evidence. Reproduction arms must distinguish as-executed 2.0 from an "
            "intention-corrected 1.5 proposal scale."
        ),
    }
    output = Path(str(config["output_metrics"]))
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
        raise RuntimeError(f"EB-JEPA planner-config confirmation failed: {passes}")
    return metrics
