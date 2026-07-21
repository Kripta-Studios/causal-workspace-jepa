"""Retained official EB-JEPA Two Rooms integration smoke."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
import tomllib
from typing import Any, Mapping

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk


def _distribution_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


def undeclared_two_rooms_imports(source_root: Path) -> list[str]:
    """Return imports used by Two Rooms but absent from upstream project dependencies."""

    project = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        _distribution_name(str(requirement))
        for requirement in project.get("project", {}).get("dependencies", [])
    }
    required = {"pandas", "pyyaml", "scipy"}
    return sorted(required - declared)


def evaluate_two_rooms_smoke(payload: Mapping[str, Any], revision: str) -> dict[str, bool]:
    runtime = payload.get("runtime", {})
    dataset = payload.get("dataset", {})
    training = payload.get("training", {})
    checkpoint = payload.get("checkpoint", {})
    planner = payload.get("planner", {})
    memory = payload.get("memory", {})
    return {
        "source_revision_matches_pin": payload.get("source", {}).get("revision") == revision,
        "source_checkout_clean": bool(payload.get("source", {}).get("clean", False)),
        "compatible_python_torch_cuda": str(runtime.get("python", "")).startswith("3.12.")
        and str(runtime.get("torch", "")).startswith("2.10.0+cu128")
        and str(runtime.get("cuda_runtime")) == "12.8",
        "sm120_binary_and_device": runtime.get("capability") == [12, 0]
        and "sm_120" in runtime.get("arch_list", []),
        "official_dataset_shapes": dataset.get("batch_shapes")
        == [[2, 2, 9, 65, 65], [2, 2, 9], [2, 2, 9], [2, 1], [2, 1]],
        "training_forward_backward_finite": bool(training.get("loss_finite", False))
        and bool(training.get("gradients_finite", False))
        and float(training.get("parameter_delta", 0.0)) > 0.0,
        "official_prediction_shape": training.get("prediction_shape") == [2, 512, 5, 1, 1],
        "official_parameter_counts": training.get("encoder_parameters") == 1_426_096
        and training.get("predictor_parameters") == 793_600,
        "checkpoint_exact_restore": float(checkpoint.get("max_restore_error", 1.0)) == 0.0
        and int(checkpoint.get("bytes", 0)) > 0,
        "integrated_mppi_plan_finite": planner.get("name") == "MPPIPlanner"
        and planner.get("action_shape") == [1, 2]
        and bool(planner.get("finite", False)),
        "bounded_peak_memory": 0 < int(memory.get("peak_reserved_bytes", 0)) < 12_000_000_000,
    }


def _probe(interpreter: Path, script: Path, source_root: Path, revision: str, seed: int):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    result = subprocess.run(
        [
            str(interpreter),
            str(script),
            "--source-root",
            str(source_root),
            "--revision",
            revision,
            "--seed",
            str(seed),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=180,
    )
    return json.loads(result.stdout), result.stderr.strip(), time.perf_counter() - started


def run_eb_jepa_two_rooms_integration(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resources = require_free_disk(resource_profile)
    source_root = Path(str(config["source_root"])).resolve()
    revision = str(config["revision"])
    seed = int(config.get("seed", 29))
    payload, stderr, wall_seconds = _probe(
        Path(str(config["interpreter"])).resolve(),
        Path(str(config["probe_script"])).resolve(),
        source_root,
        revision,
        seed,
    )
    passes = evaluate_two_rooms_smoke(payload, revision)
    missing_declarations = undeclared_two_rooms_imports(source_root)
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-INTEGRATION-001")),
        "status": "SMOKE_VALIDATED" if all(passes.values()) else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "seed": seed,
        "passes": passes,
        "all_passed": bool(all(passes.values())),
        "probe": payload,
        "probe_stderr": stderr,
        "orchestration_wall_seconds": wall_seconds,
        "undeclared_two_rooms_runtime_imports": missing_declarations,
        "dependency_lock": str(config["dependency_lock"]),
        "hardware": resources.as_dict(),
        "claim": (
            "The pinned official Two Rooms dataset, Impala/GRU JEPA training step, exact "
            "checkpoint restore, and integrated MPPI call execute on the declared SM120-compatible "
            "runtime. Upstream omits pandas, scipy, and PyYAML from the dependency declarations "
            "needed by this path. This is an engineering integration result, not planning competence."
        ),
        "scientific_boundary": (
            "One tiny batch and a random-weight planner do not establish learned dynamics, planning "
            "success, a circuit, or a workspace. Planner constraint semantics are audited separately."
        ),
    }
    output = Path(
        str(config.get("output_metrics", "artifacts/metrics/eb_jepa_two_rooms_integration.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {Path(config_path).as_posix()}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output.as_posix(), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"EB-JEPA Two Rooms integration smoke failed: {passes}")
    return metrics
