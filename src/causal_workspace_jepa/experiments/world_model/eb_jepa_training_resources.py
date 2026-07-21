"""Retained capacity and compiler profile for official EB-JEPA training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk


def evaluate_training_resource_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    revision: str,
    minimum_eager_batch: int,
) -> dict[str, bool]:
    eager = [row for row in rows if row.get("profile", {}).get("mode") == "eager"]
    compile_rows = [row for row in rows if row.get("profile", {}).get("mode") == "compile"]
    minimum_rows = [
        row
        for row in eager
        if int(row.get("profile", {}).get("batch_size", -1)) == minimum_eager_batch
    ]
    return {
        "all_jobs_returned_structured_payloads": len(rows) == len(eager) + len(compile_rows)
        and bool(eager)
        and len(compile_rows) == 1,
        "source_revision_and_cleanliness": all(
            row.get("source", {}).get("revision") == revision
            and bool(row.get("source", {}).get("clean", False))
            for row in rows
        ),
        "sm120_compatible_runtime": all(
            str(row.get("runtime", {}).get("torch", "")).startswith("2.10.0+cu128")
            and row.get("runtime", {}).get("capability") == [12, 0]
            and "sm_120" in row.get("runtime", {}).get("arch_list", [])
            for row in rows
        ),
        "minimum_eager_batch_trains": len(minimum_rows) == 1
        and minimum_rows[0].get("training", {}).get("status") == "SUCCESS"
        and bool(minimum_rows[0].get("training", {}).get("losses_finite", False))
        and bool(minimum_rows[0].get("training", {}).get("gradients_finite", False)),
        "every_success_has_memory_and_update": all(
            row.get("training", {}).get("status") != "SUCCESS"
            or (
                int(row.get("training", {}).get("peak_reserved_bytes", 0)) > 0
                and float(row.get("training", {}).get("parameter_delta", 0.0)) > 0.0
            )
            for row in rows
        ),
    }


def _probe(
    *,
    interpreter: Path,
    script: Path,
    source_root: Path,
    revision: str,
    batch_size: int,
    mode: str,
    steps: int,
    seed: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], str, float]:
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
            "--batch-size",
            str(batch_size),
            "--mode",
            mode,
            "--steps",
            str(steps),
            "--seed",
            str(seed),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=timeout_seconds,
    )
    return json.loads(result.stdout), result.stderr.strip(), time.perf_counter() - started


def run_eb_jepa_training_resources(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resources = require_free_disk(resource_profile)
    interpreter = Path(str(config["interpreter"])).resolve()
    script = Path(str(config["probe_script"])).resolve()
    source_root = Path(str(config["source_root"])).resolve()
    revision = str(config["revision"])
    steps = int(config.get("steps", 2))
    seed = int(config.get("seed", 607))
    timeout_seconds = int(config.get("timeout_seconds", 900))
    jobs = [
        ("eager", int(batch_size)) for batch_size in config.get("eager_batch_sizes", [])
    ] + [("compile", int(config["compile_batch_size"]))]
    rows = []
    stderr_rows = []
    wall_seconds = 0.0
    for mode, batch_size in jobs:
        payload, stderr, elapsed = _probe(
            interpreter=interpreter,
            script=script,
            source_root=source_root,
            revision=revision,
            batch_size=batch_size,
            mode=mode,
            steps=steps,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        rows.append(payload)
        stderr_rows.append({"mode": mode, "batch_size": batch_size, "stderr": stderr})
        wall_seconds += elapsed

    passes = evaluate_training_resource_profile(
        rows,
        revision=revision,
        minimum_eager_batch=int(config["minimum_eager_batch"]),
    )
    successful_eager = [
        row
        for row in rows
        if row.get("profile", {}).get("mode") == "eager"
        and row.get("training", {}).get("status") == "SUCCESS"
    ]
    safe_budget_bytes = int(config["safe_reserved_budget_bytes"])
    safe_eager = [
        row
        for row in successful_eager
        if int(row.get("training", {}).get("peak_reserved_bytes", 0)) <= safe_budget_bytes
    ]
    recommended_batch = max(
        (int(row["profile"]["batch_size"]) for row in safe_eager), default=None
    )
    default_batch = int(config["official_default_batch_size"])
    default_eager = next(
        (
            row
            for row in rows
            if row.get("profile", {}).get("mode") == "eager"
            and int(row.get("profile", {}).get("batch_size", -1)) == default_batch
        ),
        None,
    )
    compile_row = next(row for row in rows if row.get("profile", {}).get("mode") == "compile")
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-TRAIN-RESOURCE-001")),
        "status": "PROFILED" if all(passes.values()) else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "seed": seed,
        "passes": passes,
        "all_passed": bool(all(passes.values())),
        "profiles": rows,
        "probe_stderr": stderr_rows,
        "orchestration_wall_seconds": wall_seconds,
        "summary": {
            "official_default_batch_size": default_batch,
            "official_default_eager_status": (
                default_eager.get("training", {}).get("status") if default_eager else "NOT_RUN"
            ),
            "compile_probe_status": compile_row.get("training", {}).get("status"),
            "compile_error_category": compile_row.get("training", {}).get("error_category"),
            "compile_effective_graph_capture": compile_row.get("compile", {}).get(
                "effective_graph_capture"
            ),
            "compile_unique_graphs": compile_row.get("compile", {}).get("unique_graphs"),
            "safe_reserved_budget_bytes": safe_budget_bytes,
            "recommended_eager_batch_size": recommended_batch,
            "successful_eager_batch_sizes": [
                int(row["profile"]["batch_size"]) for row in successful_eager
            ],
        },
        "hardware": resources.as_dict(),
        "claim": (
            "This is a capacity/compiler diagnostic for the pinned official Two Rooms training "
            "step. It determines an executable local batch and records whether the custom unroll "
            "entrypoint actually produces a torch.compile graph."
        ),
        "scientific_boundary": (
            "Capacity, finite losses, and parameter updates do not establish learned dynamics, "
            "planning competence, a circuit, or a workspace."
        ),
    }
    output = Path(str(config["output_metrics"]))
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
        raise RuntimeError(f"EB-JEPA training resource profile failed: {passes}")
    return metrics
