"""Compare the exact EB-JEPA Torch pin with an SM120-compatible runtime."""

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


_OPERATIONS = ("matmul", "conv2d", "gru")


def evaluate_runtime_pair(
    exact: Mapping[str, Any], compatible: Mapping[str, Any]
) -> dict[str, bool]:
    """Return frozen acceptance checks for the two-runtime diagnostic."""

    exact_checks = exact.get("checks", {})
    compatible_checks = compatible.get("checks", {})
    exact_kernel_failures = all(
        not bool(exact_checks.get(name, {}).get("passed", False))
        and "no kernel image is available"
        in str(exact_checks.get(name, {}).get("error", "")).lower()
        for name in _OPERATIONS
    )
    compatible_kernel_success = all(
        bool(compatible_checks.get(name, {}).get("passed", False))
        and bool(compatible_checks.get(name, {}).get("finite", False))
        for name in _OPERATIONS
    )
    return {
        "both_use_python_3_12": str(exact.get("python", "")).startswith("3.12.")
        and str(compatible.get("python", "")).startswith("3.12."),
        "same_sm120_device": exact.get("capability") == [12, 0]
        and compatible.get("capability") == [12, 0]
        and exact.get("device") == compatible.get("device"),
        "exact_is_torch_2_6_cu126": str(exact.get("torch", "")).startswith("2.6.0")
        and str(exact.get("cuda_runtime")) == "12.6",
        "exact_binary_omits_sm120": "sm_120" not in exact.get("arch_list", []),
        "exact_three_kernel_failure": exact_kernel_failures,
        "compatible_is_torch_2_10_cu128": str(compatible.get("torch", "")).startswith(
            "2.10.0"
        )
        and str(compatible.get("cuda_runtime")) == "12.8",
        "compatible_binary_includes_sm120": "sm_120" in compatible.get("arch_list", []),
        "compatible_three_kernel_success": compatible_kernel_success,
    }


def _probe(interpreter: Path, script: Path, label: str) -> tuple[dict[str, Any], float]:
    if not interpreter.is_file():
        raise FileNotFoundError(f"missing isolated interpreter: {interpreter}")
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    result = subprocess.run(
        [str(interpreter), str(script), "--label", label],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    elapsed = time.perf_counter() - started
    payload = json.loads(result.stdout)
    payload["stderr"] = result.stderr.strip()
    return payload, elapsed


def run_eb_jepa_runtime_compatibility(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resources = require_free_disk(resource_profile)
    probe_script = Path(str(config.get("probe_script", "scripts/probe_cuda_runtime.py"))).resolve()
    exact, exact_seconds = _probe(
        Path(str(config["exact_interpreter"])).resolve(), probe_script, "upstream_exact_pin"
    )
    compatible, compatible_seconds = _probe(
        Path(str(config["compatible_interpreter"])).resolve(),
        probe_script,
        "sm120_compatible_deviation",
    )
    passes = evaluate_runtime_pair(exact, compatible)
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-RUNTIME-001")),
        "status": "SMOKE_VALIDATED" if all(passes.values()) else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "passes": passes,
        "all_passed": bool(all(passes.values())),
        "exact_upstream_pin": exact,
        "sm120_compatible_deviation": compatible,
        "probe_runtime_seconds": {
            "exact": exact_seconds,
            "compatible": compatible_seconds,
        },
        "hardware": resources.as_dict(),
        "primary_sources": {
            "pytorch_2_6_release": "https://pytorch.org/blog/pytorch2-6/",
            "pytorch_previous_versions": "https://docs.pytorch.org/get-started/previous-versions/",
            "pytorch_blackwell_tracking": "https://github.com/pytorch/pytorch/issues/145949",
            "pytorch_sm120_maintainer_confirmation": (
                "https://discuss.pytorch.org/t/request-add-cuda-sm-120-blackwell-support-"
                "for-convnextv2-fused-kernels/224904/4"
            ),
        },
        "claim": (
            "On this RTX 5070 Ti (SM120), the upstream Torch 2.6/cu126 pin detects CUDA but cannot "
            "execute matmul, Conv2D, or GRU kernels because its binary omits sm_120. Python "
            "3.12/Torch 2.10/cu128 includes sm_120 and passes the same three finite-kernel checks. "
            "Torch 2.10 is therefore a disclosed hardware-compatibility deviation required for "
            "local GPU execution, not an exact dependency reproduction."
        ),
        "scientific_boundary": (
            "This diagnostic validates runtime feasibility only; it provides no evidence about "
            "JEPA training, planning competence, learned representations, circuits, or workspace."
        ),
    }
    output = Path(
        str(config.get("output_metrics", "artifacts/metrics/eb_jepa_runtime_compatibility.json"))
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
        raise RuntimeError(f"EB-JEPA runtime compatibility diagnostic failed: {passes}")
    return metrics
