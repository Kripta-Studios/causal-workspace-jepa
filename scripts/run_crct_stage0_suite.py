from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "crct_stage0_v1.json"


def _run(
    command: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except FileNotFoundError as exc:
        return {
            "command": list(command),
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": list(command),
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + f"\nTIMEOUT after {timeout}s",
            "elapsed_seconds": time.perf_counter() - started,
        }
    except KeyboardInterrupt:
        return {
            "command": list(command),
            "returncode": 130,
            "stdout": "",
            "stderr": "KeyboardInterrupt while waiting for child process",
            "elapsed_seconds": time.perf_counter() - started,
        }


def _write_command_result(path: Path, result: dict[str, Any]) -> None:
    body = [
        "$ " + " ".join(str(part) for part in result["command"]),
        f"returncode={result['returncode']}",
        f"elapsed_seconds={result['elapsed_seconds']:.6f}",
        "",
        "--- STDOUT ---",
        result["stdout"],
        "",
        "--- STDERR ---",
        result["stderr"],
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8", errors="replace")


def _stage_event(run_dir: Path, stage: str, status: str, **details: Any) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **details,
    }
    with (run_dir / "STAGE_EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"[CRCT] {stage}: {status}", flush=True)


def _safe_env() -> dict[str, str]:
    allowed = (
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "TORCH_LOGS",
        "TORCHDYNAMO_VERBOSE",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {key: os.environ[key] for key in allowed if key in os.environ}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_environment(run_dir: Path) -> dict[str, Any]:
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[str, list[str]]] = [
        ("git_head", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--short", "--branch"]),
        ("git_diff_check", ["git", "diff", "--check"]),
        ("git_diff_cached_check", ["git", "diff", "--cached", "--check"]),
        ("git_diff", ["git", "diff", "--binary"]),
        ("git_diff_cached", ["git", "diff", "--cached", "--binary"]),
        ("git_log", ["git", "log", "-12", "--oneline", "--decorate"]),
        ("git_version", ["git", "--version"]),
        ("python_version", [sys.executable, "--version"]),
        ("pip_list", [sys.executable, "-m", "pip", "list", "--format=json"]),
        ("pip_freeze", [sys.executable, "-m", "pip", "freeze"]),
        ("torch_collect_env", [sys.executable, "-m", "torch.utils.collect_env"]),
        (
            "nvidia_smi_query",
            [
                "nvidia-smi",
                (
                    "--query-gpu=name,uuid,driver_version,pstate,memory.total,memory.free,"
                    "memory.used,temperature.gpu,power.draw,power.limit"
                ),
                "--format=csv,noheader,nounits",
            ],
        ),
        ("nvidia_smi_full", ["nvidia-smi", "-q"]),
        ("nvidia_topology", ["nvidia-smi", "topo", "-m"]),
        ("nvcc_version", ["nvcc", "--version"]),
        (
            "powershell_hardware",
            [
                "pwsh",
                "-NoProfile",
                "-Command",
                (
                    "$ErrorActionPreference='SilentlyContinue'; "
                    "$PSVersionTable | Format-List *; "
                    "Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,"
                    "NumberOfLogicalProcessors,MaxClockSpeed; "
                    "Get-CimInstance Win32_PhysicalMemory | Format-Table Capacity,Speed,"
                    "Manufacturer,PartNumber -AutoSize; "
                    "Get-CimInstance Win32_OperatingSystem | Format-List Caption,Version,"
                    "BuildNumber,TotalVisibleMemorySize,FreePhysicalMemory"
                ),
            ],
        ),
    ]
    command_results: dict[str, Any] = {}
    for name, command in commands:
        result = _run(command, timeout=120)
        command_results[name] = {
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        _write_command_result(diagnostics / f"{name}.txt", result)

    torch_probe_code = r'''
import json, os, platform, sys
payload = {"python": sys.version, "platform": platform.platform(), "pid": os.getpid()}
try:
    import torch
    payload.update({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_config": torch.__config__.show(),
    })
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        payload["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        payload["cuda_mem_get_info"] = {"free": int(free), "total": int(total)}
except Exception as exc:
    payload["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, indent=2, sort_keys=True))
'''
    torch_probe = _run([sys.executable, "-c", torch_probe_code], timeout=120)
    _write_command_result(diagnostics / "torch_probe.txt", torch_probe)
    try:
        torch_payload = json.loads(torch_probe["stdout"]) if torch_probe["returncode"] == 0 else {}
    except json.JSONDecodeError:
        torch_payload = {"parse_error": True}

    disk = shutil.disk_usage(ROOT)
    system = {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "disk_usage_bytes": {
            "total": int(disk.total),
            "used": int(disk.used),
            "free": int(disk.free),
        },
        "python_executable": sys.executable,
        "safe_environment": _safe_env(),
        "commands": command_results,
        "torch": torch_payload,
    }
    (diagnostics / "environment.json").write_text(
        json.dumps(system, indent=2, sort_keys=True), encoding="utf-8"
    )
    return system


def _read_head() -> str:
    result = _run(["git", "rev-parse", "HEAD"])
    if result["returncode"] != 0:
        raise RuntimeError("not inside a readable Git repository")
    return result["stdout"].strip()


def _cuda_free_gib(environment: dict[str, Any]) -> float | None:
    info = environment.get("torch", {}).get("cuda_mem_get_info")
    if not isinstance(info, dict) or "free" not in info:
        return None
    return float(info["free"]) / (1024**3)


def _write_seed_log(path: Path, result: dict[str, Any]) -> None:
    _write_command_result(path, result)


def _aggregate(metrics_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_results.append(payload)
        ranking = payload["rankings"]["residual_causal_fraction"]
        diff = payload["differential_diagnostics"]
        student = payload["residual_student"]
        gauge = payload["gauge_diagnostics"]
        controls = payload["random_matched_controls"]
        rows.append({
            "seed": payload["seed"],
            "status": payload["status"],
            "residual_power_fraction": diff["residual_power_fraction"],
            "first_order_nmse": diff["first_order_nmse"],
            "second_order_nmse": diff["second_order_nmse"],
            "residual_circuit_average_precision": ranking["average_precision"],
            "residual_circuit_precision_at_k": ranking["precision_at_truth_k"],
            "random_control_p95": controls["random_p95"],
            "candidate_residual_score": controls["candidate_topk_total_residual_causal_fraction"],
            "empirical_p_value_plus_one": controls["empirical_p_value_plus_one"],
            "gauge_function_max_abs_error": gauge["function_max_abs_error"],
            "gauge_activation_rank_spearman": gauge["route_activation_rank_spearman"],
            "gauge_causal_rank_spearman": gauge["route_residual_causal_rank_spearman"],
            "student_validation_improvement_fraction": student["validation_improvement_fraction"],
            "student_test_improvement_fraction": student["test_improvement_fraction"],
            "student_test_full_effect_replay_nmse": student["test_full_effect_replay_nmse"],
            "peak_cuda_reserved_bytes": payload["runtime"]["peak_cuda_reserved_bytes"],
            "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
        })

    if not rows:
        aggregate = {"status": "NO_RESULTS", "seed_count": 0, "rows": []}
        (output_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        return aggregate

    rows.sort(key=lambda row: int(row["seed"]))
    raw_results.sort(key=lambda result: int(result["seed"]))
    numeric_keys = [key for key in rows[0] if key not in {"seed", "status"}]
    summary_stats: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        summary_stats[key] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
    aggregate = {
        "schema_version": "crct_stage0_aggregate_v1",
        "status": (
            "SMOKE_VALIDATED"
            if all(row["status"] == "SMOKE_VALIDATED" for row in rows)
            else "NEGATIVE_RESULT"
        ),
        "seed_count": len(rows),
        "seeds": [row["seed"] for row in rows],
        "summary": summary_stats,
        "rows": rows,
        "per_seed_result_sha256": {str(r["seed"]): r["result_sha256"] for r in raw_results},
    }
    aggregate["aggregate_sha256"] = hashlib.sha256(
        json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CRCT Stage-0 diagnostic summary",
        "",
        f"Status: `{aggregate['status']}`",
        f"Seeds: `{', '.join(str(s) for s in aggregate['seeds'])}`",
        "",
        (
            "This is a synthetic falsification/circuit-recovery result only. It is not "
            "evidence for a Qwen circuit, a JEPA workspace, or a protected-model mechanism."
        ),
        "",
        "## Per-seed metrics",
        "",
        (
            "| seed | status | residual power | T1 NMSE | T2 NMSE | residual AP | "
            "random p95 | student test improvement | replay NMSE | gauge causal rho |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['status']} | {row['residual_power_fraction']:.4f} | "
            f"{row['first_order_nmse']:.4f} | {row['second_order_nmse']:.4f} | "
            f"{row['residual_circuit_average_precision']:.4f} | {row['random_control_p95']:.4f} | "
            f"{row['student_test_improvement_fraction']:.4f} | "
            f"{row['student_test_full_effect_replay_nmse']:.4f} | "
            f"{row['gauge_causal_rank_spearman']:.4f} |"
        )
    lines.extend([
        "",
        "## What to inspect next",
        "",
        (
            "1. Confirm every per-seed gate in `metrics/seed_*.json` rather than relying "
            "on aggregate means."
        ),
        (
            "2. Compare `activation_rms` against `residual_causal_fraction`: nuisance should "
            "fool magnitude but not causal residual recovery."
        ),
        (
            "3. Inspect the diagonal-gauge diagnostic: function and residual-causal scores "
            "should remain invariant while coordinate magnitude ranking moves substantially."
        ),
        (
            "4. Check whether the residual student improves held-out replay without changing "
            "the circuit-recovery decision."
        ),
        (
            "5. If any gate fails, preserve the negative result and diagnose it before "
            "changing thresholds."
        ),
        "",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate


def _snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, str]:
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "src" / "causal_workspace_jepa" / "experiments" / "cross_domain" / "crct_stage0.py",
        ROOT / "scripts" / "run_crct_stage0_suite.py",
        ROOT / "scripts" / "run_crct_stage0.ps1",
        ROOT / "tests" / "scientific" / "test_crct_stage0.py",
        ROOT / "docs" / "CRCT_STAGE0_RUNBOOK.md",
        config_path,
    ]
    hashes: dict[str, str] = {}
    for source in sources:
        if not source.exists():
            continue
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path(source.name)
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative.as_posix()] = _sha256_file(source)
    (snapshot_dir / "SOURCE_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return hashes


def _write_manifest(run_dir: Path) -> Path:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            entries.append({
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
    manifest = {
        "schema_version": "crct_stage0_bundle_manifest_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": entries,
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = run_dir / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete CRCT Stage-0 diagnostic suite")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=("smoke", "full", "max"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "reports" / "crct_stage0",
    )
    parser.add_argument("--allow-different-base", action="store_true")
    parser.add_argument("--allow-cpu-full", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--full-test-suite", action="store_true")
    parser.add_argument(
        "--preflight-device",
        default=None,
        help="Device for the smoke preflight. Defaults to config value (auto).",
    )
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=int,
        default=None,
        help="Hard timeout for the smoke preflight subprocess.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"{config['experiment_id']}_{timestamp}"
    logs = run_dir / "logs"
    metrics = run_dir / "metrics"
    logs.mkdir(parents=True, exist_ok=False)
    metrics.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    source_hashes = _snapshot_sources(run_dir, args.config.resolve())
    _stage_event(run_dir, "source_snapshot", "complete", file_count=len(source_hashes))

    _stage_event(run_dir, "environment", "started")
    environment = _collect_environment(run_dir)
    _stage_event(run_dir, "environment", "complete")
    head = _read_head()
    expected = str(config["expected_base_commit"])
    guard = {
        "head": head,
        "expected_base_commit": expected,
        "base_matches": head == expected,
        "allow_different_base": bool(args.allow_different_base),
    }
    (run_dir / "base_guard.json").write_text(json.dumps(guard, indent=2), encoding="utf-8")
    if head != expected and not args.allow_different_base:
        (run_dir / "FAILURE.txt").write_text(
            (
                f"HEAD {head} does not match expected base {expected}. Re-run only after "
                "auditing the diff, or use --allow-different-base explicitly.\n"
            ),
            encoding="utf-8",
        )
        _write_manifest(run_dir)
        print(f"Base-commit guard failed. Diagnostics retained at {run_dir}")
        return 3

    profile = args.profile or str(config["default_profile"])
    device = args.device or str(config["default_device"])
    seeds = args.seeds if args.seeds else [int(v) for v in config["seeds"]]
    if not seeds:
        raise ValueError("at least one seed is required")
    preflight_device = args.preflight_device or str(config.get("preflight_device", "auto"))
    preflight_timeout = (
        args.preflight_timeout_seconds
        if args.preflight_timeout_seconds is not None
        else int(config.get("preflight_timeout_seconds", 300))
    )
    if preflight_timeout <= 0:
        raise ValueError("preflight timeout must be positive")

    torch_env = environment.get("torch", {})
    cuda_available = bool(torch_env.get("cuda_available", False))
    resolved_cuda_run = device.startswith("cuda") or (device == "auto" and cuda_available)
    if profile != "smoke" and not resolved_cuda_run and not args.allow_cpu_full:
        (run_dir / "FAILURE.txt").write_text(
            (
                "Full/max profile requested without a detected CUDA runtime. Use the repo's "
                "SM120-compatible Torch environment; --allow-cpu-full is an explicit override.\n"
            ),
            encoding="utf-8",
        )
        _write_manifest(run_dir)
        print(f"CUDA guard failed. Diagnostics retained at {run_dir}")
        return 4
    free_gib = _cuda_free_gib(environment)
    minimum_free = float(config.get("minimum_cuda_free_gib_for_full", 0.0))
    if profile != "smoke" and free_gib is not None and free_gib < minimum_free:
        (run_dir / "FAILURE.txt").write_text(
            (
                f"CUDA free-memory guard failed: {free_gib:.2f} GiB free < "
                f"{minimum_free:.2f} GiB required.\n"
            ),
            encoding="utf-8",
        )
        _write_manifest(run_dir)
        print(f"CUDA-memory guard failed. Diagnostics retained at {run_dir}")
        return 5

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    test_failure = False
    if not args.skip_tests:
        _stage_event(run_dir, "pytest", "started", full_suite=bool(args.full_test_suite))
        if args.full_test_suite:
            command = [sys.executable, "-m", "pytest", "-q"]
        else:
            command = [sys.executable, "-m", "pytest", "-q", *config["targeted_tests"]]
        test_result = _run(command, env=env)
        _write_command_result(logs / "pytest.txt", test_result)
        test_failure = test_result["returncode"] != 0
        _stage_event(
            run_dir,
            "pytest",
            "failed" if test_failure else "complete",
            returncode=test_result["returncode"],
            elapsed_seconds=test_result["elapsed_seconds"],
        )
    else:
        _stage_event(run_dir, "pytest", "skipped")

    # The targeted tests already exercise the CPU path.  The process-level preflight defaults to
    # `auto` so a CUDA scientific run also validates the actual accelerator/runtime path.  A hard
    # timeout prevents a native backend hang from silently blocking diagnostics for hours.
    preflight_output = metrics / "preflight.json"
    preflight_failure = False
    preflight_returncode: int | None = None
    if args.skip_preflight:
        _stage_event(run_dir, "preflight", "skipped")
    else:
        preflight_command = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "causal_workspace_jepa.experiments.cross_domain.crct_stage0",
            "--profile",
            str(config["preflight_profile"]),
            "--seed",
            str(config["preflight_seed"]),
            "--device",
            preflight_device,
            "--output",
            str(preflight_output),
        ]
        _stage_event(
            run_dir,
            "preflight",
            "started",
            device=preflight_device,
            timeout_seconds=preflight_timeout,
        )
        preflight = _run(preflight_command, env=env, timeout=preflight_timeout)
        _write_seed_log(logs / "preflight.txt", preflight)
        preflight_returncode = int(preflight["returncode"])
        preflight_failure = preflight_returncode != 0
        _stage_event(
            run_dir,
            "preflight",
            "failed" if preflight_failure else "complete",
            returncode=preflight_returncode,
            elapsed_seconds=preflight["elapsed_seconds"],
        )

    seed_failures: list[int] = []
    if preflight_failure:
        _stage_event(
            run_dir,
            "scientific_seeds",
            "blocked_by_preflight_failure",
            preflight_returncode=preflight_returncode,
        )
    for seed in ([] if preflight_failure else seeds):
        output = metrics / f"seed_{seed}.json"
        command = [
            sys.executable,
            "-m",
            "causal_workspace_jepa.experiments.cross_domain.crct_stage0",
            "--profile",
            profile,
            "--seed",
            str(seed),
            "--device",
            device,
            "--output",
            str(output),
        ]
        _stage_event(run_dir, f"seed_{seed}", "started", profile=profile, device=device)
        result = _run(command, env=env)
        _write_seed_log(logs / f"seed_{seed}.txt", result)
        if result["returncode"] != 0:
            seed_failures.append(seed)
        _stage_event(
            run_dir,
            f"seed_{seed}",
            "failed" if result["returncode"] != 0 else "complete",
            returncode=result["returncode"],
            elapsed_seconds=result["elapsed_seconds"],
        )

    aggregate = _aggregate(metrics, run_dir)
    suite_status = {
        "tests_failed": test_failure,
        "preflight_failed": preflight_failure,
        "preflight_device": preflight_device,
        "preflight_timeout_seconds": preflight_timeout,
        "preflight_returncode": preflight_returncode,
        "seed_failures": seed_failures,
        "aggregate_status": aggregate.get("status"),
        "profile": profile,
        "device": device,
        "seeds": seeds,
        "run_dir": str(run_dir),
        "source_hashes": source_hashes,
    }
    suite_status["status"] = (
        "SMOKE_VALIDATED"
        if not test_failure
        and not preflight_failure
        and not seed_failures
        and aggregate.get("status") == "SMOKE_VALIDATED"
        else "NEGATIVE_RESULT_OR_ENGINEERING_FAILURE"
    )
    (run_dir / "SUITE_STATUS.json").write_text(
        json.dumps(suite_status, indent=2, sort_keys=True), encoding="utf-8"
    )
    _stage_event(run_dir, "suite", suite_status["status"])
    _write_manifest(run_dir)
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(json.dumps({**suite_status, "bundle_zip": archive}, indent=2, sort_keys=True))
    return 0 if suite_status["status"] == "SMOKE_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
