from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
LAST_RUN_DIR: Path | None = None
DEFAULT_CONFIG = ROOT / "configs/experiments/crct_stage0_hard_v2.json"
SOURCE_FILES = (
    "configs/experiments/crct_stage0_hard_v2.json",
    "docs/CRCT_STAGE0_001_RESULT_2026-08-17.md",
    "docs/CRCT_STAGE0_HARD002_PROTOCOL.md",
    "scripts/run_crct_stage0_hard.ps1",
    "scripts/run_crct_stage0_hard_suite.py",
    "src/causal_workspace_jepa/experiments/cross_domain/crct_stage0_hard.py",
    "tests/scientific/test_crct_stage0_hard.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    cwd: Path = ROOT,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            errors="replace",
        )
        return {
            "command": list(command),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except FileNotFoundError as exc:
        return {
            "command": list(command),
            "returncode": 127,
            "stdout": "",
            "stderr": f"FileNotFoundError: {exc}",
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


def _write_command(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "$ " + " ".join(result["command"]),
        f"returncode={result['returncode']}",
        f"elapsed_seconds={result['elapsed_seconds']:.6f}",
        "",
        "--- STDOUT ---",
        result.get("stdout", ""),
        "",
        "--- STDERR ---",
        result.get("stderr", ""),
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8", errors="replace")


def _event(run_dir: Path, stage: str, status: str, **details: Any) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **details,
    }
    with (run_dir / "STAGE_EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"[CRCT-HARD] {stage}: {status}", flush=True)


def _git(*args: str) -> str:
    result = _run(["git", *args])
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"])
    return result["stdout"].strip()


def _snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, str]:
    destination = run_dir / "source_snapshot"
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"required HARD-002 source is missing: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = _sha256(source)
    # The effective config may be custom; preserve it even when not the default path.
    if config_path.resolve() != (ROOT / SOURCE_FILES[0]).resolve():
        target = destination / "effective_config_input.json"
        shutil.copy2(config_path, target)
        hashes["effective_config_input.json"] = _sha256(config_path)
    (destination / "SOURCE_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return hashes


def _guard_current_state(config: dict[str, Any], *, allow_different_base: bool) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    expected = str(config["expected_base_commit"])
    hash_checks: dict[str, Any] = {}
    for relative, expected_hash in config["required_crct001_source_hashes"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.exists() else None
        hash_checks[relative] = {
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "matches": actual == expected_hash,
        }
    guard = {
        "head": head,
        "expected_base_commit": expected,
        "head_matches": head == expected,
        "allow_different_base": allow_different_base,
        "crct001_source_hash_checks": hash_checks,
        "all_required_crct001_sources_match": all(item["matches"] for item in hash_checks.values()),
    }
    if head != expected and not allow_different_base:
        raise RuntimeError(f"HEAD {head} != frozen base {expected}; use --allow-different-base only deliberately")
    if not guard["all_required_crct001_sources_match"]:
        failures = [name for name, item in hash_checks.items() if not item["matches"]]
        raise RuntimeError(
            "CRCT-STAGE0-001/hotfix source state differs from the analyzed successful run: "
            + ", ".join(failures)
        )
    return guard


def _collect_environment(run_dir: Path) -> dict[str, Any]:
    diagnostic = run_dir / "diagnostics"
    diagnostic.mkdir(parents=True, exist_ok=True)
    commands: dict[str, list[str]] = {
        "git_status": ["git", "status", "--short"],
        "git_diff": ["git", "diff"],
        "git_diff_cached": ["git", "diff", "--cached"],
        "git_diff_check": ["git", "diff", "--check"],
        "git_diff_cached_check": ["git", "diff", "--cached", "--check"],
        "git_log": ["git", "log", "-20", "--oneline", "--decorate"],
        "git_version": ["git", "--version"],
        "python_version": [sys.executable, "--version"],
        "pip_list": [sys.executable, "-m", "pip", "list"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "torch_collect_env": [sys.executable, "-m", "torch.utils.collect_env"],
        "nvidia_smi_full": ["nvidia-smi"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_topology": ["nvidia-smi", "topo", "-m"],
        "nvcc_version": ["nvcc", "--version"],
    }
    summary: dict[str, Any] = {}
    for name, command in commands.items():
        result = _run(command, timeout=120)
        _write_command(diagnostic / f"{name}.txt", result)
        summary[name] = {
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
        }

    probe = r'''
import json, os, platform, sys
import torch
payload = {
  "python": sys.version,
  "platform": platform.platform(),
  "torch": torch.__version__,
  "cuda_available": torch.cuda.is_available(),
  "cuda_runtime": torch.version.cuda,
  "cpu_count": os.cpu_count(),
}
if torch.cuda.is_available():
  free, total = torch.cuda.mem_get_info()
  payload.update({
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_free_bytes": int(free),
    "cuda_total_bytes": int(total),
  })
print(json.dumps(payload, indent=2, sort_keys=True))
'''
    result = _run([sys.executable, "-c", probe], timeout=120)
    _write_command(diagnostic / "torch_probe.txt", result)
    torch_payload: dict[str, Any] = {}
    if result["returncode"] == 0:
        try:
            torch_payload = json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass

    if os.name == "nt":
        ps = shutil.which("pwsh") or shutil.which("powershell")
        if ps:
            command = [
                ps,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor,Win32_PhysicalMemory,Win32_VideoController | Format-List *",
            ]
            result = _run(command, timeout=120)
            _write_command(diagnostic / "powershell_hardware.txt", result)

    environment = {"commands": summary, "torch": torch_payload}
    (diagnostic / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    return environment


def _latest_crct001_zip() -> dict[str, Any] | None:
    root = ROOT / "artifacts/reports/crct_stage0"
    if not root.exists():
        return None
    paths = sorted(root.glob("CRCT-STAGE0-001_*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return None
    path = paths[0]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(metrics_dir: Path, seeds: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    for seed in seeds:
        path = metrics_dir / f"seed_{seed}.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        statuses.append(str(payload["status"]))
        iid = payload["iid_confirmation"]
        ood = payload["ood_confirmation"]
        ranking = payload["ranking_diagnostics"]
        diff = payload["differential_diagnostics"]
        gauge = payload["gauge_diagnostics"]
        students = payload["students"]
        screen = payload["screen_flag_fix"]
        rows.append(
            {
                "seed": seed,
                "status": payload["status"],
                "result_sha256": payload["result_sha256"],
                "residual_power_fraction": diff["iid_residual_power_fraction"],
                "t1_nmse": diff["iid_first_order_nmse"],
                "t2_nmse": diff["iid_second_order_nmse"],
                "iid_recovery": iid["circuit_recovery_fraction"],
                "ood_recovery": ood["circuit_recovery_fraction"],
                "node_precision": ranking["selected_node_precision"],
                "node_recall": ranking["selected_node_recall"],
                "edge_precision": ranking["selected_edge_precision"],
                "edge_recall": ranking["selected_edge_recall"],
                "matched_control_p": iid["matched_control_empirical_p_plus_one"],
                "selected_minus_control_p95": iid["selected_minus_control_p95"],
                "decoy_rejection": iid["decoy_rejection_fraction"],
                "gauge_activation_rho": gauge["activation_rank_spearman"],
                "gauge_causal_rho": gauge["causal_rank_spearman"],
                "screen_flag_fix_ap": screen["screen_flag_fix_ap"],
                "exact_finite_ap": screen["exact_finite_ap"],
                "residual_exact_ap": screen["residual_exact_ap"],
                "residual_student_iid_nmse": students["residual_student"]["iid_test"]["nmse"],
                "residual_student_ood_nmse": students["residual_student"]["ood_test"]["nmse"],
                "direct_student_iid_nmse": students["direct_delta_student"]["iid_test"]["nmse"],
                "direct_student_ood_nmse": students["direct_delta_student"]["ood_test"]["nmse"],
                "selected_count": len(payload["frozen_discovery_plan"]["selected"]),
                "plan_sha256": payload["frozen_discovery_plan"]["sha256"],
            }
        )
    if not rows:
        status = "INFRASTRUCTURE_FAILURE"
    elif all(value == "HARD_VALIDATED" for value in statuses) and len(rows) == len(seeds):
        status = "HARD_VALIDATED"
    elif all(value in {"HARD_VALIDATED", "NEGATIVE_RESULT"} for value in statuses) and len(rows) == len(seeds):
        status = "NEGATIVE_RESULT"
    else:
        status = "INFRASTRUCTURE_FAILURE"
    payload = {
        "schema_version": "crct_stage0_hard_aggregate_v2",
        "status": status,
        "seed_count": len(rows),
        "expected_seed_count": len(seeds),
        "seeds": list(seeds),
        "rows": rows,
    }
    payload["aggregate_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(run_dir: Path, aggregate: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# CRCT Stage-0 HARD-002 diagnostic summary",
        "",
        f"Status: `{aggregate['status']}`",
        f"Primary seeds: `{', '.join(map(str, aggregate['seeds']))}`",
        "",
        "This is a planted synthetic falsification benchmark. A positive result validates only the",
        "frozen discovery/confirmation machinery; a negative result is retained without threshold retuning.",
        "Neither outcome is a Qwen/JEPA circuit or workspace result.",
        "",
        "## Per-seed confirmation",
        "",
        "| seed | status | residual power | IID recovery | OOD recovery | node P/R | edge P/R | matched p | margin over p95 | gauge causal rho | residual/direct OOD NMSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["rows"]:
        lines.append(
            "| {seed} | {status} | {residual_power_fraction:.4f} | {iid_recovery:.4f} | "
            "{ood_recovery:.4f} | {node_precision:.3f}/{node_recall:.3f} | "
            "{edge_precision:.3f}/{edge_recall:.3f} | {matched_control_p:.4f} | "
            "{selected_minus_control_p95:.4f} | {gauge_causal_rho:.4f} | "
            "{residual_student_ood_nmse:.4f}/{direct_student_ood_nmse:.4f} |".format(**row)
        )
    lines += [
        "",
        "## Frozen rules",
        "",
        "- Discovery uses validation-only signed residual reconstruction.",
        "- Selection and matched controls are hashed before IID/OOD confirmation is generated.",
        "- Matched controls are evaluated on confirmation without re-selection.",
        "- HVP/Screen-Flag-Fix and residual-vs-direct students are diagnostics, not mandatory wins.",
        "- Exact finite patching/reconstruction remains the confirmation standard.",
        "",
        "## Frozen gates",
        "",
    ]
    for key, value in config["frozen_gates"].items():
        lines.append(f"- `{key}` = `{value}`")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_registry_candidate(run_dir: Path, aggregate: dict[str, Any], previous: dict[str, Any] | None) -> None:
    lines = [
        "# Candidate registry update — do not copy blindly before review",
        "",
        "## CRCT-STAGE0-001 historical bounded result",
        "",
        "Three full-profile seeds 7/13/23 passed the original synthetic positive-control gates.",
        "Mean residual power was 0.527858; residual-causal AP and precision@k were 1.0 in 3/3 seeds;",
        "gauge-causal Spearman was 1.0 in 3/3 seeds. The original matched-random p-value is not",
        "accepted as confirmatory specificity because its candidate top-k was selected using the same",
        "score later compared against random sets. Preserve CRCT-001 as method validation only.",
        "",
    ]
    if previous:
        lines += [
            f"Local CRCT-001 bundle reference: `{previous['path']}`",
            f"Bundle SHA-256: `{previous['sha256']}`",
            "",
        ]
    lines += [
        "## CRCT-STAGE0-HARD-002",
        "",
        f"Aggregate disposition: `{aggregate['status']}`.",
        "",
        "HARD-002 separates validation discovery from generated-after-freeze confirmation; introduces",
        "active state decoys, cancelling action-sensitive decoys, redundant/cancelling true paths,",
        "QK-like bilinear routing edges, IID/OOD confirmation, node/edge recovery, and frozen matched",
        "controls. HVP Screen-Flag-Fix and equal-capacity residual/direct students are non-gating.",
        "",
        "Per-seed result hashes:",
    ]
    for row in aggregate["rows"]:
        lines.append(f"- seed {row['seed']}: `{row['status']}` — `{row['result_sha256']}`")
    lines += [
        "",
        "After independent review, integrate this disposition into SUMMARY.md, docs/RESULTS.md,",
        "docs/EXPERIMENT_REGISTRY.md, docs/APPROACH_REGISTRY.md, docs/ROADMAP.md, docs/TODO.md,",
        "docs/RESEARCH_GAPS.md, docs/LITERATURE.md, and the working paper without upgrading the",
        "evidence level beyond synthetic method validation.",
    ]
    (run_dir / "registry_candidate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(run_dir: Path) -> None:
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            relative = path.relative_to(run_dir).as_posix()
            entries[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    payload = {
        "schema_version": "crct_stage0_hard_manifest_v2",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(entries),
        "files": entries,
    }
    (run_dir / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=["smoke", "full"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--preflight-device", default=None)
    parser.add_argument("--preflight-timeout-seconds", type=int, default=None)
    parser.add_argument("--seed-timeout-seconds", type=int, default=None)
    parser.add_argument("--allow-different-base", action="store_true")
    parser.add_argument("--allow-cpu-full", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--full-test-suite", action="store_true")
    return parser.parse_args()


def main() -> int:
    global LAST_RUN_DIR
    args = _parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile = args.profile or str(config["default_profile"])
    device = args.device or str(config["default_device"])
    seeds = args.seeds if args.seeds else [int(v) for v in config["primary_seeds"]]
    preflight_device = args.preflight_device or str(config["preflight_device"])
    preflight_timeout = args.preflight_timeout_seconds or int(config["preflight_timeout_seconds"])
    seed_timeout = args.seed_timeout_seconds or int(config["seed_timeout_seconds"])

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "artifacts/reports/crct_stage0_hard" / f"CRCT-STAGE0-HARD-002_{stamp}"
    LAST_RUN_DIR = run_dir
    logs = run_dir / "logs"
    metrics = run_dir / "metrics"
    logs.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)

    effective = {
        **config,
        "effective_overrides": {
            "profile": profile,
            "device": device,
            "seeds": seeds,
            "preflight_device": preflight_device,
            "preflight_timeout_seconds": preflight_timeout,
            "seed_timeout_seconds": seed_timeout,
            "full_test_suite": bool(args.full_test_suite),
        },
    }
    (run_dir / "config.source.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "config.effective.json").write_text(
        json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8"
    )

    source_hashes = _snapshot_sources(run_dir, args.config.resolve())
    _event(run_dir, "source_snapshot", "complete", file_count=len(source_hashes))

    guard = _guard_current_state(config, allow_different_base=args.allow_different_base)
    (run_dir / "base_guard.json").write_text(
        json.dumps(guard, indent=2, sort_keys=True), encoding="utf-8"
    )
    _event(run_dir, "base_guard", "complete")

    _event(run_dir, "environment", "started")
    environment = _collect_environment(run_dir)
    _event(run_dir, "environment", "complete")

    torch_env = environment.get("torch", {})
    cuda_available = bool(torch_env.get("cuda_available", False))
    if profile == "full" and not cuda_available and not args.allow_cpu_full:
        _event(run_dir, "resource_guard", "failed", reason="cuda_not_available")
        raise RuntimeError("full HARD-002 requires CUDA unless --allow-cpu-full is explicit")
    if profile == "full" and cuda_available:
        free = int(torch_env.get("cuda_free_bytes", 0))
        required = int(float(config["minimum_cuda_free_gib_for_full"]) * (1024**3))
        if free < required:
            _event(run_dir, "resource_guard", "failed", cuda_free_bytes=free, required_bytes=required)
            raise RuntimeError("insufficient free CUDA memory for frozen full-profile launch")
    _event(run_dir, "resource_guard", "complete")

    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    tests_failed = False
    if not args.skip_tests:
        _event(run_dir, "pytest", "started", full_suite=bool(args.full_test_suite))
        if args.full_test_suite:
            command = [sys.executable, "-m", "pytest", "-q"]
        else:
            command = [sys.executable, "-m", "pytest", "-q", *config["targeted_tests"]]
        test_result = _run(command, env=env, timeout=900)
        _write_command(logs / "pytest.txt", test_result)
        tests_failed = test_result["returncode"] != 0
        _event(
            run_dir,
            "pytest",
            "failed" if tests_failed else "complete",
            returncode=test_result["returncode"],
            elapsed_seconds=test_result["elapsed_seconds"],
        )
    else:
        _event(run_dir, "pytest", "skipped")

    preflight_output = metrics / "preflight.json"
    preflight_command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "causal_workspace_jepa.experiments.cross_domain.crct_stage0_hard",
        "--profile",
        str(config["preflight_profile"]),
        "--seed",
        str(config["preflight_seed"]),
        "--device",
        preflight_device,
        "--preflight-only",
        "--output",
        str(preflight_output),
    ]
    _event(run_dir, "preflight", "started", device=preflight_device, timeout_seconds=preflight_timeout)
    preflight = _run(preflight_command, env=env, timeout=preflight_timeout)
    _write_command(logs / "preflight.txt", preflight)
    preflight_failed = preflight["returncode"] != 0 or not preflight_output.exists()
    _event(
        run_dir,
        "preflight",
        "failed" if preflight_failed else "complete",
        returncode=preflight["returncode"],
        elapsed_seconds=preflight["elapsed_seconds"],
    )

    infrastructure_failures: list[int] = []
    scientific_negatives: list[int] = []
    if not tests_failed and not preflight_failed:
        for seed in seeds:
            output = metrics / f"seed_{seed}.json"
            command = [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                "causal_workspace_jepa.experiments.cross_domain.crct_stage0_hard",
                "--profile",
                profile,
                "--seed",
                str(seed),
                "--device",
                device,
                "--output",
                str(output),
            ]
            _event(run_dir, f"seed_{seed}", "started", profile=profile, device=device, timeout_seconds=seed_timeout)
            result = _run(command, env=env, timeout=seed_timeout)
            _write_command(logs / f"seed_{seed}.txt", result)
            payload: dict[str, Any] | None = None
            if output.exists():
                try:
                    payload = _load_json(output)
                except Exception:
                    payload = None
            # Return code 2 is the module's deliberate scientific-negative disposition.
            infrastructure_failure = result["returncode"] not in {0, 2} or payload is None
            if infrastructure_failure:
                infrastructure_failures.append(seed)
                event_status = "infrastructure_failure"
            elif payload["status"] == "NEGATIVE_RESULT":
                scientific_negatives.append(seed)
                event_status = "negative_result"
            else:
                event_status = "complete"
            _event(
                run_dir,
                f"seed_{seed}",
                event_status,
                returncode=result["returncode"],
                scientific_status=None if payload is None else payload.get("status"),
                elapsed_seconds=result["elapsed_seconds"],
            )
    else:
        _event(
            run_dir,
            "primary_seeds",
            "blocked",
            tests_failed=tests_failed,
            preflight_failed=preflight_failed,
        )

    aggregate = _aggregate(metrics, seeds)
    if tests_failed or preflight_failed or infrastructure_failures:
        aggregate["status"] = "INFRASTRUCTURE_FAILURE"
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(run_dir / "metrics.csv", aggregate["rows"])
    previous = _latest_crct001_zip()
    (run_dir / "prior_crct001_bundle.json").write_text(
        json.dumps(previous, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary(run_dir, aggregate, config)
    _write_registry_candidate(run_dir, aggregate, previous)

    suite = {
        "experiment_id": config["experiment_id"],
        "status": aggregate["status"],
        "profile": profile,
        "device": device,
        "seeds": seeds,
        "tests_failed": tests_failed,
        "preflight_failed": preflight_failed,
        "infrastructure_failures": infrastructure_failures,
        "scientific_negative_seeds": scientific_negatives,
        "source_hashes": source_hashes,
        "run_dir": str(run_dir),
    }
    (run_dir / "SUITE_STATUS.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True), encoding="utf-8"
    )
    _event(run_dir, "suite", aggregate["status"])
    _write_manifest(run_dir)
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    suite["bundle_zip"] = archive
    print(json.dumps(suite, indent=2, sort_keys=True))

    if aggregate["status"] == "INFRASTRUCTURE_FAILURE":
        return 4
    # A scientific negative is a successful execution and should not look like infrastructure fail.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        run_dir = LAST_RUN_DIR
        if run_dir is not None:
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "FAILURE.txt").write_text(
                    "".join(traceback.format_exception(exc)), encoding="utf-8", errors="replace"
                )
                _event(run_dir, "suite", "INFRASTRUCTURE_FAILURE", exception=type(exc).__name__)
                _write_manifest(run_dir)
                archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
                print(json.dumps({
                    "status": "INFRASTRUCTURE_FAILURE",
                    "run_dir": str(run_dir),
                    "bundle_zip": archive,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }, indent=2, sort_keys=True))
            except Exception:
                pass
        raise
