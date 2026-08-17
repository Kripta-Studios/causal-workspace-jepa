from __future__ import annotations

import argparse
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
DEFAULT_CONFIG = ROOT / "configs/experiments/crct_qwen_bridge_v1.json"
REPORT_ROOT = ROOT / "artifacts/reports/crct_qwen_bridge"
SOURCE_FILES = (
    "configs/experiments/crct_qwen_bridge_v1.json",
    "configs/experiments/qwen_binding_algebra_v2.yaml",
    "configs/experiments/qwen_binding_algebra_cr_v1.yaml",
    "docs/CRCT_QWEN_BRIDGE_V1_PROTOCOL.md",
    "scripts/run_crct_qwen_bridge.ps1",
    "scripts/run_crct_qwen_bridge_suite.py",
    "src/causal_workspace_jepa/interpretability/circuit_ontology_v3.py",
    "src/causal_workspace_jepa/experiments/cross_domain/circuit_ontology_v3_audit.py",
    "src/causal_workspace_jepa/experiments/llm/qwen_capital_crct_dev.py",
    "src/causal_workspace_jepa/experiments/llm/qwen_substrate_readiness.py",
    "src/causal_workspace_jepa/experiments/llm/qwen_binding_algebra_phase0.py",
    "src/causal_workspace_jepa/experiments/llm/qwen_binding_algebra_protocol.py",
    "tests/unit/test_circuit_ontology_v3.py",
    "tests/unit/test_qwen_capital_crct_dev.py",
    "tests/unit/test_qwen_substrate_readiness.py",
    "tests/unit/test_qwen_bridge_phase0_guard.py",
)
OPTIONAL_PROVENANCE_FILES = (
    "artifacts/metrics/qwen_capital_patch_dataset_v1.json",
    "data/manifests/qwen_capital_patches_v1.json",
    "docs/CRCT_STAGE0_HARD002_RESULT_2026-08-17.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str], *, timeout: int | None = None, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(command),
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            errors="replace",
            timeout=timeout,
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
    print(f"[CRCT-QWEN] {stage}: {status}", flush=True)


def _git(*args: str) -> str:
    result = _run(["git", *args], timeout=120)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"])
    return result["stdout"].strip()


def _snapshot_sources(run_dir: Path) -> dict[str, Any]:
    destination = run_dir / "source_snapshot"
    hashes: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"required bridge source missing: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = {"sha256": _sha256(source), "required": True}
    for relative in OPTIONAL_PROVENANCE_FILES:
        source = ROOT / relative
        if source.exists():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative] = {"sha256": _sha256(source), "required": False}
    (destination / "SOURCE_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return hashes


def _collect_environment(run_dir: Path) -> dict[str, Any]:
    diagnostic = run_dir / "diagnostics"
    diagnostic.mkdir(parents=True, exist_ok=True)
    commands: dict[str, list[str]] = {
        "git_status": ["git", "status", "--short"],
        "git_status_tracked": ["git", "status", "--porcelain", "--untracked-files=no"],
        "git_diff": ["git", "diff"],
        "git_diff_cached": ["git", "diff", "--cached"],
        "git_log": ["git", "log", "-20", "--oneline", "--decorate"],
        "git_remote": ["git", "remote", "-v"],
        "python_version": [sys.executable, "--version"],
        "pip_list": [sys.executable, "-m", "pip", "list"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "torch_collect_env": [sys.executable, "-m", "torch.utils.collect_env"],
        "nvidia_smi_full": ["nvidia-smi"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,"
            "temperature.gpu,power.draw,power.limit,utilization.gpu",
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
    probe = """
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
"""
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
            result = _run(
                [
                    ps,
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Processor,Win32_PhysicalMemory,"
                    "Win32_VideoController | Format-List *",
                ],
                timeout=120,
            )
            _write_command(diagnostic / "powershell_hardware.txt", result)
    payload = {"commands": summary, "torch": torch_payload}
    (diagnostic / "environment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _protocol_guard(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    fetch = _run(["git", "fetch", "origin", "--prune"], timeout=300)
    _write_command(run_dir / "diagnostics/git_fetch_origin.txt", fetch)
    head = _git("rev-parse", "HEAD")
    ancestor = _run(["git", "merge-base", "--is-ancestor", config["expected_base_commit"], head])
    tracked_status = _git("status", "--porcelain", "--untracked-files=no")
    remote_refs = [
        line.strip()
        for line in _git("branch", "-r", "--contains", head).splitlines()
        if line.strip().startswith("origin/")
    ]
    blob_checks: dict[str, Any] = {}
    for relative, expected in config["frozen_parent_git_blobs"].items():
        try:
            actual = _git("rev-parse", f"HEAD:{relative}")
        except RuntimeError:
            actual = None
        blob_checks[relative] = {
            "expected_git_blob": expected,
            "actual_git_blob": actual,
            "matches": actual == expected,
        }
    committed_checks: dict[str, Any] = {}
    for relative in config["required_committed_files"]:
        source = ROOT / relative
        try:
            head_blob = _git("rev-parse", f"HEAD:{relative}")
            worktree_blob = _git("hash-object", str(source)) if source.exists() else None
        except RuntimeError:
            head_blob = None
            worktree_blob = None
        committed_checks[relative] = {
            "exists": source.exists(),
            "head_blob": head_blob,
            "worktree_blob": worktree_blob,
            "committed_exactly": bool(head_blob and head_blob == worktree_blob),
        }
    guard = {
        "git_fetch_origin_returncode": fetch["returncode"],
        "head": head,
        "expected_base_is_ancestor": ancestor["returncode"] == 0,
        "tracked_worktree_clean": tracked_status == "",
        "origin_refs_containing_head": remote_refs,
        "head_is_pushed_to_origin": bool(remote_refs),
        "frozen_parent_git_blobs": blob_checks,
        "required_committed_files": committed_checks,
    }
    guard["pass"] = bool(
        fetch["returncode"] == 0
        and guard["expected_base_is_ancestor"]
        and guard["tracked_worktree_clean"]
        and guard["head_is_pushed_to_origin"]
        and all(item["matches"] for item in blob_checks.values())
        and all(item["committed_exactly"] for item in committed_checks.values())
    )
    return guard


def _resource_guard(config: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    torch_payload = environment.get("torch", {})
    free = int(torch_payload.get("cuda_free_bytes", 0) or 0)
    capability = torch_payload.get("cuda_capability") or [0, 0]
    minimum = int(float(config["resource_guard"]["min_cuda_free_gib"]) * 1024**3)
    required_major = int(config["resource_guard"]["required_compute_capability_major"])
    return {
        "cuda_available": bool(torch_payload.get("cuda_available")),
        "cuda_free_bytes": free,
        "minimum_cuda_free_bytes": minimum,
        "cuda_capability": capability,
        "required_compute_capability_major": required_major,
        "pass": bool(
            torch_payload.get("cuda_available")
            and free >= minimum
            and int(capability[0]) >= required_major
        ),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_summary(run_dir: Path, aggregate: dict[str, Any]) -> None:
    ontology = aggregate.get("ontology_v3") or {}
    capital = aggregate.get("capital_dev") or {}
    substrate = aggregate.get("substrate_readiness") or {}
    phase0 = aggregate.get("phase0") or {}
    lines = [
        "# CRCT → Qwen Bridge V1 summary",
        "",
        f"Suite status: `{aggregate['status']}`",
        "",
        "This bundle is a development/eligibility bridge. It contains no protected Qwen outcome.",
        "",
        "## Circuit ontology v3",
        "",
        f"- status: `{ontology.get('status', 'NOT_RUN')}`",
        "- HARD-002 registered status preserved: "
        f"`{ontology.get('registered_suite_status_preserved', 'n/a')}`",
        "- ontology v3 is descriptive and cannot rescue HARD-002's registered negative result.",
        "",
        "## Already-open capital development audit",
        "",
        f"- status: `{capital.get('status', 'NOT_RUN')}`",
        "- no new Qwen forward; no fresh-confirmation claim is permitted.",
        "",
        "## Substrate readiness",
        "",
        f"- status: `{substrate.get('status', 'NOT_RUN')}`",
        "- missing sparse/QK artifacts are not replaced by ad-hoc substitutes.",
        "",
        "## Binding-algebra Phase-0",
        "",
        f"- status: `{phase0.get('status', 'NOT_RUN')}`",
        f"- allowed splits executed: `{phase0.get('allowed_splits_executed', [])}`",
        f"- protected splits executed: `{phase0.get('protected_splits_executed', [])}`",
        "- B2/B3/B4 and protected evaluation remain unauthorized.",
        "",
        "Upload the complete ZIP for adjudication before any later authorization.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(run_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    payload = {
        "schema_version": "crct_qwen_bridge_manifest_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    (run_dir / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def _zip(run_dir: Path) -> Path:
    return Path(
        shutil.make_archive(str(run_dir), "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
    )


def _structured_phase0_status(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "PHASE0_MISSING_OUTPUT"
    status = str(payload.get("status", "UNKNOWN"))
    return {
        "PHASE0_B1_ELIGIBLE_FOR_LATER_B2": "PHASE0_B1_ELIGIBLE",
        "COMPLETED_NEGATIVE_LOCALLY_DIFFERENTIAL": "PHASE0_LOCALLY_DIFFERENTIAL_NEGATIVE",
        "INELIGIBLE_TASK_PHASE0": "PHASE0_INELIGIBLE_TASK",
        "DERIVATIVE_UNAVAILABLE_PHASE0": "PHASE0_DERIVATIVE_UNAVAILABLE",
        "AVAILABILITY_BLOCKED": "PHASE0_AVAILABILITY_BLOCKED",
        "INFRASTRUCTURE_FAILURE": "PHASE0_INFRASTRUCTURE_FAILURE",
    }.get(status, status)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-batch", type=int, default=16)
    parser.add_argument("--replay-batch", type=int, default=8)
    parser.add_argument("--derivative-batch", type=int, default=1)
    parser.add_argument("--phase0-timeout-seconds", type=int, default=21600)
    parser.add_argument("--skip-capital-dev", action="store_true")
    parser.add_argument("--skip-ontology-audit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPORT_ROOT / f"CRCT-QWEN-BRIDGE-001_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    global_status = "INFRASTRUCTURE_FAILURE"
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        (run_dir / "config.source.json").write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        effective = json.loads(json.dumps(config))
        effective["execution"].update(
            {
                "device": args.device,
                "forward_batch": args.forward_batch,
                "replay_batch": args.replay_batch,
                "derivative_batch": args.derivative_batch,
                "phase0_timeout_seconds": args.phase0_timeout_seconds,
            }
        )
        (run_dir / "config.effective.json").write_text(
            json.dumps(effective, indent=2, sort_keys=True), encoding="utf-8"
        )

        source_hashes = _snapshot_sources(run_dir)
        _event(run_dir, "source_snapshot", "complete", file_count=len(source_hashes))

        _event(run_dir, "environment", "started")
        environment = _collect_environment(run_dir)
        _event(run_dir, "environment", "complete")

        tests = [
            "tests/unit/test_circuit_ontology_v3.py",
            "tests/unit/test_qwen_capital_crct_dev.py",
            "tests/unit/test_qwen_substrate_readiness.py",
            "tests/unit/test_qwen_bridge_phase0_guard.py",
            "tests/unit/test_qwen_binding_algebra_protocol.py",
            "tests/unit/test_qwen_binding_algebra_cr_preregistration.py",
        ]
        existing_tests = [item for item in tests if (ROOT / item).exists()]
        _event(run_dir, "pytest", "started", test_count=len(existing_tests))
        test_result = _run(
            [sys.executable, "-m", "pytest", "-q", *existing_tests], timeout=900
        )
        _write_command(run_dir / "logs/pytest.txt", test_result)
        _event(
            run_dir,
            "pytest",
            "complete" if test_result["returncode"] == 0 else "failed",
            returncode=test_result["returncode"],
            elapsed_seconds=test_result["elapsed_seconds"],
        )

        ontology_payload: dict[str, Any] | None = None
        if not args.skip_ontology_audit:
            _event(run_dir, "ontology_v3", "started")
            ontology_result = _run(
                [
                    sys.executable,
                    "-m",
                    "causal_workspace_jepa.experiments.cross_domain.circuit_ontology_v3_audit",
                    "--output",
                    str(run_dir / "metrics/circuit_ontology_v3.json"),
                    "--functional-threshold",
                    str(config["ontology_v3"]["hard002_functional_sufficiency_threshold"]),
                ],
                timeout=300,
            )
            _write_command(run_dir / "logs/ontology_v3.txt", ontology_result)
            ontology_payload = _load_json(run_dir / "metrics/circuit_ontology_v3.json")
            _event(
                run_dir,
                "ontology_v3",
                "complete" if ontology_result["returncode"] == 0 else "failed",
                returncode=ontology_result["returncode"],
            )

        capital_payload: dict[str, Any] | None = None
        if not args.skip_capital_dev:
            _event(run_dir, "capital_dev", "started")
            capital_result = _run(
                [
                    sys.executable,
                    "-m",
                    "causal_workspace_jepa.experiments.llm.qwen_capital_crct_dev",
                    "--shard",
                    str(ROOT / config["capital_dev"]["shard"]),
                    "--expected-sha256",
                    config["capital_dev"]["expected_sha256"],
                    "--output",
                    str(run_dir / "metrics/qwen_capital_crct_dev.json"),
                ],
                timeout=1800,
            )
            _write_command(run_dir / "logs/capital_dev.txt", capital_result)
            capital_payload = _load_json(run_dir / "metrics/qwen_capital_crct_dev.json")
            _event(
                run_dir,
                "capital_dev",
                "complete" if capital_result["returncode"] == 0 else "failed",
                returncode=capital_result["returncode"],
            )

        _event(run_dir, "substrate_readiness", "started")
        substrate_result = _run(
            [
                sys.executable,
                "-m",
                "causal_workspace_jepa.experiments.llm.qwen_substrate_readiness",
                "--output",
                str(run_dir / "metrics/qwen_substrate_readiness.json"),
            ],
            timeout=120,
        )
        _write_command(run_dir / "logs/substrate_readiness.txt", substrate_result)
        substrate_payload = _load_json(run_dir / "metrics/qwen_substrate_readiness.json")
        _event(
            run_dir,
            "substrate_readiness",
            "complete" if substrate_result["returncode"] == 0 else "failed",
            returncode=substrate_result["returncode"],
        )

        protocol_guard = _protocol_guard(config, run_dir)
        (run_dir / "protocol_guard.json").write_text(
            json.dumps(protocol_guard, indent=2, sort_keys=True), encoding="utf-8"
        )
        _event(run_dir, "protocol_guard", "complete" if protocol_guard["pass"] else "blocked")

        resource_guard = _resource_guard(config, environment)
        (run_dir / "resource_guard.json").write_text(
            json.dumps(resource_guard, indent=2, sort_keys=True), encoding="utf-8"
        )
        _event(run_dir, "resource_guard", "complete" if resource_guard["pass"] else "blocked")

        phase0_payload: dict[str, Any] | None = None
        phase0_result: dict[str, Any] | None = None
        if test_result["returncode"] != 0:
            global_status = "TEST_FAILURE_BLOCKED_QWEN"
        elif not protocol_guard["pass"]:
            global_status = "PROTOCOL_GUARD_BLOCKED_QWEN"
        elif not resource_guard["pass"]:
            global_status = "RESOURCE_GUARD_BLOCKED_QWEN"
        else:
            _event(run_dir, "qwen_phase0", "started", protected_splits=False)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONFAULTHANDLER"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            phase0_result = _run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "causal_workspace_jepa.experiments.llm.qwen_binding_algebra_phase0",
                    "--bridge-config",
                    str(args.config),
                    "--parent-config",
                    str(ROOT / "configs/experiments/qwen_binding_algebra_v2.yaml"),
                    "--output",
                    str(run_dir / "metrics/qwen_binding_algebra_phase0.json"),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    args.device,
                    "--forward-batch",
                    str(args.forward_batch),
                    "--replay-batch",
                    str(args.replay_batch),
                    "--derivative-batch",
                    str(args.derivative_batch),
                ],
                timeout=args.phase0_timeout_seconds,
                env=env,
            )
            _write_command(run_dir / "logs/qwen_phase0.txt", phase0_result)
            phase0_payload = _load_json(run_dir / "metrics/qwen_binding_algebra_phase0.json")
            scientific_status = _structured_phase0_status(phase0_payload)
            _event(
                run_dir,
                "qwen_phase0",
                scientific_status,
                returncode=phase0_result["returncode"],
                protected_splits=False,
            )
            global_status = scientific_status

        aggregate = {
            "schema_version": "crct_qwen_bridge_aggregate_v1",
            "experiment_id": config["experiment_id"],
            "status": global_status,
            "tests_returncode": test_result["returncode"],
            "protocol_guard": protocol_guard,
            "resource_guard": resource_guard,
            "ontology_v3": ontology_payload,
            "capital_dev": capital_payload,
            "substrate_readiness": substrate_payload,
            "phase0": phase0_payload,
            "scientific_boundary": {
                "protected_test_executed": False,
                "protected_paraphrase_executed": False,
                "b2_b3_b4_executed": False,
                "capital_is_development_only": True,
                "hard002_registered_status_changed": False,
                "workspace_claim_permitted": False,
            },
        }
        (run_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "SUITE_STATUS.json").write_text(
            json.dumps(
                {
                    "experiment_id": config["experiment_id"],
                    "status": global_status,
                    "run_dir": str(run_dir),
                    "protected_splits_executed": [],
                    "phase0_returncode": phase0_result["returncode"] if phase0_result else None,
                    "source_hashes": source_hashes,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_summary(run_dir, aggregate)
        _event(run_dir, "suite", global_status)
        _manifest(run_dir)
        archive = _zip(run_dir)
        print(
            json.dumps(
                {
                    "status": global_status,
                    "run_dir": str(run_dir),
                    "bundle_zip": str(archive),
                    "protected_splits_executed": [],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if global_status != "PHASE0_INFRASTRUCTURE_FAILURE" else 3
    except Exception as exc:
        failure = {
            "status": "INFRASTRUCTURE_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (run_dir / "FAILURE.txt").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        try:
            _event(run_dir, "suite", "INFRASTRUCTURE_FAILURE", error=str(exc))
            _manifest(run_dir)
            archive = _zip(run_dir)
            print(
                json.dumps(
                    {"status": "INFRASTRUCTURE_FAILURE", "bundle_zip": str(archive)},
                    indent=2,
                )
            )
        except Exception:
            traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
