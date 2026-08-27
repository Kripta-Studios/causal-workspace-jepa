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
DEFAULT_CONFIG = ROOT / "configs/experiments/qwen_competence_recovery_v1.json"
REPORT_ROOT = ROOT / "artifacts/reports/qwen_competence_recovery"

SOURCE_FILES = (
    "configs/experiments/qwen_competence_recovery_v1.json",
    "configs/experiments/qwen_binding_algebra_v3.yaml",
    "configs/experiments/qwen_binding_algebra_cr_v2.yaml",
    "configs/experiments/qwen_binding_algebra_v3_token_contract.json",
    "docs/QWEN_BINDING_ALGEBRA_V3_B0_ADJUDICATION_2026-08-18.md",
    "docs/QWEN_COMPETENCE_RECOVERY_V1_PROTOCOL.md",
    "scripts/run_qwen_competence_recovery.ps1",
    "scripts/run_qwen_competence_recovery_suite.py",
    "src/causal_workspace_jepa/experiments/llm/qwen_binding_competence_recovery.py",
    "src/causal_workspace_jepa/experiments/llm/qwen_binding_algebra_protocol.py",
    "tests/unit/test_qwen_binding_competence_recovery.py",
    "tests/unit/test_qwen_competence_recovery_entrypoints.py",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(command),
            cwd=ROOT,
            text=True,
            capture_output=True,
            errors="replace",
            timeout=timeout,
            env=os.environ.copy(),
        )
        return {
            "command": list(command),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
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
    except FileNotFoundError as exc:
        return {
            "command": list(command),
            "returncode": 127,
            "stdout": "",
            "stderr": f"FileNotFoundError: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        }


def _write_command(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
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
        ),
        encoding="utf-8",
        errors="replace",
    )


def _event(run_dir: Path, stage: str, status: str, **extra: Any) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **extra,
    }
    with (run_dir / "STAGE_EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"[QWEN-COMP] {stage}: {status}", flush=True)


def _git(*args: str) -> str:
    result = _run(["git", *args], timeout=120)
    if result["returncode"] != 0:
        raise RuntimeError(result["stderr"] or result["stdout"])
    return result["stdout"].strip()


def _snapshot(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"required source missing: {relative}")
        target = run_dir / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = _sha(source)
    (run_dir / "source_snapshot/SOURCE_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hashes


def _environment(run_dir: Path) -> None:
    commands = {
        "git_status": ["git", "status", "--short"],
        "git_status_tracked": ["git", "status", "--porcelain", "--untracked-files=no"],
        "git_log": ["git", "log", "-10", "--oneline", "--decorate"],
        "git_remote": ["git", "remote", "-v"],
        "python_version": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "torch_env": [sys.executable, "-m", "torch.utils.collect_env"],
        "nvidia_smi": ["nvidia-smi"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,"
            "temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    }
    for name, command in commands.items():
        result = _run(command, timeout=120)
        _write_command(run_dir / f"diagnostics/{name}.txt", result)


def _guard(config: dict[str, Any]) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    base_ok = (
        _run(
            ["git", "merge-base", "--is-ancestor", config["expected_base_commit"], head],
            timeout=120,
        )["returncode"]
        == 0
    )
    tracked_clean = _git("status", "--porcelain", "--untracked-files=no") == ""
    refs = [
        line.strip()
        for line in _git("branch", "-r", "--contains", head).splitlines()
        if line.strip().startswith("origin/")
    ]
    required = {}
    for relative in SOURCE_FILES:
        source = ROOT / relative
        head_blob = _git("rev-parse", f"HEAD:{relative}")
        worktree_blob = _git("hash-object", str(source))
        required[relative] = {
            "head_blob": head_blob,
            "worktree_blob": worktree_blob,
            "committed_exactly": head_blob == worktree_blob,
        }
    payload = {
        "head": head,
        "expected_base_is_ancestor": base_ok,
        "tracked_worktree_clean": tracked_clean,
        "origin_refs_containing_head": refs,
        "head_is_pushed_to_origin": bool(refs),
        "required_source_checks": required,
    }
    payload["pass"] = bool(
        base_ok
        and tracked_clean
        and refs
        and all(item["committed_exactly"] for item in required.values())
    )
    return payload


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary(run_dir: Path, aggregate: dict[str, Any]) -> None:
    result = aggregate.get("result") or {}
    variants = result.get("prompt_variants") or {}
    lines = [
        "# Qwen Binding Competence Recovery V1",
        "",
        f"Suite status: `{aggregate['status']}`",
        "",
        "Calibration-only development experiment. No train/validation/test/paraphrase "
        "model forward is authorized.",
        "",
        f"Selected prompt: `{result.get('selected_prompt_variant_id')}`",
        "",
        "| variant | eligible | clean full | direct full | clean cand | direct cand |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant_id, metrics in variants.items():
        if metrics.get("status") != "EVALUATED":
            lines.append(f"| {variant_id} | unavailable | - | - | - | - |")
            continue
        lines.append(
            "| {vid} | {eligible} | {cf:.4f} | {df:.4f} | {cc:.4f} | {dc:.4f} |".format(
                vid=variant_id,
                eligible=metrics.get("eligible", False),
                cf=metrics["clean"]["full_vocab_accuracy"],
                df=metrics["direct_permuted"]["full_vocab_accuracy"],
                cc=metrics["clean"]["candidate_only_accuracy_diagnostic"],
                dc=metrics["direct_permuted"]["candidate_only_accuracy_diagnostic"],
            )
        )
    lines.extend(
        [
            "",
            f"Model-forward splits: `{result.get('model_forward_splits_executed', [])}`",
            f"Protected splits: `{result.get('protected_splits_executed', [])}`",
            "",
            "Do not execute a new validation split from this runner. Upload the ZIP for "
            "independent adjudication first.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            rows.append(
                {
                    "path": str(path.relative_to(run_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
    (run_dir / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "qwen_competence_recovery_manifest_v1",
                "file_count": len(rows),
                "files": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _zip(run_dir: Path) -> Path:
    return Path(
        shutil.make_archive(
            str(run_dir),
            "zip",
            root_dir=run_dir.parent,
            base_dir=run_dir.name,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPORT_ROOT / f"QWEN-BINDING-COMPETENCE-RECOVERY-001_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status = "INFRASTRUCTURE_FAILURE"
    result_payload: dict[str, Any] | None = None
    source_hashes: dict[str, str] = {}
    guard: dict[str, Any] | None = None
    test_result: dict[str, Any] | None = None
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        (run_dir / "config.source.json").write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        source_hashes = _snapshot(run_dir)
        _event(run_dir, "source_snapshot", "complete", count=len(source_hashes))
        _event(run_dir, "environment", "started")
        _environment(run_dir)
        _event(run_dir, "environment", "complete")

        _event(run_dir, "pytest", "started")
        test_result = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_qwen_binding_competence_recovery.py",
                "tests/unit/test_qwen_binding_algebra_protocol.py",
                "tests/unit/test_qwen_binding_algebra_v3_token_amendment.py",
            ],
            timeout=900,
        )
        _write_command(run_dir / "logs/pytest.txt", test_result)
        _event(
            run_dir,
            "pytest",
            "complete" if test_result["returncode"] == 0 else "failed",
            returncode=test_result["returncode"],
        )

        guard = _guard(config)
        (run_dir / "protocol_guard.json").write_text(
            json.dumps(guard, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _event(run_dir, "protocol_guard", "complete" if guard["pass"] else "blocked")

        if test_result["returncode"] != 0:
            status = "TEST_FAILURE_BLOCKED"
        elif not guard["pass"]:
            status = "PROTOCOL_GUARD_BLOCKED"
        else:
            _event(run_dir, "competence_recovery", "started")
            output = run_dir / "metrics/qwen_binding_competence_recovery.json"
            result = _run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "causal_workspace_jepa.experiments.llm.qwen_binding_competence_recovery",
                    "--config",
                    str(args.config),
                    "--output",
                    str(output),
                    "--run-dir",
                    str(run_dir),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ],
                timeout=args.timeout_seconds,
            )
            _write_command(run_dir / "logs/competence_recovery.txt", result)
            result_payload = _load(output)
            status = str(
                (result_payload or {}).get(
                    "status",
                    "INFRASTRUCTURE_FAILURE",
                )
            )
            _event(
                run_dir,
                "competence_recovery",
                status,
                returncode=result["returncode"],
            )
    except Exception as exc:
        (run_dir / "FAILURE.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _event(
            run_dir,
            "suite_exception",
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        status = "INFRASTRUCTURE_FAILURE"
    finally:
        aggregate = {
            "schema_version": "qwen_competence_recovery_aggregate_v1",
            "experiment_id": "QWEN-BINDING-COMPETENCE-RECOVERY-001",
            "status": status,
            "source_hashes": source_hashes,
            "protocol_guard": guard,
            "tests_returncode": None if test_result is None else test_result["returncode"],
            "result": result_payload,
            "protected_splits_executed": [],
        }
        (run_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _summary(run_dir, aggregate)
        (run_dir / "SUITE_STATUS.json").write_text(
            json.dumps(
                {
                    "experiment_id": aggregate["experiment_id"],
                    "status": status,
                    "run_dir": str(run_dir),
                    "model_forward_splits_executed": (
                        (result_payload or {}).get("model_forward_splits_executed", [])
                    ),
                    "protected_splits_executed": [],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _event(run_dir, "suite", status)
        _manifest(run_dir)
        bundle = _zip(run_dir)
        print(
            json.dumps(
                {
                    "status": status,
                    "run_dir": str(run_dir),
                    "bundle_zip": str(bundle),
                    "protected_splits_executed": [],
                },
                indent=2,
            )
        )
    return 0 if status in {
        "COMPETENCE_RECOVERY_PROMPT_SELECTED",
        "COMPETENCE_RECOVERY_FAILED",
    } else 3


if __name__ == "__main__":
    raise SystemExit(main())
