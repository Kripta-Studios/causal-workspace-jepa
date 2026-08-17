from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

import run_crct_qwen_bridge_suite as base

ROOT = base.ROOT
DEFAULT_CONFIG = ROOT / "configs/experiments/crct_qwen_bridge_v2.json"
REPORT_ROOT = ROOT / "artifacts/reports/crct_qwen_bridge"
OPTIONAL_PROVENANCE_FILES = (
    "artifacts/metrics/qwen_capital_patch_dataset_v1.json",
    "data/manifests/qwen_capital_patches_v1.json",
    "docs/CRCT_STAGE0_HARD002_RESULT_2026-08-17.md",
)


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _snapshot_sources(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    destination = run_dir / "source_snapshot"
    relative_paths = list(dict.fromkeys(config["required_committed_files"]))
    hashes: dict[str, Any] = {}
    for relative in relative_paths:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(f"required Bridge-002 source missing: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = {"sha256": base._sha256(source), "required": True}
    for relative in OPTIONAL_PROVENANCE_FILES:
        source = ROOT / relative
        if source.exists():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[relative] = {"sha256": base._sha256(source), "required": False}
    (destination / "SOURCE_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    return hashes


def _amendment_guard(config: dict[str, Any]) -> dict[str, Any]:
    amendment = config["token_amendment"]
    contract_path = ROOT / amendment["resolved_contract"]
    v3_path = ROOT / amendment["resolved_parent"]
    cr_v2_path = ROOT / amendment["resolved_cr_extension"]
    spec_path = ROOT / amendment["spec"]
    paths = {
        "contract": contract_path,
        "v3": v3_path,
        "cr_v2": cr_v2_path,
        "spec": spec_path,
    }
    existence = {name: path.is_file() for name, path in paths.items()}
    result: dict[str, Any] = {"files_exist": existence, "pass": False}
    if not all(existence.values()):
        result["reason"] = "resolved tokenizer amendment files are missing"
        return result
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        copy_contract = dict(contract)
        expected_self = str(copy_contract.pop("self_sha256"))
        observed_self = _canonical_sha(copy_contract)
        result.update(
            {
                "contract_self_hash_matches": expected_self == observed_self,
                "v3_sha256_matches": base._sha256(v3_path)
                == contract["resolved_v3_sha256"],
                "cr_v2_sha256_matches": base._sha256(cr_v2_path)
                == contract["resolved_cr_v2_sha256"],
                "spec_sha256_matches": base._sha256(spec_path)
                == contract["amendment_spec_sha256"],
                "source_invalid_total": int(contract["source_invalid_total"]),
                "replacement_count": int(contract["replacement_count"]),
                "selection_tokenizer_only": bool(
                    contract["selection_depends_only_on_tokenizer_metadata"]
                ),
                "model_outputs_used": bool(contract["model_outputs_or_logits_used"]),
                "strict_contract_pass": bool(
                    contract["all_resolved_values_strict_spaced_single_token"]
                ),
            }
        )
        result["pass"] = bool(
            result["contract_self_hash_matches"]
            and result["v3_sha256_matches"]
            and result["cr_v2_sha256_matches"]
            and result["spec_sha256_matches"]
            and result["source_invalid_total"] == int(amendment["source_invalid_count"])
            and result["replacement_count"] == int(amendment["source_invalid_count"])
            and result["selection_tokenizer_only"]
            and not result["model_outputs_used"]
            and result["strict_contract_pass"]
        )
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["pass"] = False
    return result


def _structured_phase0_status(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "PHASE0_MISSING_OUTPUT"
    status = str(payload.get("status", "UNKNOWN"))
    return {
        "PHASE0_B1_ELIGIBLE_FOR_LATER_B2": "PHASE0_B1_ELIGIBLE",
        "COMPLETED_NEGATIVE_LOCALLY_DIFFERENTIAL": "PHASE0_LOCALLY_DIFFERENTIAL_NEGATIVE",
        "INELIGIBLE_TASK_PHASE0": "PHASE0_INELIGIBLE_TASK",
        "DERIVATIVE_UNAVAILABLE_PHASE0": "PHASE0_DERIVATIVE_UNAVAILABLE",
        "TOKEN_CONTRACT_BLOCKED_PRE_MODEL": "PHASE0_TOKEN_CONTRACT_BLOCKED_PRE_MODEL",
        "AVAILABILITY_BLOCKED": "PHASE0_AVAILABILITY_BLOCKED",
        "INFRASTRUCTURE_FAILURE": "PHASE0_INFRASTRUCTURE_FAILURE",
    }.get(status, status)


def _write_summary(run_dir: Path, aggregate: dict[str, Any]) -> None:
    phase0 = aggregate.get("phase0") or {}
    amendment = aggregate.get("token_amendment_guard") or {}
    capital = aggregate.get("capital_dev") or {}
    lines = [
        "# CRCT → Qwen Bridge V2 summary",
        "",
        f"Suite status: `{aggregate['status']}`",
        "",
        "Bridge-002 supersedes Bridge-001 only because the V2 answer-token contract was",
        "ineligible before the first competence forward. V2/CR-V1 remain immutable.",
        "",
        "## Tokenizer-only amendment",
        "",
        f"- amendment guard: `{amendment.get('pass', False)}`",
        f"- source invalid values: `{amendment.get('source_invalid_total', 'n/a')}`",
        f"- replacements: `{amendment.get('replacement_count', 'n/a')}`",
        "- thresholds/actions/splits/seeds are not selected from model outcomes.",
        "",
        "## Already-open capital development audit",
        "",
        f"- status: `{capital.get('status', 'NOT_RUN')}`",
        "- this remains development-only and cannot confirm the new binding task.",
        "",
        "## Binding-algebra Phase-0",
        "",
        f"- status: `{phase0.get('status', 'NOT_RUN')}`",
        f"- splits materialized: `{phase0.get('allowed_splits_materialized', [])}`",
        "- model-forward execution started: "
        f"`{phase0.get('model_forward_execution_started', False)}`",
        f"- model-forward splits completed: `{phase0.get('model_forward_splits_completed', [])}`",
        f"- protected splits executed: `{phase0.get('protected_splits_executed', [])}`",
        "- B2/B3/B4 and protected evaluation remain unauthorized.",
        "",
        "Upload this complete ZIP for independent adjudication before any later authorization.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    run_dir = REPORT_ROOT / f"CRCT-QWEN-BRIDGE-002_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    global_status = "INFRASTRUCTURE_FAILURE"
    source_hashes: dict[str, Any] = {}
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

        source_hashes = _snapshot_sources(run_dir, config)
        base._event(run_dir, "source_snapshot", "complete", file_count=len(source_hashes))

        base._event(run_dir, "environment", "started")
        environment = base._collect_environment(run_dir)
        base._event(run_dir, "environment", "complete")

        tests = [
            "tests/unit/test_qwen_binding_algebra_v3_token_amendment.py",
            "tests/unit/test_qwen_bridge_phase0_v2_guard.py",
            "tests/unit/test_qwen_bridge_phase0_guard.py",
            "tests/unit/test_qwen_capital_crct_dev.py",
            "tests/unit/test_qwen_binding_algebra_protocol.py",
            "tests/unit/test_qwen_binding_algebra_cr_preregistration.py",
            "tests/unit/test_circuit_ontology_v3.py",
            "tests/unit/test_qwen_substrate_readiness.py",
        ]
        existing_tests = [item for item in tests if (ROOT / item).exists()]
        base._event(run_dir, "pytest", "started", test_count=len(existing_tests))
        test_result = base._run(
            [sys.executable, "-m", "pytest", "-q", *existing_tests], timeout=1200
        )
        base._write_command(run_dir / "logs/pytest.txt", test_result)
        base._event(
            run_dir,
            "pytest",
            "complete" if test_result["returncode"] == 0 else "failed",
            returncode=test_result["returncode"],
        )

        amendment_guard = _amendment_guard(config)
        (run_dir / "token_amendment_guard.json").write_text(
            json.dumps(amendment_guard, indent=2, sort_keys=True), encoding="utf-8"
        )
        base._event(
            run_dir,
            "token_amendment_guard",
            "complete" if amendment_guard["pass"] else "blocked",
        )

        ontology_payload: dict[str, Any] | None = None
        if not args.skip_ontology_audit:
            base._event(run_dir, "ontology_v3", "started")
            result = base._run(
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
            base._write_command(run_dir / "logs/ontology_v3.txt", result)
            ontology_payload = base._load_json(run_dir / "metrics/circuit_ontology_v3.json")
            base._event(
                run_dir,
                "ontology_v3",
                "complete" if result["returncode"] == 0 else "failed",
            )

        capital_payload: dict[str, Any] | None = None
        if not args.skip_capital_dev:
            base._event(run_dir, "capital_dev", "started")
            result = base._run(
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
            base._write_command(run_dir / "logs/capital_dev.txt", result)
            capital_payload = base._load_json(run_dir / "metrics/qwen_capital_crct_dev.json")
            base._event(
                run_dir,
                "capital_dev",
                "complete" if result["returncode"] == 0 else "failed",
            )

        base._event(run_dir, "substrate_readiness", "started")
        result = base._run(
            [
                sys.executable,
                "-m",
                "causal_workspace_jepa.experiments.llm.qwen_substrate_readiness",
                "--output",
                str(run_dir / "metrics/qwen_substrate_readiness.json"),
            ],
            timeout=120,
        )
        base._write_command(run_dir / "logs/substrate_readiness.txt", result)
        substrate_payload = base._load_json(run_dir / "metrics/qwen_substrate_readiness.json")
        base._event(
            run_dir,
            "substrate_readiness",
            "complete" if result["returncode"] == 0 else "failed",
        )

        protocol_guard = base._protocol_guard(config, run_dir)
        (run_dir / "protocol_guard.json").write_text(
            json.dumps(protocol_guard, indent=2, sort_keys=True), encoding="utf-8"
        )
        base._event(run_dir, "protocol_guard", "complete" if protocol_guard["pass"] else "blocked")

        resource_guard = base._resource_guard(config, environment)
        (run_dir / "resource_guard.json").write_text(
            json.dumps(resource_guard, indent=2, sort_keys=True), encoding="utf-8"
        )
        base._event(run_dir, "resource_guard", "complete" if resource_guard["pass"] else "blocked")

        phase0_payload: dict[str, Any] | None = None
        phase0_result: dict[str, Any] | None = None
        if test_result["returncode"] != 0:
            global_status = "TEST_FAILURE_BLOCKED_QWEN"
        elif not amendment_guard["pass"]:
            global_status = "TOKEN_AMENDMENT_GUARD_BLOCKED_QWEN"
        elif not protocol_guard["pass"]:
            global_status = "PROTOCOL_GUARD_BLOCKED_QWEN"
        elif not resource_guard["pass"]:
            global_status = "RESOURCE_GUARD_BLOCKED_QWEN"
        else:
            base._event(run_dir, "qwen_phase0_v2", "started", protected_splits=False)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONFAULTHANDLER"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            phase0_result = base._run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-m",
                    "causal_workspace_jepa.experiments.llm.qwen_binding_algebra_phase0_v2",
                    "--bridge-config",
                    str(args.config),
                    "--parent-config",
                    str(ROOT / config["token_amendment"]["resolved_parent"]),
                    "--output",
                    str(run_dir / "metrics/qwen_binding_algebra_phase0_v2.json"),
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
            base._write_command(run_dir / "logs/qwen_phase0_v2.txt", phase0_result)
            phase0_payload = base._load_json(
                run_dir / "metrics/qwen_binding_algebra_phase0_v2.json"
            )
            global_status = _structured_phase0_status(phase0_payload)
            base._event(
                run_dir,
                "qwen_phase0_v2",
                global_status,
                returncode=phase0_result["returncode"],
                protected_splits=False,
            )

        aggregate = {
            "schema_version": "crct_qwen_bridge_aggregate_v2",
            "experiment_id": config["experiment_id"],
            "status": global_status,
            "tests_returncode": test_result["returncode"],
            "token_amendment_guard": amendment_guard,
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
                "v2_cr_v1_disposition_changed": False,
            },
        }
        (run_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_summary(run_dir, aggregate)
        suite_status = {
            "experiment_id": config["experiment_id"],
            "status": global_status,
            "run_dir": str(run_dir),
            "phase0_returncode": phase0_result["returncode"] if phase0_result else None,
            "model_forward_splits_completed": (
                phase0_payload.get("model_forward_splits_completed", [])
                if phase0_payload
                else []
            ),
            "protected_splits_executed": [],
            "source_hashes": source_hashes,
        }
        (run_dir / "SUITE_STATUS.json").write_text(
            json.dumps(suite_status, indent=2, sort_keys=True), encoding="utf-8"
        )
        base._event(run_dir, "suite", global_status)
    except Exception as exc:
        global_status = "INFRASTRUCTURE_FAILURE"
        (run_dir / "FAILURE.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", encoding="utf-8"
        )
        if not (run_dir / "SUITE_STATUS.json").exists():
            (run_dir / "SUITE_STATUS.json").write_text(
                json.dumps(
                    {
                        "experiment_id": "CRCT-QWEN-BRIDGE-002",
                        "status": global_status,
                        "run_dir": str(run_dir),
                        "protected_splits_executed": [],
                        "source_hashes": source_hashes,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        base._event(run_dir, "suite", global_status, error_type=type(exc).__name__)
    finally:
        try:
            base._manifest(run_dir)
            bundle = base._zip(run_dir)
        except Exception as exc:
            print(f"FAIL-SAFE BUNDLE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3

    payload = base._load_json(run_dir / "SUITE_STATUS.json") or {
        "status": global_status,
        "run_dir": str(run_dir),
    }
    payload["bundle_zip"] = str(bundle)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if global_status in {
        "INFRASTRUCTURE_FAILURE",
        "TEST_FAILURE_BLOCKED_QWEN",
        "TOKEN_AMENDMENT_GUARD_BLOCKED_QWEN",
        "PROTOCOL_GUARD_BLOCKED_QWEN",
        "RESOURCE_GUARD_BLOCKED_QWEN",
        "PHASE0_INFRASTRUCTURE_FAILURE",
    }:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
