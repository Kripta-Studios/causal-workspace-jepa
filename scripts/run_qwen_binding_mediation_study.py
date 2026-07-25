#!/usr/bin/env python
"""Run one fail-closed phase of Qwen binding mediation v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_study import (
    QwenProtectedOutcomeExecutor,
    run_calibration_phase,
    run_protected_evaluation,
    run_train_plan_phase,
    sha256_file,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", required=True, choices=("calibration", "train_plan", "protected_eval")
    )
    parser.add_argument("--calibration")
    parser.add_argument("--plan")
    parser.add_argument("--output")
    parser.add_argument("--progress")
    parser.add_argument("--max-gpu-hours", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    command = "python scripts/run_qwen_binding_mediation_study.py " + " ".join(
        [
            f"--config {Path(args.config).as_posix()}",
            f"--phase {args.phase}",
            *([f"--calibration {Path(args.calibration).as_posix()}"] if args.calibration else []),
            *([f"--plan {Path(args.plan).as_posix()}"] if args.plan else []),
            *([f"--output {Path(args.output).as_posix()}"] if args.output else []),
            *([f"--progress {Path(args.progress).as_posix()}"] if args.progress else []),
            *([f"--max-gpu-hours {args.max_gpu_hours}"] if args.max_gpu_hours is not None else []),
        ]
    )
    provenance = collect_provenance(
        command=command,
        resource_profile=str(config["resource_profile"]),
        seed=int(config["seed"]),
    )
    if provenance.git_dirty:
        raise RuntimeError("binding study phases require a clean committed worktree")

    if args.phase == "calibration":
        if not args.output:
            parser.error("calibration requires --output")
        result = run_calibration_phase(
            args.config,
            args.output,
            source_git_commit=provenance.git_commit,
        )
        artifact = Path(args.output)
    elif args.phase == "train_plan":
        if not args.calibration or not args.plan or args.max_gpu_hours is None:
            parser.error("train_plan requires --calibration, --plan, and --max-gpu-hours")
        result = run_train_plan_phase(
            args.config,
            args.calibration,
            args.plan,
            max_gpu_hours=args.max_gpu_hours,
            source_git_commit=provenance.git_commit,
        )
        artifact = Path(args.plan)
    else:
        if not all(
            value is not None
            for value in (
                args.calibration,
                args.plan,
                args.output,
                args.progress,
                args.max_gpu_hours,
            )
        ):
            parser.error(
                "protected_eval requires --calibration, --plan, --output, --progress, "
                "and --max-gpu-hours"
            )
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        result = run_protected_evaluation(
            args.config,
            args.plan,
            QwenProtectedOutcomeExecutor(args.config),
            max_gpu_hours=args.max_gpu_hours,
            calibration=calibration,
            progress_path=args.progress,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        artifact = output
    provenance_extra = {
        "artifact": artifact.as_posix(),
        "artifact_sha256": sha256_file(artifact),
        "phase": args.phase,
    }
    if "execution_runtime" in result:
        provenance_extra["execution_runtime"] = result["execution_runtime"]
        provenance_extra["capture_identity"] = result["capture_identity"]
        provenance_extra["source_git_commit"] = result["source_git_commit"]
        provenance_extra["execution_git_commit"] = result["execution_git_commit"]
    write_provenance(
        artifact.with_suffix(".provenance.json"),
        provenance,
        extra=provenance_extra,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
