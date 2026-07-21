#!/usr/bin/env python
"""Run and aggregate all frozen EB-JEPA competence jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_workspace_jepa.experiments.world_model.eb_jepa_competence_protocol import (  # noqa: E402
    aggregate_competence,
)
from causal_workspace_jepa.experiments.world_model.eb_jepa_training_protocol import (  # noqa: E402
    required_checkpoint_names,
)
from causal_workspace_jepa.common.provenance import (  # noqa: E402
    collect_provenance,
    write_provenance,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _paired_summary(rows: list[dict], seeds: list[int]) -> dict:
    keyed: dict[tuple[int, int, int, str], dict] = {
        (
            int(row["training_seed"]),
            int(row["checkpoint_epoch"]),
            int(row["episode"]),
            str(row["arm"]),
        ): row
        for row in rows
    }
    pairs = []
    for seed in seeds:
        seed_keys = sorted(
            key[:3]
            for key in keyed
            if key[0] == seed and key[3] == "official_mppi_as_executed"
        )
        for prefix in seed_keys:
            official = keyed[prefix + ("official_mppi_as_executed",)]
            bounded = keyed[prefix + ("bound_corrected_mppi_as_executed",)]
            pairs.append(
                {
                    "training_seed": seed,
                    "checkpoint_epoch": prefix[1],
                    "episode": prefix[2],
                    "success_difference": float(bounded["success"])
                    - float(official["success"]),
                    "distance_difference": float(bounded["final_state_distance"])
                    - float(official["final_state_distance"]),
                }
            )
    per_seed = {
        str(seed): sum(row["success_difference"] for row in pairs if row["training_seed"] == seed)
        / sum(row["training_seed"] == seed for row in pairs)
        for seed in seeds
    }
    rng = random.Random(730019)
    bootstrap = []
    for _ in range(10000):
        sampled = [rng.choice(seeds) for _ in seeds]
        values = [float(per_seed[str(seed)]) for seed in sampled]
        bootstrap.append(sum(values) / len(values))
    bootstrap.sort()
    return {
        "paired_rows": len(pairs),
        "bounded_minus_official_success": sum(row["success_difference"] for row in pairs)
        / len(pairs),
        "bounded_minus_official_final_distance": sum(
            row["distance_difference"] for row in pairs
        )
        / len(pairs),
        "per_seed_success_difference": per_seed,
        "seed_cluster_bootstrap_95": [bootstrap[249], bootstrap[9749]],
        "bootstrap_seed": 730019,
        "bootstrap_replicates": 10000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--training-config",
        default="configs/experiments/eb_jepa_two_rooms_training.yaml",
    )
    args = parser.parse_args()

    import yaml

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training_config_path = Path(args.training_config).resolve()
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    repo_commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain"):
        raise RuntimeError("repository must be clean before competence portfolio")
    provenance = collect_provenance(
        command=(
            "python scripts/run_eb_jepa_competence_portfolio.py "
            f"--config {Path(args.config).as_posix()}"
        ),
        resource_profile=str(config["resource_profile"]),
        seed=None,
    )
    interpreter = (REPO_ROOT / config["interpreter"]).resolve()
    source_root = (REPO_ROOT / config["source_root"]).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(REPO_ROOT / "src")]
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    status_path = REPO_ROOT / ".cache" / "runs" / "eb_jepa_competence_status.json"
    jobs = [
        (seed, epoch, arm)
        for seed in config["training_seeds"]
        for epoch in config["checkpoint_epochs"]
        for arm in config["planner_arms"]
    ]
    portfolio = {
        "status": "RUNNING",
        "experiment_id": config["id"],
        "repo_commit": repo_commit,
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "active_job": None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write(status_path, portfolio)
    started = time.perf_counter()
    for seed, epoch, arm in jobs:
        portfolio["active_job"] = {"seed": seed, "epoch": epoch, "arm": arm}
        _write(status_path, portfolio)
        result = subprocess.run(
            [
                str(interpreter),
                str(REPO_ROOT / "scripts" / "evaluate_eb_jepa_competence_job.py"),
                "--config",
                str(config_path),
                "--training-seed",
                str(seed),
                "--epoch",
                str(epoch),
                "--arm",
                arm,
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            portfolio.update(
                {
                    "status": "FAILED",
                    "failed_job": portfolio["active_job"],
                    "returncode": result.returncode,
                }
            )
            _write(status_path, portfolio)
            return result.returncode
        portfolio["completed_jobs"] += 1

    job_root = REPO_ROOT / config["job_output_root"]
    job_payloads = [
        json.loads(
            (
                job_root / f"seed-{seed}" / f"epoch-{epoch}" / f"{arm}.json"
            ).read_text(encoding="utf-8")
        )
        for seed, epoch, arm in jobs
    ]
    rows = [episode for payload in job_payloads for episode in payload["episodes"]]
    aggregate = aggregate_competence(
        rows,
        seeds=config["training_seeds"],
        required_arms=config["planner_arms"],
        overall_threshold=float(config["overall_success_threshold"]),
        per_seed_threshold=float(config["per_seed_success_threshold"]),
    )
    bounded = aggregate["arm_summaries"]["bound_corrected_mppi_as_executed"]
    engineering_passes = {
        "all_frozen_jobs_complete": len(job_payloads) == len(jobs)
        and all(payload["status"] == "COMPLETED" for payload in job_payloads),
        "all_expected_episode_rows": len(rows)
        == len(jobs) * int(config["num_episodes"]),
        "clean_pinned_source_and_repo": all(
            payload["repo_commit"] == repo_commit
            and payload["source_revision"] == config["revision"]
            and not payload["repo_dirty_at_start"]
            and payload["source_clean"]
            for payload in job_payloads
        ),
        "bounded_arm_respects_action_contract": bounded[
            "executed_action_violation_count"
        ]
        == 0
        and bounded["max_executed_action_norm"] <= float(config["action_max_norm"]) + 1e-6,
    }
    paired = _paired_summary(rows, [int(seed) for seed in config["training_seeds"]])
    competence_metrics = {
        "experiment_id": config["id"],
        "status": (
            "COMPETENT_BOUNDED"
            if aggregate["all_required_arms_competent"] and all(engineering_passes.values())
            else "COMPLETED_INELIGIBLE"
        ),
        "evidence_level": (
            "Generalization" if aggregate["all_required_arms_competent"] else "Availability"
        ),
        "repo_commit": repo_commit,
        "source_revision": config["revision"],
        "training_seeds": config["training_seeds"],
        "checkpoint_epochs": config["checkpoint_epochs"],
        "planner_arms": config["planner_arms"],
        "engineering_passes": engineering_passes,
        "all_engineering_passes": all(engineering_passes.values()),
        "competence": aggregate,
        "paired_planner_effect": paired,
        "job_summaries": [
            {
                "training_seed": payload["training_seed"],
                "checkpoint_epoch": payload["checkpoint_epoch"],
                "arm": payload["arm"],
                "summary": payload["summary"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "wall_seconds": payload["wall_seconds"],
            }
            for payload in job_payloads
        ],
        "episode_rows": rows,
        "wall_seconds": time.perf_counter() - started,
        "claim": (
            "This evaluates trained official EB-JEPA checkpoints under separately labeled "
            "as-executed and bound-corrected MPPI arms. Mechanistic eligibility requires the "
            "bounded arm to pass frozen multi-seed competence gates."
        ),
        "scientific_boundary": (
            "Planning competence and paired planner differences do not identify a recurrent "
            "circuit or workspace; those require new preregistered interventions."
        ),
    }
    metrics_path = REPO_ROOT / config["output_metrics"]
    _write(metrics_path, competence_metrics)
    write_provenance(
        metrics_path.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": Path(config["output_metrics"]).as_posix(),
            "all_passed": all(engineering_passes.values()),
        },
    )

    training_statuses = []
    checkpoint_root = REPO_ROOT / training_config["output_root"]
    for seed in training_config["seeds"]:
        training_statuses.append(
            json.loads(
                (checkpoint_root / f"seed-{seed}" / "training_status.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    training_complete = all(
        status["status"] == "COMPLETED"
        and set(status["checkpoint_manifest"])
        == set(required_checkpoint_names(int(training_config["epochs"])))
        for status in training_statuses
    )
    training_metrics = {
        "experiment_id": training_config["id"],
        "status": "TRAINING_COMPLETED" if training_complete else "TRAINING_INCOMPLETE",
        "evidence_level": "Availability",
        "all_passed": training_complete,
        "source_revision": training_config["revision"],
        "seeds": training_config["seeds"],
        "seed_summaries": training_statuses,
        "scientific_boundary": (
            "Checkpoint completion and hashes alone provide no competence, circuit, or workspace evidence."
        ),
    }
    training_metrics_path = REPO_ROOT / training_config["output_metrics"]
    _write(training_metrics_path, training_metrics)
    write_provenance(
        training_metrics_path.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": Path(training_config["output_metrics"]).as_posix(),
            "all_passed": training_complete,
            "derived_during": config["id"],
        },
    )
    portfolio.update(
        {
            "status": "COMPLETED",
            "active_job": None,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": str(metrics_path),
        }
    )
    _write(status_path, portfolio)
    print(json.dumps(competence_metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
