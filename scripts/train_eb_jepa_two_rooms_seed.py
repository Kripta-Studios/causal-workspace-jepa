#!/usr/bin/env python
"""Train or safely resume one pinned official EB-JEPA Two Rooms seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_workspace_jepa.experiments.world_model.eb_jepa_training_protocol import (  # noqa: E402
    required_checkpoint_names,
    validate_source_training_config,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    import torch
    import yaml
    from omegaconf import OmegaConf

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    if args.seed not in seeds:
        raise ValueError(f"seed {args.seed} is not preregistered in {seeds}")
    source_root = (REPO_ROOT / config["source_root"]).resolve()
    revision = _git(source_root, "rev-parse", "HEAD")
    source_clean = not bool(_git(source_root, "status", "--porcelain"))
    if revision != config["revision"] or not source_clean:
        raise RuntimeError("official source does not match the clean pinned revision")
    repo_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    repo_dirty = bool(_git(REPO_ROOT, "status", "--porcelain"))
    if repo_dirty:
        raise RuntimeError("repository must be clean before training starts")

    upstream_config_path = (source_root / config["upstream_training_config"]).resolve()
    upstream_config = OmegaConf.load(upstream_config_path)
    source_values = validate_source_training_config(
        OmegaConf.to_container(upstream_config, resolve=True)
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_TRAINED",
                    "seed": args.seed,
                    "repo_commit": repo_commit,
                    "source_revision": revision,
                    "source_config": source_values,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output_root = (REPO_ROOT / config["output_root"]).resolve()
    run_directory = output_root / f"seed-{args.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    status_path = run_directory / "training_status.json"
    existing_status = None
    if status_path.exists():
        existing_status = json.loads(status_path.read_text(encoding="utf-8"))
        if existing_status.get("status") == "COMPLETED":
            print(json.dumps(existing_status, indent=2, sort_keys=True))
            return 0

    latest_checkpoint = run_directory / "latest.pth.tar"
    resume = latest_checkpoint.exists()
    cfg = OmegaConf.load(upstream_config_path)
    cfg.logging.log_wandb = False
    cfg.logging.tqdm_silent = True
    cfg.meta.seed = args.seed
    cfg.meta.load_model = resume
    cfg.meta.enable_plan_eval = False
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = {
        "status": "RUNNING",
        "experiment_id": config["id"],
        "seed": args.seed,
        "repo_commit": repo_commit,
        "repo_dirty_at_start": repo_dirty,
        "source_revision": revision,
        "source_clean": source_clean,
        "started_utc": started_utc,
        "resume_from_latest": resume,
        "source_config": source_values,
        "workflow_deviations": {
            "torch_runtime": "2.10.0+cu128 instead of pinned 2.6.0+cu126 for SM120",
            "wandb_disabled": True,
            "inline_planning_and_unroll_eval_disabled": True,
            "checkpoint_evaluation_deferred": True,
            "compile_flag_retained_but_profile_observed_zero_dynamo_graphs": True,
        },
    }
    _write_status(status_path, status)
    started = time.perf_counter()
    try:
        from examples.ac_video_jepa.main import run

        run(cfg=cfg, folder=run_directory)
        checkpoint_names = required_checkpoint_names(int(cfg.optim.epochs))
        missing = [name for name in checkpoint_names if not (run_directory / name).exists()]
        if missing:
            raise RuntimeError(f"training returned without required checkpoints: {missing}")
        checkpoint_manifest = {
            name: {
                "bytes": (run_directory / name).stat().st_size,
                "sha256": _sha256(run_directory / name),
            }
            for name in checkpoint_names
        }
        status.update(
            {
                "status": "COMPLETED",
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_seconds": time.perf_counter() - started,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "cuda_runtime": torch.version.cuda,
                    "device": torch.cuda.get_device_name(0),
                },
                "checkpoint_manifest": checkpoint_manifest,
                "mechanism_claim": False,
            }
        )
        _write_status(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        status.update(
            {
                "status": "FAILED",
                "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback_tail": traceback.format_exc()[-8000:],
            }
        )
        _write_status(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
