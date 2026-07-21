#!/usr/bin/env python
"""Run the three preregistered official EB-JEPA training seeds sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    interpreter = (repo_root / config["interpreter"]).resolve()
    source_root = (repo_root / config["source_root"]).resolve()
    status_path = repo_root / ".cache" / "runs" / "eb_jepa_training_portfolio_status.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), str(repo_root / "src")]
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["WANDB_DISABLED"] = "true"
    environment["MPLBACKEND"] = "Agg"
    portfolio = {
        "status": "RUNNING",
        "experiment_id": config["id"],
        "seeds": config["seeds"],
        "completed_seeds": [],
        "active_seed": None,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write(status_path, portfolio)
    for seed in config["seeds"]:
        portfolio["active_seed"] = seed
        _write(status_path, portfolio)
        result = subprocess.run(
            [
                str(interpreter),
                str(repo_root / "scripts" / "train_eb_jepa_two_rooms_seed.py"),
                "--config",
                str(config_path),
                "--seed",
                str(seed),
            ],
            cwd=repo_root,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            portfolio.update(
                {
                    "status": "FAILED",
                    "failed_seed": seed,
                    "returncode": result.returncode,
                    "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            _write(status_path, portfolio)
            return result.returncode
        portfolio["completed_seeds"].append(seed)
    portfolio.update(
        {
            "status": "COMPLETED",
            "active_seed": None,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _write(status_path, portfolio)
    print(json.dumps(portfolio, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
