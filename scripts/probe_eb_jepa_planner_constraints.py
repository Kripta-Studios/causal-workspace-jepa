#!/usr/bin/env python
"""Dynamic and source-contract audit of official CEM/MPPI action constraints."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import platform
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--max-norm", type=float, default=2.45)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    import numpy as np
    import torch

    from eb_jepa.datasets.two_rooms.env import DotWall
    from eb_jepa.planning import CEMPlanner, MPPIPlanner

    source_root = Path(args.source_root).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != args.revision:
        raise RuntimeError(f"source revision mismatch: {revision} != {args.revision}")

    def unroll(_observation, actions):
        return torch.cumsum(actions, dim=2).unsqueeze(-1).unsqueeze(-1)

    def objective(encodings):
        return -encodings[:, 0, -1, 0, 0]

    observation = torch.zeros(1, 1, 1, 1, 1, device="cuda")
    rows = []
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        mppi = MPPIPlanner(
            unroll=unroll,
            n_iters=3,
            num_samples=128,
            plan_length=3,
            action_dim=2,
            num_elites=16,
            max_std=2.0,
            max_norms=[args.max_norm],
            max_norm_dims=[0, 1],
        )
        mppi.set_objective(objective)
        mppi_actions = mppi.plan(observation, eval_mode=True).actions
        torch.manual_seed(seed)
        np.random.seed(seed)
        cem = CEMPlanner(
            unroll=unroll,
            n_iters=3,
            num_samples=128,
            plan_length=3,
            action_dim=2,
            num_elites=16,
            var_scale=2.0,
            max_norms=[args.max_norm],
            max_norm_dims=[0, 1],
            decode_each_iteration=False,
        )
        cem.set_objective(objective)
        cem_actions = cem.plan(observation, eval_mode=True).actions
        rows.append(
            {
                "seed": seed,
                "mppi_max_norm": float(mppi_actions.norm(dim=-1).max().item()),
                "cem_max_norm": float(cem_actions.norm(dim=-1).max().item()),
                "mppi_max_abs_component": float(mppi_actions.abs().max().item()),
                "cem_max_abs_component": float(cem_actions.abs().max().item()),
            }
        )
    mppi_norms = np.array([row["mppi_max_norm"] for row in rows])
    cem_norms = np.array([row["cem_max_norm"] for row in rows])
    tolerance = 1e-6
    mppi_source = inspect.getsource(MPPIPlanner.plan)
    cem_source = inspect.getsource(CEMPlanner.plan)
    environment_step_source = inspect.getsource(DotWall.step)
    payload = {
        "source_revision": revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "design": {
            "num_seeds": args.num_seeds,
            "seeds": list(range(args.num_seeds)),
            "max_norm": args.max_norm,
            "n_iters": 3,
            "num_samples": 128,
            "num_elites": 16,
            "plan_length": 3,
            "objective": "maximize terminal cumulative x under identical deterministic unroll",
        },
        "source_contract": {
            "cem_plan_mentions_max_norms": "max_norms" in cem_source,
            "mppi_plan_mentions_max_norms": "max_norms" in mppi_source,
            "environment_step_checks_action_space": "action_space" in environment_step_source,
            "environment_step_passes_action_to_transition": (
                "_calculate_next_position(action)" in environment_step_source
            ),
        },
        "summary": {
            "mppi_violation_count": int((mppi_norms > args.max_norm + tolerance).sum()),
            "cem_violation_count": int((cem_norms > args.max_norm + tolerance).sum()),
            "mppi_violation_fraction": float(
                (mppi_norms > args.max_norm + tolerance).mean()
            ),
            "cem_violation_fraction": float((cem_norms > args.max_norm + tolerance).mean()),
            "mppi_median_max_norm": float(np.median(mppi_norms)),
            "cem_median_max_norm": float(np.median(cem_norms)),
            "mppi_max_observed_norm": float(mppi_norms.max()),
            "cem_max_observed_norm": float(cem_norms.max()),
        },
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
