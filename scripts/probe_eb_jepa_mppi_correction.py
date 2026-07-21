#!/usr/bin/env python
"""Compare official and constraint-corrected EB-JEPA MPPI implementations."""

from __future__ import annotations

import argparse
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

    from causal_workspace_jepa.planning.eb_jepa_constrained_mppi import (
        ConstrainedMPPIPlanner,
    )
    from eb_jepa.planning import MPPIPlanner

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

    def objective(encodings):
        return -encodings[:, 0, -1, 0, 0]

    observation = torch.zeros(1, 1, 1, 1, 1, device="cuda")
    rows = []
    for seed in range(args.num_seeds):
        observed_candidate_norms: list[float] = []

        def unroll(_observation, actions):
            observed_candidate_norms.append(float(actions.permute(2, 0, 1).norm(dim=-1).max()))
            return torch.cumsum(actions, dim=2).unsqueeze(-1).unsqueeze(-1)

        common = {
            "unroll": unroll,
            "n_iters": 3,
            "num_samples": 128,
            "plan_length": 3,
            "action_dim": 2,
            "num_elites": 16,
            "max_std": 2.0,
        }
        torch.manual_seed(seed)
        np.random.seed(seed)
        official_unbounded = MPPIPlanner(**common)
        official_unbounded.set_objective(objective)
        official_unbounded_result = official_unbounded.plan(observation, eval_mode=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        corrected_unbounded = ConstrainedMPPIPlanner(**common)
        corrected_unbounded.set_objective(objective)
        corrected_unbounded_result = corrected_unbounded.plan(observation, eval_mode=True)

        observed_candidate_norms.clear()
        torch.manual_seed(seed)
        np.random.seed(seed)
        corrected_bounded = ConstrainedMPPIPlanner(
            **common,
            max_norms=[args.max_norm],
            max_norm_dims=[0, 1],
        )
        corrected_bounded.set_objective(objective)
        corrected_bounded_result = corrected_bounded.plan(observation, eval_mode=True)
        rows.append(
            {
                "seed": seed,
                "unbounded_action_max_abs_diff": float(
                    (
                        official_unbounded_result.actions
                        - corrected_unbounded_result.actions
                    )
                    .abs()
                    .max()
                ),
                "unbounded_loss_max_abs_diff": float(
                    (official_unbounded_result.losses - corrected_unbounded_result.losses)
                    .abs()
                    .max()
                ),
                "corrected_returned_max_norm": float(
                    corrected_bounded_result.actions.norm(dim=-1).max()
                ),
                "corrected_cost_input_max_norm": max(observed_candidate_norms),
            }
        )
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
        },
        "summary": {
            "max_unbounded_action_abs_diff": max(
                row["unbounded_action_max_abs_diff"] for row in rows
            ),
            "max_unbounded_loss_abs_diff": max(
                row["unbounded_loss_max_abs_diff"] for row in rows
            ),
            "corrected_returned_violation_count": sum(
                row["corrected_returned_max_norm"] > args.max_norm + 1e-6 for row in rows
            ),
            "corrected_cost_input_violation_count": sum(
                row["corrected_cost_input_max_norm"] > args.max_norm + 1e-6 for row in rows
            ),
            "corrected_max_returned_norm": max(
                row["corrected_returned_max_norm"] for row in rows
            ),
            "corrected_max_cost_input_norm": max(
                row["corrected_cost_input_max_norm"] for row in rows
            ),
        },
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
