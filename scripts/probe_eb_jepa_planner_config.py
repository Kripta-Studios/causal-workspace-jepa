#!/usr/bin/env python
"""Confirm the pinned EB-JEPA planner proposal-scale configuration contract."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    import torch
    import yaml

    from eb_jepa.planning import CEMPlanner, MPPIPlanner

    source_root = Path(args.source_root).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = not bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if revision != args.revision or not clean:
        raise RuntimeError("source checkout does not match the clean pinned revision")

    config_root = source_root / "examples" / "ac_video_jepa" / "cfgs"
    with (config_root / "planning_mppi.yaml").open(encoding="utf-8") as handle:
        mppi_yaml = yaml.safe_load(handle)["planner"]
    with (config_root / "planning_cem.yaml").open(encoding="utf-8") as handle:
        cem_yaml = yaml.safe_load(handle)["planner"]

    def unroll(_observation, actions):
        return torch.cumsum(actions, dim=2).unsqueeze(-1).unsqueeze(-1)

    mppi = MPPIPlanner(unroll=unroll, action_dim=2, **mppi_yaml)
    cem = CEMPlanner(unroll=unroll, action_dim=2, **cem_yaml)
    corrected_keyword = dict(mppi_yaml)
    corrected_keyword["max_std"] = corrected_keyword.pop("var_scale")
    mppi_keyword_corrected = MPPIPlanner(
        unroll=unroll,
        action_dim=2,
        **corrected_keyword,
    )
    payload = {
        "source": {"revision": revision, "clean": clean},
        "yaml": {
            "mppi_var_scale": mppi_yaml.get("var_scale"),
            "mppi_has_max_std": "max_std" in mppi_yaml,
            "cem_var_scale": cem_yaml.get("var_scale"),
        },
        "signatures": {
            "mppi_parameters": list(inspect.signature(MPPIPlanner.__init__).parameters),
            "cem_parameters": list(inspect.signature(CEMPlanner.__init__).parameters),
            "mppi_accepts_kwargs": any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in inspect.signature(MPPIPlanner.__init__).parameters.values()
            ),
        },
        "instances": {
            "official_mppi_max_std": float(mppi.max_std),
            "official_cem_var_scale": float(cem.var_scale),
            "keyword_corrected_mppi_max_std": float(mppi_keyword_corrected.max_std),
        },
        "interpretation": {
            "var_scale_is_silently_ignored_by_mppi": float(mppi.max_std)
            != float(mppi_yaml["var_scale"]),
            "actual_proposal_scales_differ": float(mppi.max_std) != float(cem.var_scale),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
