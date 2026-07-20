"""Tier 0 dataset generation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_workspace_jepa.common.config import get_nested, load_config
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.synthetic.base import save_dataset
from causal_workspace_jepa.data.synthetic.bouncing_ball import generate_bouncing_ball2d
from causal_workspace_jepa.data.synthetic.minipush import generate_minipush
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.data.synthetic.tiny_maze import generate_tiny_maze
from causal_workspace_jepa.data.synthetic.two_body import generate_two_body_collision


def generate_tier0(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    require_free_disk(resource_profile)
    seed = int(config.get("seed", 0))
    trajectories = int(config.get("trajectories_per_env", 16))
    steps = int(config.get("steps_per_trajectory", 32))
    output_dir = Path(str(config.get("output_dir", "data/processed/tier0_smoke")))
    env_config = config.get("environments", {})
    records: list[dict[str, Any]] = []

    if get_nested(env_config, "pointmass2d", False):
        records.append(save_dataset(generate_pointmass2d(trajectories=trajectories, steps=steps, seed=seed), output_dir))
    if get_nested(env_config, "bouncing_ball2d", False):
        records.append(
            save_dataset(
                generate_bouncing_ball2d(trajectories=trajectories, steps=steps, seed=seed + 1),
                output_dir,
            )
        )
    if get_nested(env_config, "two_body_collision", False):
        records.append(
            save_dataset(
                generate_two_body_collision(trajectories=trajectories, steps=steps, seed=seed + 2),
                output_dir,
            )
        )
    if get_nested(env_config, "tiny_maze", False):
        records.append(save_dataset(generate_tiny_maze(trajectories=trajectories, steps=steps, seed=seed + 3), output_dir))
    if get_nested(env_config, "minipush", False):
        records.append(save_dataset(generate_minipush(trajectories=trajectories, steps=steps, seed=seed + 4), output_dir))

    total_bytes = sum(int(record["bytes"]) for record in records)
    max_total_mb = float(config.get("max_total_mb", 100))
    if total_bytes > max_total_mb * 1024 * 1024:
        raise RuntimeError(f"Tier 0 output {total_bytes} bytes exceeds cap {max_total_mb} MB")
    manifest = {
        "config": str(config_path),
        "resource_profile": resource_profile,
        "seed": seed,
        "output_dir": str(output_dir),
        "total_bytes": total_bytes,
        "datasets": records,
    }
    manifest_path = Path("data/manifests/tier0_smoke_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
