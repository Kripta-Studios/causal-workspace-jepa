#!/usr/bin/env python
"""Evaluate one frozen EB-JEPA checkpoint/planner arm without rendering artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_workspace_jepa.experiments.world_model.eb_jepa_competence_protocol import (  # noqa: E402
    planner_arm_contract,
)
from causal_workspace_jepa.planning.eb_jepa_constrained_mppi import (  # noqa: E402
    ConstrainedMPPIPlanner,
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


def _build_model(data_config, device):
    import torch

    from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, data_config.img_size, data_config.img_size),
    )
    predictor = RNNPredictor(hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln)
    idm = InverseDynamicsModel(state_dim=512, hidden_dim=256, action_dim=2).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=8,
        std_coeff=16,
        sim_coeff_t=12,
        idm_coeff=1,
        idm=idm,
        first_t_only=False,
        projector=None,
        spatial_as_samples=False,
        idm_after_proj=False,
        sim_t_after_proj=False,
    )
    return JEPA(
        encoder,
        torch.nn.Identity(),
        predictor,
        regularizer,
        SquareLossSeq(),
    ).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    import numpy as np
    from omegaconf import OmegaConf
    import torch
    import yaml

    from eb_jepa.datasets.two_rooms.env import DotWall
    from eb_jepa.datasets.utils import init_data
    from eb_jepa.planning import GCAgent
    from eb_jepa.training_utils import load_checkpoint

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.training_seed not in config["training_seeds"]:
        raise ValueError("training seed is not preregistered")
    if args.epoch not in config["checkpoint_epochs"]:
        raise ValueError("checkpoint epoch is not preregistered")
    if args.arm not in config["planner_arms"]:
        raise ValueError("planner arm is not preregistered")
    source_root = (REPO_ROOT / config["source_root"]).resolve()
    revision = _git(source_root, "rev-parse", "HEAD")
    source_clean = not bool(_git(source_root, "status", "--porcelain"))
    if revision != config["revision"] or not source_clean:
        raise RuntimeError("official source does not match the clean pinned revision")
    repo_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    repo_dirty = bool(_git(REPO_ROOT, "status", "--porcelain"))
    if repo_dirty:
        raise RuntimeError("repository must be clean before competence evaluation")

    checkpoint = (
        REPO_ROOT
        / config["checkpoint_root"]
        / f"seed-{args.training_seed}"
        / f"e-{args.epoch}.pth.tar"
    ).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output = (
        REPO_ROOT
        / config["job_output_root"]
        / f"seed-{args.training_seed}"
        / f"epoch-{args.epoch}"
        / f"{args.arm}.json"
    ).resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing.get(
            "checkpoint_sha256"
        ) == _sha256(checkpoint):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    analysis_seed = int(config["analysis_seed_base"]) + args.training_seed + args.epoch
    random.seed(analysis_seed)
    np.random.seed(analysis_seed)
    torch.manual_seed(analysis_seed)
    torch.cuda.manual_seed_all(analysis_seed)
    device = torch.device("cuda")
    _, _, data_config = init_data(
        "two_rooms",
        {
            "size": 4,
            "val_size": 4,
            "batch_size": 4,
            "num_workers": 0,
            "pin_mem": False,
            "persistent_workers": False,
        },
    )
    model = _build_model(data_config, device)
    checkpoint_info = load_checkpoint(checkpoint, model, device=device)
    model.eval()
    plan_path = source_root / config["upstream_planner_config"]
    plan_config = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    frozen = config["planner_parameters"]
    for key, value in frozen.items():
        if plan_config["planner"].get(key) != value:
            raise RuntimeError(f"upstream planner config drifted at {key}")
    plan_config["logging"]["tqdm_silent"] = True
    plan_cfg = OmegaConf.create(plan_config)
    environment = DotWall(
        config=data_config,
        rng=np.random.default_rng(int(config["environment_seed"])),
        level="normal",
        n_allowed_steps=int(config["max_environment_steps"]),
    )
    environment.reset()  # preserve the official pre-loop discarded reset
    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_cfg,
        normalizer=environment.normalizer,
        env=environment,
    )
    arm_contract = planner_arm_contract(args.arm)
    if arm_contract["planner_implementation"] == "constraint_corrected":
        planner_kwargs = dict(plan_config["planner"])
        planner_kwargs["max_std"] = arm_contract["max_std"]
        agent.planner = ConstrainedMPPIPlanner(
            unroll=agent.unroll,
            action_dim=2,
            decode_loc_to_pixel=agent.decode_loc_to_pixel,
            **planner_kwargs,
        )
    elif float(agent.planner.max_std) != float(arm_contract["max_std"]):
        raise RuntimeError("official MPPI instantiated with an unexpected proposal scale")

    episodes = []
    started = time.perf_counter()
    for episode in range(int(config["num_episodes"])):
        episode_started = time.perf_counter()
        observation, info = environment.reset()
        observation, _, _, _, info = environment.step(
            np.zeros(environment.action_space.shape[0])
        )
        agent.set_goal(info["target_obs"].float(), info["target_position"])
        steps_left = environment.n_allowed_steps
        first_step = True
        action_norms = []
        while steps_left > 0:
            observation_tensor = (
                environment.normalizer.normalize_state(
                    observation.detach().clone().float().to(device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )
            action = agent.act(
                observation_tensor,
                steps_left=steps_left,
                t0=first_step,
            ).cpu().numpy()
            for single_action in action:
                action_norms.append(float(np.linalg.norm(single_action)))
                observation, _, _, _, info = environment.step(single_action)
                first_step = False
                steps_left -= 1
        final = environment.eval_state(info["target_position"], info["dot_position"])
        episodes.append(
            {
                "arm": args.arm,
                "training_seed": args.training_seed,
                "checkpoint_epoch": args.epoch,
                "episode": episode,
                "success": bool(final["success"]),
                "final_state_distance": float(final["state_dist"]),
                "episode_seconds": time.perf_counter() - episode_started,
                "executed_action_count": len(action_norms),
                "executed_action_violation_count": sum(
                    norm > float(config["action_max_norm"]) + 1e-6 for norm in action_norms
                ),
                "max_executed_action_norm": max(action_norms),
            }
        )
    payload = {
        "status": "COMPLETED",
        "experiment_id": config["id"],
        "repo_commit": repo_commit,
        "repo_dirty_at_start": repo_dirty,
        "source_revision": revision,
        "source_clean": source_clean,
        "training_seed": args.training_seed,
        "checkpoint_epoch": args.epoch,
        "checkpoint_recorded_epoch": int(checkpoint_info["epoch"]) - 1,
        "checkpoint_sha256": _sha256(checkpoint),
        "arm": args.arm,
        "arm_contract": arm_contract,
        "analysis_seed": analysis_seed,
        "environment_seed": int(config["environment_seed"]),
        "episodes": episodes,
        "summary": {
            "success_rate": sum(row["success"] for row in episodes) / len(episodes),
            "mean_final_state_distance": sum(
                row["final_state_distance"] for row in episodes
            )
            / len(episodes),
            "executed_action_violation_count": sum(
                row["executed_action_violation_count"] for row in episodes
            ),
            "max_executed_action_norm": max(
                row["max_executed_action_norm"] for row in episodes
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
