#!/usr/bin/env python
"""Standalone official Two Rooms dataset/train/checkpoint/planner smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time


def _git(source_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    import numpy as np
    from omegaconf import OmegaConf
    import torch

    from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
    from eb_jepa.datasets.two_rooms.env import DotWall
    from eb_jepa.datasets.utils import init_data
    from eb_jepa.jepa import JEPA
    from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
    from eb_jepa.planning import GCAgent

    source_root = Path(args.source_root).resolve()
    resolved_revision = _git(source_root, "rev-parse", "HEAD")
    source_clean = not bool(_git(source_root, "status", "--porcelain"))
    if resolved_revision != args.revision or not source_clean:
        raise RuntimeError("source checkout does not match the clean pinned revision")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    data_started = time.perf_counter()
    loader, validation_loader, data_config = init_data(
        "two_rooms",
        {
            "size": 8,
            "val_size": 4,
            "batch_size": 2,
            "num_workers": 0,
            "pin_mem": False,
            "persistent_workers": False,
            "sample_length": 9,
            "n_steps": 16,
        },
    )
    batch = next(iter(loader))
    data_seconds = time.perf_counter() - data_started
    batch_digest = hashlib.sha256()
    for value in batch:
        cpu_value = value.detach().cpu().contiguous()
        batch_digest.update(str(cpu_value.dtype).encode("ascii"))
        batch_digest.update(str(tuple(cpu_value.shape)).encode("ascii"))
        batch_digest.update(cpu_value.numpy().tobytes())
    observations, actions = batch[0].to(device), batch[1].to(device)

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
    model = JEPA(
        encoder,
        torch.nn.Identity(),
        predictor,
        regularizer,
        SquareLossSeq(),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    first_parameter = next(model.parameters())
    parameter_before = first_parameter.detach().clone()
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prediction, losses = model.unroll(
            observations,
            actions,
            nsteps=4,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=True,
            return_all_steps=False,
        )
        loss = losses[0]
    scaler.scale(loss).backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    scaler.step(optimizer)
    scaler.update()
    parameter_delta = float((first_parameter.detach() - parameter_before).abs().max().item())
    state_digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu_value = value.detach().cpu().contiguous()
        state_digest.update(name.encode("utf-8"))
        state_digest.update(str(cpu_value.dtype).encode("ascii"))
        state_digest.update(cpu_value.numpy().tobytes())

    with tempfile.TemporaryDirectory(prefix="eb-jepa-smoke-") as temporary_directory:
        checkpoint = Path(temporary_directory) / "state.pt"
        torch.save(model.state_dict(), checkpoint)
        restored_value = first_parameter.detach().clone()
        with torch.no_grad():
            first_parameter.zero_()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        checkpoint_error = float((first_parameter.detach() - restored_value).abs().max().item())
        checkpoint_bytes = checkpoint.stat().st_size

    plan_config = OmegaConf.create(
        {
            "ctxt_window_time": 1,
            "planner": {
                "planner_name": "mppi",
                "num_act_stepped": 1,
                "plan_length": 3,
                "n_iters": 2,
                "num_samples": 8,
                "num_elites": 2,
                "max_std": 1.5,
                "max_norms": [2.45],
                "max_norm_dims": [0, 1],
                "decode_each_iteration": False,
                "planning_objective": {
                    "objective_type": "repr_dist",
                    "sum_all_diffs": True,
                },
            },
        }
    )
    environment = DotWall(
        config=data_config,
        rng=np.random.default_rng(args.seed + 1),
        level="normal",
        n_steps=16,
        n_allowed_steps=4,
    )
    observation, info = environment.reset()
    model.eval()
    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_config,
        normalizer=environment.normalizer,
        env=environment,
    )
    agent.set_goal(info["target_obs"].float(), info["target_position"])
    normalized_observation = (
        environment.normalizer.normalize_state(observation.float().to(device))
        .unsqueeze(0)
        .unsqueeze(2)
    )
    planned_action = agent.act(normalized_observation, steps_left=3, t0=True)
    torch.cuda.synchronize(device)
    planned_norms = planned_action.norm(dim=-1)
    package_names = (
        "numpy",
        "scipy",
        "pandas",
        "PyYAML",
        "omegaconf",
        "einops",
        "gymnasium",
        "scikit-learn",
        "wandb",
    )
    fingerprint_fields = {
        "seed": args.seed,
        "batch_sha256": batch_digest.hexdigest(),
        "model_state_sha256": state_digest.hexdigest(),
        "loss": float(loss.detach().item()),
        "parameter_delta": parameter_delta,
        "planned_action": planned_action.detach().cpu().tolist(),
        "planner_losses": agent._prev_losses.tolist(),
    }
    fingerprint_sha256 = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "source": {
            "revision": resolved_revision,
            "clean": source_clean,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "arch_list": list(torch.cuda.get_arch_list()),
            "packages": {
                name: importlib.metadata.version(name) for name in package_names
            },
        },
        "dataset": {
            "batch_shapes": [list(value.shape) for value in batch],
            "train_batches": len(loader),
            "validation_batches": len(validation_loader),
            "image_size": data_config.img_size,
            "generation_seconds": data_seconds,
        },
        "training": {
            "loss": float(loss.detach().item()),
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
            "gradients_finite": gradients_finite,
            "parameter_delta": parameter_delta,
            "prediction_shape": list(prediction.shape),
            "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "predictor_parameters": sum(
                parameter.numel() for parameter in predictor.parameters()
            ),
        },
        "checkpoint": {
            "bytes": checkpoint_bytes,
            "max_restore_error": checkpoint_error,
        },
        "planner": {
            "name": type(agent.planner).__name__,
            "action_shape": list(planned_action.shape),
            "action": planned_action.detach().cpu().tolist(),
            "max_action_norm": float(planned_norms.max().item()),
            "configured_max_norm": 2.45,
            "within_configured_max_norm": bool((planned_norms <= 2.450001).all().item()),
            "losses": agent._prev_losses.tolist(),
            "finite": bool(torch.isfinite(planned_action).all().item())
            and bool(torch.isfinite(agent._prev_losses).all().item()),
        },
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "determinism": {
            "python_random_seed": args.seed,
            "numpy_seed": args.seed,
            "torch_seed": args.seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "fingerprint_fields": fingerprint_fields,
            "fingerprint_sha256": fingerprint_sha256,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
