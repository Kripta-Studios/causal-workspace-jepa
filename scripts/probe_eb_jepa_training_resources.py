#!/usr/bin/env python
"""Profile one official EB-JEPA Two Rooms training batch in an isolated process."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback


def _git(source_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=source_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _error_category(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}".lower()
    if "out of memory" in message:
        return "CUDA_OOM"
    if "compile" in message or "inductor" in message or "triton" in message:
        return "COMPILE_FAILURE"
    return "RUNTIME_FAILURE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--mode", choices=("eager", "compile"), required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=607)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    import numpy as np
    import torch

    source_root = Path(args.source_root).resolve()
    resolved_revision = _git(source_root, "rev-parse", "HEAD")
    source_clean = not bool(_git(source_root, "status", "--porcelain"))
    if resolved_revision != args.revision or not source_clean:
        raise RuntimeError("source checkout does not match the clean pinned revision")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    started = time.perf_counter()
    payload: dict[str, object] = {
        "source": {"revision": resolved_revision, "clean": source_clean},
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "arch_list": list(torch.cuda.get_arch_list()),
        },
        "profile": {
            "batch_size": args.batch_size,
            "mode": args.mode,
            "steps_requested": args.steps,
            "sample_length": 17,
            "dataset_n_steps": 91,
            "prediction_steps": 8,
            "dtype": "bfloat16",
        },
    }

    try:
        from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
        from eb_jepa.datasets.utils import init_data
        from eb_jepa.jepa import JEPA, JEPAProbe
        from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
        from eb_jepa.state_decoder import MLPXYHead

        data_started = time.perf_counter()
        loader, _, data_config = init_data(
            "two_rooms",
            {
                "size": args.batch_size,
                "val_size": 4,
                "batch_size": args.batch_size,
                "num_workers": 0,
                "pin_mem": False,
                "persistent_workers": False,
                "sample_length": 17,
                "n_steps": 91,
            },
        )
        batch = next(iter(loader))
        payload["dataset"] = {
            "batch_shapes": [list(value.shape) for value in batch],
            "image_size": data_config.img_size,
            "generation_seconds": time.perf_counter() - data_started,
        }
        observations = batch[0].to(device)
        actions = batch[1].to(device)
        locations = batch[2].to(device)

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
        predictor = RNNPredictor(
            hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln
        )
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
        jepa = JEPA(
            encoder,
            torch.nn.Identity(),
            predictor,
            regularizer,
            SquareLossSeq(),
        ).to(device)
        xy_head = MLPXYHead(
            input_shape=encoder.mlp_output_dim, normalizer=loader.dataset.normalizer
        ).to(device)
        xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=torch.nn.MSELoss())
        jepa_optimizer = torch.optim.AdamW(jepa.parameters(), lr=1e-3, weight_decay=1e-5)
        probe_optimizer = torch.optim.AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        dynamo_counters = None
        if args.mode == "compile":
            from torch._dynamo.utils import counters

            counters.clear()
            dynamo_counters = counters
            model_for_step = torch.compile(jepa)
        else:
            model_for_step = jepa

        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        step_rows = []
        first_parameter = next(jepa.parameters())
        parameter_before = first_parameter.detach().clone()
        for step in range(args.steps):
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            jepa_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction, losses = model_for_step.unroll(
                    observations,
                    actions,
                    nsteps=8,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
                jepa_loss = losses[0]
            scaler.scale(jepa_loss).backward()
            scaler.unscale_(jepa_optimizer)
            torch.nn.utils.clip_grad_norm_(jepa.encoder.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(jepa.predictor.parameters(), 2.0)
            scaler.step(jepa_optimizer)
            scaler.update()

            probe_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                xy_loss = xy_prober(
                    observations=observations[:, :, :1],
                    targets=locations[:, :, :1],
                )
                xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
            scaler.scale(xy_loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - step_started
            step_rows.append(
                {
                    "step": step,
                    "seconds": seconds,
                    "trajectories_per_second": args.batch_size / seconds,
                    "jepa_loss": float(jepa_loss.detach().item()),
                    "probe_loss": float(xy_loss.detach().item()),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )

        parameter_delta = float((first_parameter.detach() - parameter_before).abs().max().item())
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in jepa.parameters()
        )
        payload["training"] = {
            "status": "SUCCESS",
            "steps_completed": len(step_rows),
            "step_rows": step_rows,
            "losses_finite": all(
                np.isfinite(row["jepa_loss"]) and np.isfinite(row["probe_loss"])
                for row in step_rows
            ),
            "gradients_finite": gradients_finite,
            "parameter_delta": parameter_delta,
            "prediction_shape": list(prediction.shape),
            "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "predictor_parameters": sum(parameter.numel() for parameter in predictor.parameters()),
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in step_rows),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in step_rows),
        }
        if dynamo_counters is not None:
            frames_total = int(dynamo_counters["frames"]["total"])
            frames_ok = int(dynamo_counters["frames"]["ok"])
            unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
            payload["compile"] = {
                "wrapper_type": type(model_for_step).__name__,
                "frames_total": frames_total,
                "frames_ok": frames_ok,
                "unique_graphs": unique_graphs,
                "effective_graph_capture": frames_total > 0 and unique_graphs > 0,
                "called_entrypoint": "unroll",
            }
    except BaseException as error:  # isolated diagnostic must preserve failures as data
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        payload["training"] = {
            "status": "FAILED",
            "error_category": _error_category(error),
            "error_type": type(error).__name__,
            "error_message": str(error)[:4000],
            "traceback_tail": traceback.format_exc()[-8000:],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    payload["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
