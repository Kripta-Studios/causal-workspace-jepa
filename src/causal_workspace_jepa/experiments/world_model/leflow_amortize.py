"""CPU amortized planning control. Not a LeFlow paper reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from causal_workspace_jepa.common.provenance import collect_provenance, is_git_dirty, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.splits import deterministic_named_split_ids
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.amortized_latent import (
    amortized_latent_plan,
    fit_action_flow,
    fit_inverse_dynamics,
    action_flow_plan,
    decode_actions,
)
from causal_workspace_jepa.planning.cem import iterative_cem_plan, random_shooting_plan
from causal_workspace_jepa.planning.closed_loop import pointmass_position_mse, pointmass_rollout_state

ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_SPLIT_NAMES = ("test", "paraphrase")
MKNN_PASS = "TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED"
MKNN_FAIL = "NEGATIVE_RESULT"


def load_json_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_array(array: np.ndarray) -> str:
    packed = np.ascontiguousarray(array)
    return hashlib.sha256(
        packed.tobytes() + str(tuple(packed.shape)).encode("utf-8") + str(packed.dtype).encode("utf-8")
    ).hexdigest()


def model_fingerprint(model: TinyActionConditionedJEPA) -> str:
    return hashlib.sha256(
        "".join(
            sha256_array(item)
            for item in (model.encoder, model.predictor, model.decoder, model.latent_mean)
        ).encode("utf-8")
    ).hexdigest()


def reject_forbidden_seed(seed: int, forbidden: list[int]) -> None:
    if int(seed) in {int(item) for item in forbidden}:
        raise ValueError(f"seed {seed} is frozen for another experiment")


def assert_protocol(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "WM-LEFLOW-AMORTIZE-001":
        raise ValueError("wrong experiment_id")
    if not config.get("not_a_leflow_reproduction", False):
        raise ValueError("candidate must not be labeled a LeFlow reproduction")
    if int(config["splits"]["train"]) + int(config["splits"]["development"]) + int(
        config["splits"]["confirmation"]
    ) != int(config["environment"]["trajectories"]):
        raise ValueError("split sizes must cover the frozen trajectory population")
    if int(config["splits"]["n_tasks"]) != int(config["splits"]["confirmation"]):
        raise ValueError("n_tasks must equal the confirmation population")


def require_mknn_adjudication(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "WM-PLATONIC-MKNN-001":
        raise RuntimeError("T2 requires WM-PLATONIC-MKNN-001 adjudication")
    if payload.get("integrity_blockers"):
        raise RuntimeError("T2 blocked by MKNN integrity_blockers")
    if payload.get("status") not in {MKNN_PASS, MKNN_FAIL}:
        raise RuntimeError("T2 requires a completed MKNN status")
    return payload


def _latents(model: TinyActionConditionedJEPA, observations: np.ndarray) -> np.ndarray:
    packed = observations.reshape(-1, observations.shape[-1])
    encoded = model.encode(packed).tensor
    return encoded.reshape(*observations.shape[:-1], -1)


def _tasks(states: np.ndarray, horizon: int) -> list[dict[str, np.ndarray]]:
    tasks = []
    for index in range(states.shape[0]):
        start = states[index, 0]
        goal_state = states[index, horizon]
        tasks.append(
            {
                "task_id": np.array([index], dtype=np.int32),
                "start": start,
                "goal_position": goal_state[:2].copy(),
                "goal_observation": goal_state.copy(),
            }
        )
    return tasks


def _evaluate_plan(
    start: np.ndarray,
    goal_position: np.ndarray,
    actions: np.ndarray,
    success_max: float,
) -> dict[str, float | bool]:
    final_state = pointmass_rollout_state(start, actions)
    mse = pointmass_position_mse(final_state, goal_position)
    return {"position_mse": mse, "success": bool(mse < success_max)}


def _time_plan(fn: Callable[[], Mapping[str, Any]]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    plan = dict(fn())
    elapsed = time.perf_counter() - started
    return plan, elapsed


def run_seed(config: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    assert_protocol(config)
    reject_forbidden_seed(seed, list(config["forbidden_seeds"]))
    env = config["environment"]
    model_cfg = config["model"]
    splits_cfg = config["splits"]
    plan_cfg = config["planning"]
    id_cfg = config["inverse_dynamics"]
    dataset = generate_pointmass2d(
        trajectories=int(env["trajectories"]),
        steps=int(env["steps"]),
        seed=int(env["environment_base_seed"]) + int(seed),
    )
    split_ids = deterministic_named_split_ids(
        dataset.states.shape[0],
        int(splits_cfg["split_seed"]),
        train=int(splits_cfg["train"]),
        development=int(splits_cfg["development"]),
        confirmation=int(splits_cfg["confirmation"]),
    )
    if any(name in split_ids for name in FORBIDDEN_SPLIT_NAMES):
        raise RuntimeError("protected split name leaked into amortized control")
    train_obs = dataset.observations[split_ids["train"]]
    train_actions = dataset.actions[split_ids["train"]]
    confirm_obs = dataset.observations[split_ids["confirmation"]]
    confirm_actions = dataset.actions[split_ids["confirmation"]]
    confirm_states = dataset.states[split_ids["confirmation"]]
    model = TinyActionConditionedJEPA.fit(
        train_obs,
        train_actions,
        latent_dim=int(model_cfg["latent_dim"]),
        seed=int(seed),
        ridge=float(model_cfg["ridge"]),
        action_mode="conditioned",
    )
    fingerprint_before = model_fingerprint(model)
    train_latents = _latents(model, train_obs)
    confirm_latents = _latents(model, confirm_obs)
    weights_a = fit_inverse_dynamics(
        train_latents,
        train_actions,
        include_delta=False,
        ridge=float(id_cfg["ridge"]),
    )
    weights_b = fit_inverse_dynamics(
        train_latents,
        train_actions,
        include_delta=True,
        ridge=float(id_cfg["ridge"]),
    )
    if weights_a.shape != weights_b.shape:
        raise RuntimeError("inverse-dynamics arms are not capacity-matched")
    z_t = confirm_latents[:, :-1, :].reshape(-1, confirm_latents.shape[-1])
    z_next = confirm_latents[:, 1:, :].reshape(-1, confirm_latents.shape[-1])
    true_a = confirm_actions.reshape(-1, confirm_actions.shape[-1])
    pred_a = decode_actions(weights_a, z_t, z_next, include_delta=False)
    pred_b = decode_actions(weights_b, z_t, z_next, include_delta=True)
    id_mse_a = float(np.mean((pred_a - true_a) ** 2))
    id_mse_b = float(np.mean((pred_b - true_a) ** 2))
    success_max = float(plan_cfg["success_position_mse_max"])
    horizon_reports: dict[str, Any] = {}
    for horizon in [int(plan_cfg["primary_horizon"]), *[int(item) for item in plan_cfg["diagnostic_horizons"]]]:
        action_weights = fit_action_flow(
            train_latents,
            train_actions,
            horizon=horizon,
            ridge=float(id_cfg["ridge"]),
        )
        tasks = _tasks(confirm_states, horizon)
        if [int(task["task_id"][0]) for task in tasks] != list(range(len(tasks))):
            raise RuntimeError("start/goal population identity drifted")
        arm_stats: dict[str, dict[str, float]] = {}
        planners: dict[str, Callable[[dict[str, np.ndarray]], dict[str, Any]]] = {
            "random_shooting_n64": lambda task, h=horizon: random_shooting_plan(
                model,
                task["start"],
                task["goal_position"],
                horizon=h,
                candidates=int(plan_cfg["shooting_candidates"]),
                seed=int(seed) + 1000 * h + int(task["task_id"][0]),
            ),
            "iterative_cem": lambda task, h=horizon: iterative_cem_plan(
                model,
                task["start"],
                task["goal_position"],
                horizon=h,
                candidates=int(plan_cfg["cem_candidates"]),
                iterations=int(plan_cfg["cem_iterations"]),
                elite_fraction=0.25,
                seed=int(seed) + 2000 * h + int(task["task_id"][0]),
            ),
            "latent_flow_n1": lambda task, h=horizon: amortized_latent_plan(
                model,
                weights_b,
                task["start"],
                task["goal_observation"],
                horizon=h,
                candidates=1,
                seed=int(seed) + 3000 * h + int(task["task_id"][0]),
                noise_std=float(plan_cfg["n1_noise_std"]),
                include_delta=True,
            ),
            "latent_flow_n64": lambda task, h=horizon: amortized_latent_plan(
                model,
                weights_b,
                task["start"],
                task["goal_observation"],
                horizon=h,
                candidates=int(plan_cfg["amortized_rerank_n"][1]),
                seed=int(seed) + 4000 * h + int(task["task_id"][0]),
                noise_std=float(plan_cfg["n64_noise_std"]),
                include_delta=True,
            ),
            "action_flow_n1": lambda task, h=horizon, w=action_weights: action_flow_plan(
                model,
                w,
                task["start"],
                task["goal_observation"],
                horizon=h,
            ),
        }
        for arm_name, planner in planners.items():
            successes = []
            clocks = []
            evaluated = []
            for task in tasks:
                plan, elapsed = _time_plan(lambda p=planner, t=task: p(t))
                result = _evaluate_plan(task["start"], task["goal_position"], plan["actions"], success_max)
                successes.append(float(result["success"]))
                clocks.append(elapsed)
                evaluated.append(int(plan["candidates_evaluated"]))
            arm_stats[arm_name] = {
                "success_rate": float(np.mean(successes)),
                "mean_wall_clock_s": float(np.mean(clocks)),
                "mean_candidates_evaluated": float(np.mean(evaluated)),
                "n_tasks": float(len(tasks)),
                "horizon": float(horizon),
            }
        fingerprint_after = model_fingerprint(model)
        if fingerprint_after != fingerprint_before:
            raise RuntimeError("world model changed during planner comparison")
        horizon_reports[str(horizon)] = {
            "horizon": horizon,
            "is_primary": horizon == int(plan_cfg["primary_horizon"]),
            "is_ood": False,
            "arms": arm_stats,
            "world_model_fingerprint": fingerprint_after,
        }
    if model_fingerprint(model) != fingerprint_before:
        raise RuntimeError("world model changed during planner comparison")
    return {
        "seed": int(seed),
        "split_ids": {name: ids.tolist() for name, ids in split_ids.items()},
        "world_model_fingerprint": fingerprint_before,
        "inverse_dynamics_mse_no_delta": id_mse_a,
        "inverse_dynamics_mse_with_delta": id_mse_b,
        "inverse_dynamics_weight_shape": list(weights_b.shape),
        "horizons": horizon_reports,
        "protected_splits_executed": [],
        "development_metrics_computed": False,
    }


def adjudicate(seed_rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    primary_h = str(int(gates["primary_horizon"]))
    slack = float(gates["success_slack"])
    adjudicated = []
    all_passed = True
    for row in seed_rows:
        arms = row["horizons"][primary_h]["arms"]
        amortized = arms[str(gates["primary_arm"])]
        shooting = arms[str(gates["baseline"])]
        seed_passed = bool(
            float(amortized["success_rate"]) >= float(shooting["success_rate"]) - slack
            and float(amortized["mean_wall_clock_s"]) < float(shooting["mean_wall_clock_s"])
        )
        adjudicated.append(
            {
                **dict(row),
                "seed_passed": seed_passed,
                "primary_amortized_success": float(amortized["success_rate"]),
                "primary_shooting_success": float(shooting["success_rate"]),
                "primary_amortized_clock_s": float(amortized["mean_wall_clock_s"]),
                "primary_shooting_clock_s": float(shooting["mean_wall_clock_s"]),
            }
        )
        all_passed = all_passed and seed_passed
    status = str(gates["pass_status"]) if all_passed else str(gates["fail_status"])
    return {
        "experiment_id": "WM-LEFLOW-AMORTIZE-001",
        "status": status,
        "evidence_level": "Availability",
        "all_seeds_passed": all_passed,
        "not_a_leflow_reproduction": True,
        "integrity_blockers": [],
        "seed_rows": adjudicated,
        "claim_boundary": dict(config["claim_boundary"]),
        "does_not_relabel_hard002": True,
        "protected_splits_executed": [],
        "downloads_performed": [],
        "model_forwards_performed": ["tiny_jepa_ridge_cpu"],
        "splits_accessed": ["train", "confirmation"],
        "world_model_frozen": True,
    }


def run_confirmation(config: Mapping[str, Any], *, mknn_metrics_path: str | Path) -> dict[str, Any]:
    if not config.get("execution_authorized", False):
        raise RuntimeError("WM-LEFLOW-AMORTIZE-001 is not authorized")
    require_mknn_adjudication(mknn_metrics_path)
    rows = [run_seed(config, seed=int(seed)) for seed in config["confirmation_seeds"]]
    return adjudicate(rows, config)


def write_artifacts(metrics: Mapping[str, Any], config: Mapping[str, Any], *, command: str) -> None:
    output = Path(str(config["output_metrics"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = collect_provenance(
        command=command,
        resource_profile=str(config["resource_profile"]),
        seed=int(config["confirmation_seeds"][0]),
    )
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": str(output).replace("\\", "/"), "status": metrics["status"]},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/wm_leflow_amortize_v1.json"))
    parser.add_argument(
        "--mknn-metrics",
        default=str(ROOT / "artifacts/metrics/wm_platonic_mknn_v1.json"),
    )
    args = parser.parse_args(argv)
    if is_git_dirty():
        raise SystemExit("WM-LEFLOW-AMORTIZE-001 requires a clean git worktree")
    config = load_json_config(args.config)
    require_free_disk(str(config["resource_profile"]))
    metrics = run_confirmation(config, mknn_metrics_path=args.mknn_metrics)
    write_artifacts(
        metrics,
        config,
        command=f"python scripts/run_wm_leflow_amortize.py --config {args.config}",
    )
    print(json.dumps({"status": metrics["status"], "all_seeds_passed": metrics["all_seeds_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
