"""CPU MiniPush amortized planning. Not a LeFlow reproduction and not T2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from causal_workspace_jepa.common.provenance import collect_provenance, is_git_dirty, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.splits import deterministic_named_split_ids
from causal_workspace_jepa.data.synthetic.minipush import (
    constructed_goal_observation,
    generate_minipush,
    manhattan_xy,
    minipush_rollout_state,
    object_goal_l2,
    quantize_minipush_actions,
)
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA
from causal_workspace_jepa.planning.amortized_latent import (
    action_flow_plan,
    amortized_latent_plan,
    decode_actions,
    fit_action_flow_pairs,
    fit_inverse_dynamics,
)
from causal_workspace_jepa.planning.cem import iterative_cem_plan, random_shooting_plan

ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_SPLIT_NAMES = ("test", "paraphrase")
REQUIRED_ARM_FIELDS = (
    "success_rate",
    "failure_rate",
    "mean_goal_distance",
    "mean_terminal_latent_goal_distance",
    "mean_wall_clock_s",
    "sum_wall_clock_s",
    "mean_wm_rollout_forwards",
    "mean_planner_forwards",
    "mean_id_forwards",
    "mean_candidates_evaluated",
    "mean_cem_iterations",
    "n_tasks",
    "cpu_peak_rss_bytes",
    "goal_distances",
    "failure_kinds",
)


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


def peak_rss_bytes() -> int | None:
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def assert_protocol(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "WM-AMORTIZED-PLANNING-MINIPUSH-002":
        raise ValueError("wrong experiment_id")
    if config.get("parent_t2_must_not_be_mutated") != "WM-LEFLOW-AMORTIZE-001":
        raise ValueError("T2 parent identity drifted")
    if not config.get("not_a_leflow_reproduction", False) or not config.get("not_a_rectified_flow", False):
        raise ValueError("must not be labeled LeFlow or Rectified Flow")
    if not config.get("not_levljepa_factorial", False):
        raise ValueError("must not execute the LeVLJEPA factorial")
    if int(config["splits"]["train"]) + int(config["splits"]["development"]) + int(
        config["splits"]["confirmation"]
    ) != int(config["environment"]["trajectories"]):
        raise ValueError("split sizes must cover the frozen trajectory population")
    if int(config["qualification_seed"]) in {int(s) for s in config["confirmation_seeds"]}:
        raise ValueError("qualification seed must not be a confirmation seed")
    if config["planning"]["cost_mode"] != "latent_goal":
        raise ValueError("all MiniPush planners must share latent-goal cost")
    if not config["planning"]["quantize_to_cardinals"]:
        raise ValueError("MiniPush actions must be quantized to cardinals")


def _latents(model: TinyActionConditionedJEPA, observations: np.ndarray) -> np.ndarray:
    packed = observations.reshape(-1, observations.shape[-1])
    encoded = model.encode(packed).tensor
    return encoded.reshape(*observations.shape[:-1], -1)


def _tasks(states: np.ndarray) -> list[dict[str, np.ndarray]]:
    tasks = []
    for index in range(states.shape[0]):
        start = states[index, 0].astype(np.float32)
        goal_observation = constructed_goal_observation(start)
        tasks.append(
            {
                "task_id": np.array([index], dtype=np.int32),
                "start": start,
                "goal_observation": goal_observation,
            }
        )
    return tasks


def _assert_identical_goals(tasks: list[dict[str, np.ndarray]]) -> None:
    for task in tasks:
        start = task["start"]
        goal = task["goal_observation"]
        if goal.shape != (6,):
            raise RuntimeError("goal observation must be the full 6-d vector")
        if not np.allclose(goal[:2], start[:2]) or not np.allclose(goal[2:6], np.tile(start[4:6], 2)):
            raise RuntimeError("constructed goal drifted from the frozen constructor")


def _failure_kind(start: np.ndarray, final: np.ndarray, *, horizon: int, success: bool) -> str:
    if success:
        return "success"
    approach = manhattan_xy(start[:2], start[2:4]) + manhattan_xy(start[2:4], start[4:6]) - 1
    if approach > int(horizon):
        return "horizon_insufficient"
    if np.allclose(final[2:4], start[2:4]):
        return "no_object_motion"
    return "goal_miss"


def _arm_compute(arm_name: str, horizon: int, plan_cfg: Mapping[str, Any]) -> tuple[int, int, int, int]:
    shooting = int(plan_cfg["shooting_candidates"])
    cem_n = int(plan_cfg["cem_candidates"])
    cem_it = int(plan_cfg["cem_iterations"])
    n1, n_rerank = (int(item) for item in plan_cfg["amortized_rerank_n"])
    if arm_name == "random_shooting_n64":
        return shooting, 0, 0, 1
    if arm_name == "iterative_cem":
        return cem_n * cem_it, 0, 0, cem_it
    if arm_name == "latent_flow_n1":
        return n1, n1, n1 * int(horizon), 1
    if arm_name == "latent_flow_n64":
        return n_rerank, n_rerank, n_rerank * int(horizon), 1
    if arm_name == "action_flow_n1":
        return 0, 1, 0, 1
    raise ValueError(arm_name)


def _one_step_rmse(model: TinyActionConditionedJEPA, observations: np.ndarray, actions: np.ndarray) -> dict[str, float]:
    z_t = model.encode(observations[:, :-1, :].reshape(-1, observations.shape[-1]))
    predicted = model.predict(z_t, actions.reshape(-1, 1, actions.shape[-1]), return_intermediates=False)
    assert predicted.decoded_state is not None
    decoded = predicted.decoded_state["state"][:, 0, :]
    true = observations[:, 1:, :].reshape(-1, observations.shape[-1])
    return {
        "one_step_state_rmse": float(np.sqrt(np.mean((decoded - true) ** 2))),
        "one_step_object_rmse": float(np.sqrt(np.mean((decoded[:, 2:4] - true[:, 2:4]) ** 2))),
    }


def _time_plan(fn: Callable[[], Mapping[str, Any]]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    plan = dict(fn())
    elapsed = time.perf_counter() - started
    return plan, elapsed


def _require_arm_schema(arm: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARM_FIELDS if field not in arm]
    if missing:
        raise RuntimeError(f"MiniPush planner artifact missing fields: {missing}")


def run_seed(config: Mapping[str, Any], *, seed: int, eval_split: str) -> dict[str, Any]:
    assert_protocol(config)
    reject_forbidden_seed(seed, list(config["forbidden_seeds"]))
    if eval_split not in {"development", "confirmation"}:
        raise ValueError("eval_split must be development or confirmation")
    if eval_split in FORBIDDEN_SPLIT_NAMES:
        raise RuntimeError("protected split name leaked")
    env = config["environment"]
    model_cfg = config["model"]
    splits_cfg = config["splits"]
    plan_cfg = config["planning"]
    id_cfg = config["inverse_dynamics"]
    dataset = generate_minipush(
        trajectories=int(env["trajectories"]),
        steps=int(env["steps"]),
        seed=int(env["environment_base_seed"]) + int(seed),
        resolution=int(env["resolution"]),
    )
    split_ids = deterministic_named_split_ids(
        dataset.states.shape[0],
        int(splits_cfg["split_seed"]),
        train=int(splits_cfg["train"]),
        development=int(splits_cfg["development"]),
        confirmation=int(splits_cfg["confirmation"]),
    )
    if any(name in split_ids for name in FORBIDDEN_SPLIT_NAMES):
        raise RuntimeError("protected split name leaked into MiniPush planning")
    if eval_split == "confirmation" and int(seed) == int(config["qualification_seed"]):
        raise RuntimeError("qualification seed confirmation split is closed")
    train_obs = dataset.states[split_ids["train"]].astype(np.float32)
    train_actions = dataset.actions[split_ids["train"]].astype(np.float32)
    eval_obs = dataset.states[split_ids[eval_split]].astype(np.float32)
    eval_actions = dataset.actions[split_ids[eval_split]].astype(np.float32)
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
    eval_latents = _latents(model, eval_obs)
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
    z_t = eval_latents[:, :-1, :].reshape(-1, eval_latents.shape[-1])
    z_next = eval_latents[:, 1:, :].reshape(-1, eval_latents.shape[-1])
    true_a = eval_actions.reshape(-1, eval_actions.shape[-1])
    pred_a = decode_actions(weights_a, z_t, z_next, include_delta=False)
    pred_b = decode_actions(weights_b, z_t, z_next, include_delta=True)
    id_mse_a = float(np.mean((pred_a - true_a) ** 2))
    id_mse_b = float(np.mean((pred_b - true_a) ** 2))
    wm_error = _one_step_rmse(model, eval_obs, eval_actions)
    success_max = float(plan_cfg["success_object_l2_max"])
    resolution = int(env["resolution"])
    rss = peak_rss_bytes()
    horizon_reports: dict[str, Any] = {}
    quantize = quantize_minipush_actions
    for horizon in [int(plan_cfg["primary_horizon"]), *[int(item) for item in plan_cfg["diagnostic_horizons"]]]:
        train_goals = np.stack([constructed_goal_observation(state) for state in train_obs[:, 0]])
        z_train_goal = model.encode(train_goals).tensor
        action_weights = fit_action_flow_pairs(
            train_latents[:, 0, :],
            z_train_goal,
            train_actions[:, :horizon],
            ridge=float(id_cfg["ridge"]),
        )
        tasks = _tasks(eval_obs)
        _assert_identical_goals(tasks)
        n1, n_rerank = (int(item) for item in plan_cfg["amortized_rerank_n"])
        planners: dict[str, Callable[[dict[str, np.ndarray]], dict[str, Any]]] = {
            "random_shooting_n64": lambda task, h=horizon: random_shooting_plan(
                model,
                task["start"],
                task["goal_observation"],
                horizon=h,
                candidates=int(plan_cfg["shooting_candidates"]),
                seed=int(seed) + 1000 * h + int(task["task_id"][0]),
                action_low=float(plan_cfg["action_low"]),
                action_high=float(plan_cfg["action_high"]),
                cost_mode="latent_goal",
                quantize_fn=quantize,
            ),
            "iterative_cem": lambda task, h=horizon: iterative_cem_plan(
                model,
                task["start"],
                task["goal_observation"],
                horizon=h,
                candidates=int(plan_cfg["cem_candidates"]),
                iterations=int(plan_cfg["cem_iterations"]),
                elite_fraction=float(plan_cfg["cem_elite_fraction"]),
                seed=int(seed) + 2000 * h + int(task["task_id"][0]),
                action_low=float(plan_cfg["action_low"]),
                action_high=float(plan_cfg["action_high"]),
                cost_mode="latent_goal",
                quantize_fn=quantize,
            ),
            "latent_flow_n1": lambda task, h=horizon: amortized_latent_plan(
                model,
                weights_b,
                task["start"],
                task["goal_observation"],
                horizon=h,
                candidates=n1,
                seed=int(seed) + 3000 * h + int(task["task_id"][0]),
                noise_std=float(plan_cfg["n1_noise_std"]),
                include_delta=True,
                cost_mode="latent_goal",
                quantize_fn=quantize,
            ),
            "latent_flow_n64": lambda task, h=horizon: amortized_latent_plan(
                model,
                weights_b,
                task["start"],
                task["goal_observation"],
                horizon=h,
                candidates=n_rerank,
                seed=int(seed) + 4000 * h + int(task["task_id"][0]),
                noise_std=float(plan_cfg["n64_noise_std"]),
                include_delta=True,
                cost_mode="latent_goal",
                quantize_fn=quantize,
            ),
            "action_flow_n1": lambda task, h=horizon, w=action_weights: action_flow_plan(
                model,
                w,
                task["start"],
                task["goal_observation"],
                horizon=h,
                quantize_fn=quantize,
            ),
        }
        arm_stats: dict[str, dict[str, Any]] = {}
        for arm_name, planner in planners.items():
            successes: list[float] = []
            distances: list[float] = []
            latent_distances: list[float] = []
            clocks: list[float] = []
            kinds: list[str] = []
            wm_fwd, planner_fwd, id_fwd, cem_iters = _arm_compute(arm_name, horizon, plan_cfg)
            goal_hashes = []
            for task in tasks:
                plan, elapsed = _time_plan(lambda p=planner, t=task: p(t))
                actions = quantize(np.asarray(plan["actions"], dtype=np.float32))
                final = minipush_rollout_state(task["start"], actions, resolution=resolution)
                distance = object_goal_l2(final)
                success = bool(distance < success_max)
                z_final = model.encode(final[None, :]).tensor[0]
                z_goal = model.encode(task["goal_observation"][None, :]).tensor[0]
                goal_hashes.append(sha256_array(task["goal_observation"]))
                successes.append(float(success))
                distances.append(distance)
                latent_distances.append(float(np.sum((z_final - z_goal) ** 2)))
                clocks.append(elapsed)
                kinds.append(_failure_kind(task["start"], final, horizon=horizon, success=success))
            if len(goal_hashes) != len(tasks):
                raise RuntimeError("goal information missing for a task")
            arm = {
                "success_rate": float(np.mean(successes)),
                "failure_rate": float(1.0 - np.mean(successes)),
                "mean_goal_distance": float(np.mean(distances)),
                "mean_terminal_latent_goal_distance": float(np.mean(latent_distances)),
                "mean_wall_clock_s": float(np.mean(clocks)),
                "sum_wall_clock_s": float(np.sum(clocks)),
                "mean_wm_rollout_forwards": float(wm_fwd),
                "mean_planner_forwards": float(planner_fwd),
                "mean_id_forwards": float(id_fwd),
                "mean_candidates_evaluated": float(wm_fwd if arm_name != "action_flow_n1" else 1),
                "mean_cem_iterations": float(cem_iters),
                "n_tasks": float(len(tasks)),
                "cpu_peak_rss_bytes": rss,
                "goal_distances": [float(item) for item in distances],
                "failure_kinds": kinds,
                "horizon": float(horizon),
            }
            if arm_name == "action_flow_n1":
                arm["mean_candidates_evaluated"] = 1.0
            _require_arm_schema(arm)
            arm_stats[arm_name] = arm
        fingerprint_after = model_fingerprint(model)
        if fingerprint_after != fingerprint_before:
            raise RuntimeError("world model changed during planner comparison")
        horizon_reports[str(horizon)] = {
            "horizon": horizon,
            "is_primary": horizon == int(plan_cfg["primary_horizon"]),
            "is_ood": False,
            "arms": arm_stats,
            "world_model_fingerprint": fingerprint_after,
            "goal_constructor": str(plan_cfg["goal_constructor"]),
            "cost_mode": "latent_goal",
        }
    if model_fingerprint(model) != fingerprint_before:
        raise RuntimeError("world model changed during planner comparison")
    return {
        "seed": int(seed),
        "eval_split": eval_split,
        "split_ids": {name: ids.tolist() for name, ids in split_ids.items()},
        "world_model_fingerprint": fingerprint_before,
        "inverse_dynamics_mse_no_delta": id_mse_a,
        "inverse_dynamics_mse_with_delta": id_mse_b,
        "inverse_dynamics_mse_per_dim_no_delta": [
            float(np.mean((pred_a[:, dim] - true_a[:, dim]) ** 2)) for dim in range(true_a.shape[-1])
        ],
        "inverse_dynamics_mse_per_dim_with_delta": [
            float(np.mean((pred_b[:, dim] - true_a[:, dim]) ** 2)) for dim in range(true_a.shape[-1])
        ],
        "inverse_dynamics_weight_shape": list(weights_b.shape),
        **wm_error,
        "horizons": horizon_reports,
        "cpu_peak_rss_bytes": rss,
        "protected_splits_executed": [],
        "development_metrics_computed": eval_split == "development",
        "confirmation_metrics_computed": eval_split == "confirmation",
    }


def qualify_development(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["qualification_gates"]
    arms = row["horizons"][str(int(gates["horizon"]))]["arms"]
    wm_ok = float(row["one_step_state_rmse"]) < float(gates["max_one_step_state_rmse"]) and float(
        row["one_step_object_rmse"]
    ) < float(gates["max_one_step_object_rmse"])
    shooting = float(arms["random_shooting_n64"]["success_rate"])
    shooting_ok = shooting < float(gates["max_shooting_success"])
    trivial = all(float(arm["success_rate"]) >= float(gates["all_planners_trivial_success"]) for arm in arms.values())
    if not wm_ok:
        status = str(config["gates"]["wm_fail_status"])
    elif not shooting_ok or trivial:
        status = str(config["gates"]["uninformative_status"])
    else:
        status = "QUALIFICATION_PASSED"
    return {
        "experiment_id": str(config["qualification_experiment_id"]),
        "parent_experiment_id": str(config["experiment_id"]),
        "status": status,
        "evidence_level": "None",
        "qualification_seed": int(row["seed"]),
        "wm_ok": wm_ok,
        "shooting_ok": shooting_ok,
        "all_planners_trivial": trivial,
        "one_step_state_rmse": float(row["one_step_state_rmse"]),
        "one_step_object_rmse": float(row["one_step_object_rmse"]),
        "development_shooting_success": shooting,
        "seed_row": dict(row),
        "splits_accessed": ["train", "development"],
        "confirmation_opened": False,
        "downloads_performed": [],
        "model_forwards_performed": ["tiny_jepa_ridge_cpu"],
        "world_model_fingerprint": row["world_model_fingerprint"],
        "does_not_relabel_hard002": True,
        "does_not_mutate_t2": True,
    }


def adjudicate(seed_rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    primary_h = str(int(gates["primary_horizon"]))
    slack = float(gates["success_slack_vs_cem"])
    cem_delta = float(gates["cem_minus_shooting_min"])
    adjudicated = []
    informative = True
    competent = True
    for row in seed_rows:
        arms = row["horizons"][primary_h]["arms"]
        amortized = arms[str(gates["primary_arm"])]
        cem = arms[str(gates["search_baseline"])]
        shooting = arms[str(gates["shooting_baseline"])]
        search_useful = float(cem["success_rate"]) >= float(shooting["success_rate"]) + cem_delta
        shooting_ceiling = float(shooting["success_rate"]) >= float(
            config["qualification_gates"]["max_shooting_success"]
        )
        seed_passed = bool(
            search_useful
            and float(amortized["success_rate"]) >= float(cem["success_rate"]) - slack
            and float(amortized["mean_wall_clock_s"]) < float(cem["mean_wall_clock_s"])
            and float(amortized["mean_wm_rollout_forwards"]) < float(cem["mean_wm_rollout_forwards"])
        )
        adjudicated.append(
            {
                **dict(row),
                "seed_passed": seed_passed,
                "search_useful": search_useful,
                "shooting_ceiling": shooting_ceiling,
                "primary_amortized_success": float(amortized["success_rate"]),
                "primary_cem_success": float(cem["success_rate"]),
                "primary_shooting_success": float(shooting["success_rate"]),
                "primary_amortized_clock_s": float(amortized["mean_wall_clock_s"]),
                "primary_cem_clock_s": float(cem["mean_wall_clock_s"]),
                "primary_amortized_goal_distance": float(amortized["mean_goal_distance"]),
                "primary_cem_goal_distance": float(cem["mean_goal_distance"]),
            }
        )
        informative = informative and search_useful and not shooting_ceiling
        competent = competent and seed_passed
    if not informative:
        status = str(gates["uninformative_status"])
        all_passed = False
    elif competent:
        status = str(gates["pass_status"])
        all_passed = True
    else:
        status = str(gates["fail_status"])
        all_passed = False
    evidence_level = str(config["claim_boundary"]["evidence_level_if_pass"]) if all_passed else "None"
    return {
        "experiment_id": "WM-AMORTIZED-PLANNING-MINIPUSH-002",
        "status": status,
        "evidence_level": evidence_level,
        "all_seeds_passed": all_passed,
        "not_a_leflow_reproduction": True,
        "not_a_rectified_flow": True,
        "integrity_blockers": [],
        "seed_rows": adjudicated,
        "claim_boundary": dict(config["claim_boundary"]),
        "does_not_relabel_hard002": True,
        "does_not_mutate_t2": True,
        "stitching_executed": False,
        "protected_splits_executed": [],
        "downloads_performed": [],
        "model_forwards_performed": ["tiny_jepa_ridge_cpu"],
        "splits_accessed": ["train", "confirmation"],
        "world_model_frozen": True,
        "confirmation_opened": True,
    }


def require_qualification_passed(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "QUALIFICATION_PASSED":
        raise RuntimeError("confirmation is closed until qualification passes")
    if payload.get("confirmation_opened") is not False:
        raise RuntimeError("qualification artifact must not have opened confirmation")
    return payload


def write_artifacts(metrics: Mapping[str, Any], output: str | Path, *, command: str, seed: int, profile: str) -> None:
    output_path = Path(output)
    provenance = collect_provenance(command=command, resource_profile=profile, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = output_path.with_name(output_path.name[: -len(".json")] + ".provenance.json")
    write_provenance(
        sidecar,
        provenance,
        extra={"metrics": str(output_path).replace("\\", "/"), "status": metrics["status"]},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/wm_amortized_minipush_v1.json"))
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.qualify) == bool(args.confirm):
        raise SystemExit("specify exactly one of --qualify or --confirm")
    if is_git_dirty():
        raise SystemExit("WM-AMORTIZED-PLANNING-MINIPUSH-002 requires a clean git worktree")
    config = load_json_config(args.config)
    require_free_disk(str(config["resource_profile"]))
    if args.qualify:
        row = run_seed(config, seed=int(config["qualification_seed"]), eval_split="development")
        metrics = qualify_development(row, config)
        write_artifacts(
            metrics,
            config["output_qualification"],
            command=f"python scripts/run_wm_amortized_minipush.py --qualify --config {args.config}",
            seed=int(config["qualification_seed"]),
            profile=str(config["resource_profile"]),
        )
        print(json.dumps({"status": metrics["status"], "confirmation_opened": False}, indent=2))
        return 0
    require_qualification_passed(config["output_qualification"])
    rows = [run_seed(config, seed=int(seed), eval_split="confirmation") for seed in config["confirmation_seeds"]]
    metrics = adjudicate(rows, config)
    write_artifacts(
        metrics,
        config["output_metrics"],
        command=f"python scripts/run_wm_amortized_minipush.py --confirm --config {args.config}",
        seed=int(config["confirmation_seeds"][0]),
        profile=str(config["resource_profile"]),
    )
    print(json.dumps({"status": metrics["status"], "all_seeds_passed": metrics["all_seeds_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
