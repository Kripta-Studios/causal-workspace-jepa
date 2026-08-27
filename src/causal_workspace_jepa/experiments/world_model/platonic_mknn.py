"""CPU Platonic m-kNN control: two frozen observation maps, same PointMass dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal_workspace_jepa.common.provenance import collect_provenance, is_git_dirty, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.data.splits import deterministic_named_split_ids
from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.interpretability.mutual_knn import chance_reference, mutual_knn
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA, evaluate_latent_mse

ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_SPLIT_NAMES = ("test", "paraphrase")
PASS_STATUS = "TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED"
FAIL_STATUS = "NEGATIVE_RESULT"


def load_json_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_array(array: np.ndarray) -> str:
    packed = np.ascontiguousarray(array)
    return hashlib.sha256(
        packed.tobytes() + str(tuple(packed.shape)).encode("utf-8") + str(packed.dtype).encode("utf-8")
    ).hexdigest()


def frozen_linear_map(state_dim: int, obs_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(state_dim), size=(state_dim, obs_dim)).astype(np.float32)


def apply_map(states: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return states @ matrix


def identity_encoder(obs_dim: int, latent_dim: int) -> np.ndarray:
    return np.eye(obs_dim, latent_dim, dtype=np.float32)


def reject_forbidden_seed(seed: int, forbidden: Mapping[str, Any] | list[int]) -> None:
    blocked = set(int(item) for item in forbidden)
    if int(seed) in blocked:
        raise ValueError(f"seed {seed} is frozen for another experiment")


def assert_protocol(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "WM-PLATONIC-MKNN-001":
        raise ValueError("wrong experiment_id")
    for name in config["splits"]["forbidden_split_names"]:
        if name not in FORBIDDEN_SPLIT_NAMES:
            raise ValueError("forbidden split contract drifted")
    if config["observation_maps"]["trainable"] or not config["observation_maps"]["learned_adapter_forbidden"]:
        raise ValueError("observation maps must stay frozen with no adapter")
    k = int(config["mknn"]["k"])
    n_eval = int(config["mknn"]["n_eval"])
    expected_chance = chance_reference(n_eval=n_eval, k=k)
    if abs(float(config["mknn"]["chance"]) - expected_chance) > 1e-12:
        raise ValueError("chance reference must equal k/(n_eval-1)")
    if int(config["splits"]["train"]) + int(config["splits"]["development"]) + int(
        config["splits"]["confirmation"]
    ) != int(config["environment"]["trajectories"]):
        raise ValueError("split sizes must cover the frozen trajectory population")


def _fit(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    latent_dim: int,
    seed: int,
    ridge: float,
    action_mode: str,
    encoder: np.ndarray,
) -> TinyActionConditionedJEPA:
    return TinyActionConditionedJEPA.fit(
        observations,
        actions,
        latent_dim=latent_dim,
        seed=seed,
        ridge=ridge,
        action_mode=action_mode,
        frozen_encoder=encoder,
    )


def _predict_next(
    model: TinyActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    latent = model.encode(observations)
    predicted = model.predict(latent, actions[:, None, :], return_intermediates=False)
    return predicted.predicted_latents[:, 0, :]


def _take_eval_states(observations: np.ndarray, n_eval: int) -> np.ndarray:
    flat = observations.reshape(-1, observations.shape[-1])
    if flat.shape[0] < n_eval:
        raise ValueError("not enough confirmation states for n_eval")
    return flat[:n_eval]


def _take_eval_transitions(
    observations: np.ndarray,
    actions: np.ndarray,
    n_eval: int,
) -> tuple[np.ndarray, np.ndarray]:
    obs_t = observations[:, :-1, :].reshape(-1, observations.shape[-1])
    act_t = actions.reshape(-1, actions.shape[-1])
    if obs_t.shape[0] < n_eval:
        raise ValueError("not enough confirmation transitions for n_eval")
    return obs_t[:n_eval], act_t[:n_eval]


def run_seed(config: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    assert_protocol(config)
    reject_forbidden_seed(seed, config["forbidden_seeds"])
    env = config["environment"]
    maps = config["observation_maps"]
    model_cfg = config["model"]
    splits_cfg = config["splits"]
    mknn_cfg = config["mknn"]
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
        raise RuntimeError("protected split name leaked into Platonic control")
    train_states = dataset.states[split_ids["train"]]
    train_actions = dataset.actions[split_ids["train"]]
    confirm_states = dataset.states[split_ids["confirmation"]]
    confirm_actions = dataset.actions[split_ids["confirmation"]]
    map_a = frozen_linear_map(int(env["state_dim"]), int(maps["obs_dim"]), int(maps["map_a_seed"]))
    map_b = frozen_linear_map(int(env["state_dim"]), int(maps["obs_dim"]), int(maps["map_b_seed"]))
    map_r = frozen_linear_map(int(env["state_dim"]), int(maps["obs_dim"]), int(maps["map_random_seed"]))
    encoder = identity_encoder(int(maps["obs_dim"]), int(model_cfg["latent_dim"]))
    noise_train = np.random.default_rng(int(maps["noise_seed"])).normal(size=train_states.shape).astype(
        np.float32
    )
    noise_confirm = (
        np.random.default_rng(int(maps["noise_seed"]) + 1)
        .normal(size=confirm_states.shape)
        .astype(np.float32)
    )
    obs_a_train = apply_map(train_states, map_a)
    obs_b_train = apply_map(train_states, map_b)
    obs_r_train = apply_map(noise_train, map_r)
    obs_a_confirm = apply_map(confirm_states, map_a)
    obs_b_confirm = apply_map(confirm_states, map_b)
    obs_r_confirm = apply_map(noise_confirm, map_r)
    if sha256_array(map_a) != sha256_array(
        frozen_linear_map(int(env["state_dim"]), int(maps["obs_dim"]), int(maps["map_a_seed"]))
    ):
        raise RuntimeError("observation map A was not frozen")
    model_a = _fit(
        obs_a_train,
        train_actions,
        latent_dim=int(model_cfg["latent_dim"]),
        seed=int(seed),
        ridge=float(model_cfg["ridge"]),
        action_mode="conditioned",
        encoder=encoder,
    )
    model_b = _fit(
        obs_b_train,
        train_actions,
        latent_dim=int(model_cfg["latent_dim"]),
        seed=int(seed),
        ridge=float(model_cfg["ridge"]),
        action_mode="conditioned",
        encoder=encoder,
    )
    model_shuffle = _fit(
        obs_b_train,
        train_actions,
        latent_dim=int(model_cfg["latent_dim"]),
        seed=int(seed),
        ridge=float(model_cfg["ridge"]),
        action_mode="shuffled_action",
        encoder=encoder,
    )
    model_random = _fit(
        obs_r_train,
        train_actions,
        latent_dim=int(model_cfg["latent_dim"]),
        seed=int(seed),
        ridge=float(model_cfg["ridge"]),
        action_mode="conditioned",
        encoder=encoder,
    )
    if not np.array_equal(model_a.encoder, encoder) or not np.array_equal(model_b.encoder, encoder):
        raise RuntimeError("encoders must remain the frozen identity")
    n_eval = int(mknn_cfg["n_eval"])
    k = int(mknn_cfg["k"])
    probe = np.asarray(mknn_cfg["probe_action"], dtype=np.float32)
    enc_a = model_a.encode(_take_eval_states(obs_a_confirm, n_eval)).tensor
    enc_b = model_b.encode(_take_eval_states(obs_b_confirm, n_eval)).tensor
    enc_r = model_random.encode(_take_eval_states(obs_r_confirm, n_eval)).tensor
    trans_a, act_eval = _take_eval_transitions(obs_a_confirm, confirm_actions, n_eval)
    trans_b, act_b = _take_eval_transitions(obs_b_confirm, confirm_actions, n_eval)
    trans_r, act_r = _take_eval_transitions(obs_r_confirm, confirm_actions, n_eval)
    if not np.array_equal(act_eval, act_b):
        raise RuntimeError("paired comparisons must share confirmation action identity")
    pred_a = _predict_next(model_a, trans_a, act_eval)
    pred_b = _predict_next(model_b, trans_b, act_eval)
    pred_shuffle = _predict_next(model_shuffle, trans_b, act_eval)
    pred_random = _predict_next(model_random, trans_r, act_r)
    probe_actions = np.repeat(probe[None, :], n_eval, axis=0)
    probe_a = _predict_next(model_a, trans_a, probe_actions)
    probe_b = _predict_next(model_b, trans_b, probe_actions)
    probe_shuffle = _predict_next(model_shuffle, trans_b, probe_actions)
    probe_random = _predict_next(model_random, trans_r, probe_actions)
    return {
        "seed": int(seed),
        "split_ids": {name: ids.tolist() for name, ids in split_ids.items()},
        "train_state_sha256": sha256_array(train_states),
        "confirmation_state_sha256": sha256_array(confirm_states),
        "map_a_sha256": sha256_array(map_a),
        "map_b_sha256": sha256_array(map_b),
        "map_random_sha256": sha256_array(map_r),
        "encoder_sha256": sha256_array(encoder),
        "encoder_mknn_ab": mutual_knn(enc_a, enc_b, k=k),
        "encoder_mknn_a_random": mutual_knn(enc_a, enc_r, k=k),
        "predictor_mknn_ab": mutual_knn(pred_a, pred_b, k=k),
        "predictor_mknn_a_shuffle": mutual_knn(pred_a, pred_shuffle, k=k),
        "predictor_mknn_a_random": mutual_knn(pred_a, pred_random, k=k),
        "action_conditioned_mknn_ab": mutual_knn(probe_a, probe_b, k=k),
        "action_conditioned_mknn_a_shuffle": mutual_knn(probe_a, probe_shuffle, k=k),
        "action_conditioned_mknn_a_random": mutual_knn(probe_a, probe_random, k=k),
        "conditioned_latent_mse_b": evaluate_latent_mse(model_b, obs_b_confirm, confirm_actions),
        "shuffled_latent_mse_b": evaluate_latent_mse(model_shuffle, obs_b_confirm, confirm_actions),
        "chance": chance_reference(n_eval=n_eval, k=k),
        "protected_splits_executed": [],
        "development_metrics_computed": False,
    }


def adjudicate(seed_rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply frozen config gates. Thresholds are never derived from seed_rows."""

    gates = config["gates"]
    chance = float(config["mknn"]["chance"])
    multiplier = float(gates["predictor_mknn_must_beat_chance_multiplier"])
    chance_floor = multiplier * chance
    adjudicated = []
    all_passed = True
    for row in seed_rows:
        primary = float(row["predictor_mknn_ab"])
        seed_passed = bool(
            primary > float(row["predictor_mknn_a_shuffle"])
            and primary > float(row["predictor_mknn_a_random"])
            and primary > chance_floor
        )
        adjudicated.append({**dict(row), "seed_passed": seed_passed, "chance_floor": chance_floor})
        all_passed = all_passed and seed_passed
    status = str(gates["pass_status"]) if all_passed else str(gates["fail_status"])
    evidence_level = str(config["claim_boundary"]["evidence_level_if_pass"]) if all_passed else "None"
    encoder_already_high = any(
        float(row["encoder_mknn_ab"]) > chance_floor for row in seed_rows
    )
    return {
        "experiment_id": "WM-PLATONIC-MKNN-001",
        "status": status,
        "evidence_level": evidence_level,
        "all_seeds_passed": all_passed,
        "chance_floor": chance_floor,
        "encoder_geometry_already_above_chance_floor": encoder_already_high,
        "integrity_blockers": [],
        "seed_rows": adjudicated,
        "claim_boundary": dict(config["claim_boundary"]),
        "does_not_relabel_hard002": True,
        "stitching_executed": False,
        "protected_splits_executed": [],
        "downloads_performed": [],
        "model_forwards_performed": ["tiny_jepa_ridge_cpu"],
        "splits_accessed": ["train", "confirmation"],
    }


def run_confirmation(config: Mapping[str, Any]) -> dict[str, Any]:
    if not config.get("execution_authorized", False):
        raise RuntimeError("WM-PLATONIC-MKNN-001 is not authorized")
    rows = [run_seed(config, seed=int(seed)) for seed in config["confirmation_seeds"]]
    return adjudicate(rows, config)


def write_artifacts(metrics: Mapping[str, Any], config: Mapping[str, Any], *, command: str) -> None:
    output = Path(str(config["output_metrics"]))
    provenance = collect_provenance(
        command=command,
        resource_profile=str(config["resource_profile"]),
        seed=int(config["confirmation_seeds"][0]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": str(output).replace("\\", "/"), "status": metrics["status"]},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/wm_platonic_mknn_v1.json"))
    args = parser.parse_args(argv)
    if is_git_dirty():
        raise SystemExit("WM-PLATONIC-MKNN-001 requires a clean git worktree")
    config = load_json_config(args.config)
    require_free_disk(str(config["resource_profile"]))
    metrics = run_confirmation(config)
    write_artifacts(
        metrics,
        config,
        command=f"python scripts/run_wm_platonic_mknn.py --config {args.config}",
    )
    print(json.dumps({"status": metrics["status"], "all_seeds_passed": metrics["all_seeds_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
