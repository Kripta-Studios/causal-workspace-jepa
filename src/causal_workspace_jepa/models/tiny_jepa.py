"""Tiny NumPy action-conditioned JEPA for CPU smoke validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from causal_workspace_jepa.common.types import LatentState, WorldModelOutput


@dataclass(frozen=True)
class TinyJepaConfig:
    observation_dim: int
    action_dim: int
    latent_dim: int = 16
    ridge: float = 1e-4
    seed: int = 0


class TinyActionConditionedJEPA:
    """Small linear JEPA that satisfies the world-model protocol.

    The encoder is a deterministic random projection. The predictor is fitted
    by ridge regression on latent transitions conditioned on environment
    actions. This is intentionally small enough for the CPU VPS.
    """

    def __init__(
        self,
        config: TinyJepaConfig,
        encoder: np.ndarray,
        predictor: np.ndarray,
        decoder: np.ndarray,
        latent_mean: np.ndarray,
    ) -> None:
        self.config = config
        self.encoder = encoder.astype(np.float32)
        self.predictor = predictor.astype(np.float32)
        self.decoder = decoder.astype(np.float32)
        self.latent_mean = latent_mean.astype(np.float32)

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        latent_dim: int = 16,
        seed: int = 0,
        ridge: float = 1e-4,
        action_mode: str = "conditioned",
    ) -> "TinyActionConditionedJEPA":
        if observations.ndim != 3:
            raise ValueError("observations must have shape [trajectories, steps, observation_dim]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [trajectories, steps - 1, action_dim]")
        obs_dim = int(observations.shape[-1])
        action_dim = int(actions.shape[-1])
        rng = np.random.default_rng(seed)
        encoder = rng.normal(0.0, 1.0 / np.sqrt(obs_dim), size=(obs_dim, latent_dim)).astype(np.float32)
        z = observations @ encoder
        z_t = z[:, :-1, :].reshape(-1, latent_dim)
        z_next = z[:, 1:, :].reshape(-1, latent_dim)
        flat_actions = actions.reshape(-1, action_dim).astype(np.float32)
        if action_mode == "no_action":
            design = _with_bias(z_t)
        elif action_mode == "shuffled_action":
            shuffled = flat_actions.copy()
            rng.shuffle(shuffled, axis=0)
            design = _with_bias(np.concatenate([z_t, shuffled], axis=-1))
        elif action_mode == "conditioned":
            design = _with_bias(np.concatenate([z_t, flat_actions], axis=-1))
        else:
            raise ValueError(f"unknown action_mode: {action_mode}")
        predictor = _ridge_solve(design, z_next, ridge)
        decoder = _ridge_solve(_with_bias(z.reshape(-1, latent_dim)), observations.reshape(-1, obs_dim), ridge)
        if action_mode == "no_action":
            padded = np.zeros((latent_dim + action_dim + 1, latent_dim), dtype=np.float32)
            padded[:latent_dim, :] = predictor[:latent_dim, :]
            padded[-1, :] = predictor[-1, :]
            predictor = padded
        return cls(
            TinyJepaConfig(
                observation_dim=obs_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                ridge=ridge,
                seed=seed,
            ),
            encoder=encoder,
            predictor=predictor,
            decoder=decoder,
            latent_mean=z_t.mean(axis=0),
        )

    def encode(self, observation: np.ndarray) -> LatentState:
        return LatentState(np.asarray(observation, dtype=np.float32) @ self.encoder, names=("latent",))

    def predict(
        self,
        latent: LatentState,
        actions: np.ndarray,
        *,
        return_intermediates: bool = False,
    ) -> WorldModelOutput:
        current = np.asarray(latent.tensor, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[None, :, :]
        if current.ndim == 1:
            current = current[None, :]
        predictions = []
        feature_rows = []
        for step in range(actions.shape[1]):
            features = _with_bias(np.concatenate([current, actions[:, step, :]], axis=-1))
            current = features @ self.predictor
            predictions.append(current)
            feature_rows.append(features)
        predicted = np.stack(predictions, axis=1)
        decoded = {
            "state": _with_bias(predicted.reshape(-1, self.config.latent_dim))
            @ self.decoder.reshape(self.config.latent_dim + 1, self.config.observation_dim)
        }
        decoded["state"] = decoded["state"].reshape(predicted.shape[0], predicted.shape[1], -1)
        intermediates = {}
        if return_intermediates:
            intermediates = {
                "predictor.input": np.stack(feature_rows, axis=1),
                "predictor.latent": predicted,
            }
        return WorldModelOutput(
            predicted_latents=predicted,
            uncertainty=None,
            decoded_state=decoded,
            intermediates=intermediates,
            action_embeddings=actions,
            cost_features=decoded["state"][..., :2],
        )

    def decode_state(self, latent: LatentState) -> Mapping[str, np.ndarray]:
        tensor = np.asarray(latent.tensor, dtype=np.float32)
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        state = _with_bias(tensor) @ self.decoder
        return {"state": state}

    def named_activation_points(self) -> Sequence[str]:
        return ("encoder.latent", "predictor.input", "predictor.latent")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            config_json=np.array(json.dumps(self.config.__dict__, sort_keys=True)),
            encoder=self.encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            latent_mean=self.latent_mean,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyActionConditionedJEPA":
        loaded = np.load(path, allow_pickle=False)
        config = TinyJepaConfig(**json.loads(str(loaded["config_json"])))
        return cls(
            config=config,
            encoder=loaded["encoder"],
            predictor=loaded["predictor"],
            decoder=loaded["decoder"],
            latent_mean=loaded["latent_mean"],
        )


def evaluate_latent_mse(
    model: TinyActionConditionedJEPA,
    observations: np.ndarray,
    actions: np.ndarray,
) -> float:
    latent = model.encode(observations[:, 0, :])
    prediction = model.predict(latent, actions, return_intermediates=False).predicted_latents
    target = model.encode(observations[:, 1:, :]).tensor
    return float(np.mean((prediction - target) ** 2))


def mean_latent_baseline_mse(
    model: TinyActionConditionedJEPA,
    observations: np.ndarray,
) -> float:
    target = model.encode(observations[:, 1:, :]).tensor
    mean = model.latent_mean
    prediction = np.broadcast_to(mean, target.shape)
    return float(np.mean((prediction - target) ** 2))


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, np.ones((*array.shape[:-1], 1), dtype=array.dtype)], axis=-1)


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    xtx = design.T @ design
    regularizer = ridge * np.eye(xtx.shape[0], dtype=np.float32)
    return np.linalg.solve(xtx + regularizer, design.T @ target).astype(np.float32)
