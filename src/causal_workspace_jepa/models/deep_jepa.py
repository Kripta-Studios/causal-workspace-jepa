"""Small learned action-conditioned JEPA with named internal sites.

The encoder is fixed so the CPU study cannot satisfy its objective by latent
collapse. Both predictor hidden layers and the latent output map are optimized
with deterministic NumPy Adam updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from causal_workspace_jepa.common.types import LatentState, WorldModelOutput


@dataclass(frozen=True)
class DeepJepaConfig:
    observation_dim: int
    action_dim: int
    latent_dim: int = 8
    hidden_dims: tuple[int, int] = (24, 16)
    learning_rate: float = 3e-3
    training_steps: int = 1_200
    batch_size: int = 128
    weight_decay: float = 1e-5
    seed: int = 0
    encoder_seed: int = 0


class DeepActionConditionedJEPA:
    """Two-hidden-layer latent predictor intended for CPU causal audits."""

    def __init__(
        self,
        config: DeepJepaConfig,
        *,
        encoder: np.ndarray,
        decoder: np.ndarray,
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        training_loss: float,
    ) -> None:
        self.config = config
        self.encoder = np.asarray(encoder, dtype=np.float32)
        self.decoder = np.asarray(decoder, dtype=np.float32)
        self.input_mean = np.asarray(input_mean, dtype=np.float32)
        self.input_scale = np.asarray(input_scale, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_scale = np.asarray(target_scale, dtype=np.float32)
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = (
            np.asarray(value, dtype=np.float32) for value in weights
        )
        self.training_loss = float(training_loss)

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        latent_dim: int = 8,
        hidden_dims: tuple[int, int] = (24, 16),
        learning_rate: float = 3e-3,
        training_steps: int = 1_200,
        batch_size: int = 128,
        weight_decay: float = 1e-5,
        seed: int = 0,
        encoder_seed: int = 0,
        encoder: np.ndarray | None = None,
    ) -> "DeepActionConditionedJEPA":
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 3:
            raise ValueError("observations must have shape [trajectories, steps, observation_dim]")
        if actions.shape[:2] != (observations.shape[0], observations.shape[1] - 1):
            raise ValueError("actions must align with observation transitions")
        if len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain two positive dimensions")

        observation_dim = int(observations.shape[-1])
        action_dim = int(actions.shape[-1])
        if encoder is None:
            encoder = _fixed_encoder(observation_dim, latent_dim, encoder_seed)
        encoder = np.asarray(encoder, dtype=np.float32)
        if encoder.shape != (observation_dim, latent_dim):
            raise ValueError("encoder shape does not match observation_dim and latent_dim")

        latents = observations @ encoder
        inputs = np.concatenate(
            [latents[:, :-1].reshape(-1, latent_dim), actions.reshape(-1, action_dim)],
            axis=-1,
        )
        targets = latents[:, 1:].reshape(-1, latent_dim)
        input_mean, input_scale = _mean_scale(inputs)
        target_mean, target_scale = _mean_scale(targets)
        normalized_inputs = (inputs - input_mean) / input_scale
        normalized_targets = (targets - target_mean) / target_scale

        rng = np.random.default_rng(seed)
        hidden1, hidden2 = hidden_dims
        parameters = [
            _xavier(rng, normalized_inputs.shape[-1], hidden1),
            np.zeros(hidden1, dtype=np.float32),
            _xavier(rng, hidden1, hidden2),
            np.zeros(hidden2, dtype=np.float32),
            _xavier(rng, hidden2, latent_dim),
            np.zeros(latent_dim, dtype=np.float32),
        ]
        first_moments = [np.zeros_like(parameter) for parameter in parameters]
        second_moments = [np.zeros_like(parameter) for parameter in parameters]
        effective_batch = min(batch_size, len(normalized_inputs))
        last_loss = float("nan")
        for step in range(1, training_steps + 1):
            batch_ids = rng.choice(len(normalized_inputs), size=effective_batch, replace=False)
            batch_x = normalized_inputs[batch_ids]
            batch_y = normalized_targets[batch_ids]
            prediction, hidden_1, hidden_2 = _forward(batch_x, parameters)
            error = prediction - batch_y
            last_loss = float(np.mean(error**2))
            gradients = _backward(
                batch_x,
                hidden_1,
                hidden_2,
                error,
                parameters,
                weight_decay,
            )
            _adam_step(
                parameters,
                gradients,
                first_moments,
                second_moments,
                step=step,
                learning_rate=learning_rate,
            )

        decoder_design = _with_bias(latents.reshape(-1, latent_dim))
        decoder = _ridge_solve(decoder_design, observations.reshape(-1, observation_dim), 1e-5)
        config = DeepJepaConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            training_steps=training_steps,
            batch_size=batch_size,
            weight_decay=weight_decay,
            seed=seed,
            encoder_seed=encoder_seed,
        )
        return cls(
            config,
            encoder=encoder,
            decoder=decoder,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            weights=tuple(parameters),  # type: ignore[arg-type]
            training_loss=last_loss,
        )

    def encode(self, observation: np.ndarray) -> LatentState:
        tensor = np.asarray(observation, dtype=np.float32) @ self.encoder
        return LatentState(tensor=tensor, names=("latent",))

    def hidden_activations(
        self,
        latent: np.ndarray,
        action: np.ndarray,
    ) -> dict[str, np.ndarray]:
        latent_array = np.asarray(latent, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if latent_array.ndim == 1:
            latent_array = latent_array[None, :]
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        normalized = (
            np.concatenate([latent_array, action_array], axis=-1) - self.input_mean
        ) / self.input_scale
        prediction, hidden1, hidden2 = _forward(
            normalized,
            [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3],
        )
        predicted_latent = prediction * self.target_scale + self.target_mean
        return {
            "predictor.hidden1": hidden1.astype(np.float32),
            "predictor.hidden2": hidden2.astype(np.float32),
            "predictor.latent": predicted_latent.astype(np.float32),
        }

    def predict_from_hidden(self, site: str, activation: np.ndarray) -> np.ndarray:
        hidden = np.asarray(activation, dtype=np.float32)
        if hidden.ndim == 1:
            hidden = hidden[None, :]
        if site == "predictor.hidden1":
            hidden = np.tanh(hidden @ self.w2 + self.b2)
        elif site != "predictor.hidden2":
            raise ValueError(f"unsupported hidden site: {site}")
        normalized = hidden @ self.w3 + self.b3
        return (normalized * self.target_scale + self.target_mean).astype(np.float32)

    def predict(
        self,
        latent: LatentState,
        actions: np.ndarray,
        *,
        return_intermediates: bool = False,
    ) -> WorldModelOutput:
        current = np.asarray(latent.tensor, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        if current.ndim == 1:
            current = current[None, :]
        if action_array.ndim == 2:
            action_array = action_array[None, :, :]
        trajectories: list[np.ndarray] = []
        hidden1: list[np.ndarray] = []
        hidden2: list[np.ndarray] = []
        for step in range(action_array.shape[1]):
            sites = self.hidden_activations(current, action_array[:, step])
            current = sites["predictor.latent"]
            trajectories.append(current)
            hidden1.append(sites["predictor.hidden1"])
            hidden2.append(sites["predictor.hidden2"])
        predicted = np.stack(trajectories, axis=1)
        decoded = self.decode_state(LatentState(predicted, names=("latent",)))["state"]
        intermediates: dict[str, np.ndarray] = {}
        if return_intermediates:
            intermediates = {
                "predictor.hidden1": np.stack(hidden1, axis=1),
                "predictor.hidden2": np.stack(hidden2, axis=1),
                "predictor.latent": predicted,
            }
        return WorldModelOutput(
            predicted_latents=predicted,
            uncertainty=None,
            decoded_state={"state": decoded},
            intermediates=intermediates,
            action_embeddings=action_array,
            cost_features=decoded[..., :2],
        )

    def decode_state(self, latent: LatentState) -> Mapping[str, np.ndarray]:
        tensor = np.asarray(latent.tensor, dtype=np.float32)
        shape = tensor.shape[:-1]
        flat = tensor.reshape(-1, self.config.latent_dim)
        decoded = _with_bias(flat) @ self.decoder
        return {"state": decoded.reshape(*shape, self.config.observation_dim).astype(np.float32)}

    def named_activation_points(self) -> Sequence[str]:
        return (
            "encoder.latent",
            "predictor.hidden1",
            "predictor.hidden2",
            "predictor.latent",
        )


def _fixed_encoder(observation_dim: int, latent_dim: int, seed: int) -> np.ndarray:
    if latent_dim < observation_dim:
        raise ValueError("latent_dim must be at least observation_dim for the fixed CPU encoder")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(latent_dim, observation_dim))
    orthonormal_columns, _ = np.linalg.qr(matrix)
    return orthonormal_columns.T.astype(np.float32)


def _mean_scale(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array.mean(axis=0).astype(np.float32)
    scale = array.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


def _forward(
    inputs: np.ndarray,
    parameters: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w1, b1, w2, b2, w3, b3 = parameters
    hidden1 = np.tanh(inputs @ w1 + b1)
    hidden2 = np.tanh(hidden1 @ w2 + b2)
    prediction = hidden2 @ w3 + b3
    return prediction, hidden1, hidden2


def _backward(
    inputs: np.ndarray,
    hidden1: np.ndarray,
    hidden2: np.ndarray,
    error: np.ndarray,
    parameters: Sequence[np.ndarray],
    weight_decay: float,
) -> list[np.ndarray]:
    w1, _, w2, _, w3, _ = parameters
    scale = 2.0 / error.size
    output_gradient = scale * error
    grad_w3 = hidden2.T @ output_gradient + weight_decay * w3
    grad_b3 = output_gradient.sum(axis=0)
    hidden2_gradient = (output_gradient @ w3.T) * (1.0 - hidden2**2)
    grad_w2 = hidden1.T @ hidden2_gradient + weight_decay * w2
    grad_b2 = hidden2_gradient.sum(axis=0)
    hidden1_gradient = (hidden2_gradient @ w2.T) * (1.0 - hidden1**2)
    grad_w1 = inputs.T @ hidden1_gradient + weight_decay * w1
    grad_b1 = hidden1_gradient.sum(axis=0)
    return [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]


def _adam_step(
    parameters: list[np.ndarray],
    gradients: list[np.ndarray],
    first_moments: list[np.ndarray],
    second_moments: list[np.ndarray],
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> None:
    total_norm = np.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients))
    clip_scale = min(1.0, 5.0 / max(total_norm, 1e-12))
    for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
        gradient = gradient * clip_scale
        first_moments[index] *= beta1
        first_moments[index] += (1.0 - beta1) * gradient
        second_moments[index] *= beta2
        second_moments[index] += (1.0 - beta2) * gradient**2
        corrected_first = first_moments[index] / (1.0 - beta1**step)
        corrected_second = second_moments[index] / (1.0 - beta2**step)
        parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [array, np.ones((*array.shape[:-1], 1), dtype=array.dtype)],
        axis=-1,
    )


def _ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    regularizer = ridge * np.eye(design.shape[-1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ target,
    ).astype(np.float32)
