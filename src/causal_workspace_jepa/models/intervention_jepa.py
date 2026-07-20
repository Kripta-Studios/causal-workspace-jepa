"""Intervention-conditioned JEPA-style predictors for CPU smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerTransitionInterventionJEPA:
    weights: np.ndarray
    target_mean: np.ndarray

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        intervention_features: np.ndarray,
        targets: np.ndarray,
        *,
        ridge: float = 1e-4,
    ) -> "LayerTransitionInterventionJEPA":
        design = _design(context, intervention_features)
        target = targets.reshape(targets.shape[0], -1).astype(np.float64)
        xtx = design.T @ design
        weights = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), design.T @ target)
        return cls(weights=weights.astype(np.float32), target_mean=target.mean(axis=0).astype(np.float32))

    def predict(self, context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
        return (_design(context, intervention_features) @ self.weights).astype(np.float32)

    def mse(self, context: np.ndarray, intervention_features: np.ndarray, targets: np.ndarray) -> float:
        target = targets.reshape(targets.shape[0], -1)
        return float(np.mean((self.predict(context, intervention_features) - target) ** 2))

    def mean_baseline_mse(self, targets: np.ndarray) -> float:
        target = targets.reshape(targets.shape[0], -1)
        prediction = np.broadcast_to(self.target_mean, target.shape)
        return float(np.mean((prediction - target) ** 2))


@dataclass
class LinearContextBaseline:
    weights: np.ndarray

    @classmethod
    def fit(cls, context: np.ndarray, targets: np.ndarray, ridge: float = 1e-4) -> "LinearContextBaseline":
        design = _with_bias(context.reshape(context.shape[0], -1).astype(np.float64))
        target = targets.reshape(targets.shape[0], -1).astype(np.float64)
        xtx = design.T @ design
        weights = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), design.T @ target)
        return cls(weights=weights.astype(np.float32))

    def predict(self, context: np.ndarray) -> np.ndarray:
        design = _with_bias(context.reshape(context.shape[0], -1).astype(np.float64))
        return (design @ self.weights).astype(np.float32)

    def mse(self, context: np.ndarray, targets: np.ndarray) -> float:
        target = targets.reshape(targets.shape[0], -1)
        return float(np.mean((self.predict(context) - target) ** 2))


@dataclass
class RidgeInterventionPredictor:
    """Standardized linear or bilinear intervention-effect predictor."""

    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    mode: str

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        intervention_features: np.ndarray,
        targets: np.ndarray,
        *,
        mode: str,
        ridge: float = 1e-3,
    ) -> "RidgeInterventionPredictor":
        features = _intervention_design(context, intervention_features, mode=mode)
        target = targets.reshape(targets.shape[0], -1).astype(np.float64)
        feature_mean = features.mean(axis=0)
        feature_scale = features.std(axis=0)
        feature_scale[feature_scale < 1e-6] = 1.0
        target_mean = target.mean(axis=0)
        target_scale = target.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
        design = _with_bias((features - feature_mean) / feature_scale)
        normalized_target = (target - target_mean) / target_scale
        regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ normalized_target,
        )
        return cls(
            weights=weights.astype(np.float32),
            feature_mean=feature_mean.astype(np.float32),
            feature_scale=feature_scale.astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            target_scale=target_scale.astype(np.float32),
            mode=mode,
        )

    def predict(self, context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
        features = _intervention_design(context, intervention_features, mode=self.mode)
        design = _with_bias((features - self.feature_mean) / self.feature_scale)
        normalized = design @ self.weights
        return (normalized * self.target_scale + self.target_mean).astype(np.float32)


@dataclass
class TinyMLPInterventionPredictor:
    """A trained one-hidden-layer CPU MLP for intervention effects."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        intervention_features: np.ndarray,
        targets: np.ndarray,
        *,
        hidden_dim: int = 64,
        steps: int = 400,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> "TinyMLPInterventionPredictor":
        import torch

        inputs = np.concatenate(
            [context.reshape(context.shape[0], -1), intervention_features],
            axis=1,
        ).astype(np.float32)
        target = targets.reshape(targets.shape[0], -1).astype(np.float32)
        input_mean = inputs.mean(axis=0)
        input_scale = inputs.std(axis=0)
        input_scale[input_scale < 1e-6] = 1.0
        target_mean = target.mean(axis=0)
        target_scale = target.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
        x = torch.from_numpy((inputs - input_mean) / input_scale)
        y = torch.from_numpy((target - target_mean) / target_scale)
        torch.manual_seed(seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, y.shape[1]),
        )
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((network(x) - y) ** 2)
            loss.backward()
            optimizer.step()
        first = network[0]
        second = network[2]
        return cls(
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            input_weights=first.weight.detach().cpu().numpy().T,
            input_bias=first.bias.detach().cpu().numpy(),
            output_weights=second.weight.detach().cpu().numpy().T,
            output_bias=second.bias.detach().cpu().numpy(),
        )

    def predict(self, context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
        inputs = np.concatenate(
            [context.reshape(context.shape[0], -1), intervention_features],
            axis=1,
        ).astype(np.float32)
        normalized = (inputs - self.input_mean) / self.input_scale
        hidden = np.tanh(normalized @ self.input_weights + self.input_bias)
        output = hidden @ self.output_weights + self.output_bias
        return (output * self.target_scale + self.target_mean).astype(np.float32)


def no_change_mse(targets: np.ndarray) -> float:
    target = targets.reshape(targets.shape[0], -1)
    return float(np.mean(target**2))


def effect_correlation(predicted: np.ndarray, observed: np.ndarray) -> float:
    pred = predicted.reshape(-1)
    obs = observed.reshape(-1)
    if float(np.std(pred)) <= 1e-12 or float(np.std(obs)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, obs)[0, 1])


def _design(context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
    context_flat = context.reshape(context.shape[0], -1).astype(np.float64)
    intervention = intervention_features.reshape(intervention_features.shape[0], -1).astype(
        np.float64
    )
    magnitude = intervention[:, -1:]
    interactions = context_flat * magnitude
    return _with_bias(np.concatenate([context_flat, intervention, interactions], axis=-1))


def _intervention_design(
    context: np.ndarray,
    intervention_features: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    context_flat = context.reshape(context.shape[0], -1).astype(np.float64)
    intervention = intervention_features.reshape(intervention_features.shape[0], -1).astype(
        np.float64
    )
    if mode == "linear":
        return np.concatenate([context_flat, intervention], axis=1)
    if mode == "bilinear":
        interactions = np.einsum("ni,nj->nij", context_flat, intervention).reshape(
            context_flat.shape[0], -1
        )
        return np.concatenate([context_flat, intervention, interactions], axis=1)
    raise ValueError(f"unknown intervention predictor mode: {mode}")


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, np.ones((array.shape[0], 1), dtype=array.dtype)], axis=-1)
