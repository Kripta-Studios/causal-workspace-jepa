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
    intervention = intervention_features.reshape(intervention_features.shape[0], -1).astype(np.float64)
    magnitude = intervention[:, -1:]
    interactions = context_flat * magnitude
    return _with_bias(np.concatenate([context_flat, intervention, interactions], axis=-1))


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, np.ones((array.shape[0], 1), dtype=array.dtype)], axis=-1)
