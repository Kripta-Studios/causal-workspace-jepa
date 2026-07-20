"""Probe models and controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeProbe:
    weights: np.ndarray
    label_mean: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, ridge: float = 1e-4) -> "RidgeProbe":
        x = _with_bias(np.asarray(features, dtype=np.float64).reshape(features.shape[0], -1))
        y = np.asarray(labels, dtype=np.float64).reshape(labels.shape[0], -1)
        xtx = x.T @ x
        weights = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), x.T @ y)
        return cls(weights=weights.astype(np.float32), label_mean=y.mean(axis=0).astype(np.float32))

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = _with_bias(np.asarray(features, dtype=np.float64).reshape(features.shape[0], -1))
        return (x @ self.weights).astype(np.float32)

    def score_mse(self, features: np.ndarray, labels: np.ndarray) -> float:
        target = np.asarray(labels).reshape(labels.shape[0], -1)
        return float(np.mean((self.predict(features) - target) ** 2))

    def mean_baseline_mse(self, labels: np.ndarray) -> float:
        target = np.asarray(labels).reshape(labels.shape[0], -1)
        prediction = np.broadcast_to(self.label_mean, target.shape)
        return float(np.mean((prediction - target) ** 2))


def random_label_control(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.array(labels, copy=True)
    rng.shuffle(shuffled, axis=0)
    return shuffled


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, np.ones((array.shape[0], 1), dtype=array.dtype)], axis=-1)
