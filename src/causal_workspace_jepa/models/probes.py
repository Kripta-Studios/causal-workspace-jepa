"""Model-level probes and small frozen-representation readout heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal_workspace_jepa.interpretability.probing import RidgeProbe, random_label_control


@dataclass
class QuadraticRidgeHead:
    """A small nonlinear consumer trained on a frozen representation."""

    weights: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray
    output_shape: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        ridge: float = 1e-3,
    ) -> "QuadraticRidgeHead":
        x = np.asarray(inputs, dtype=np.float64).reshape(inputs.shape[0], -1)
        y = np.asarray(targets, dtype=np.float64).reshape(targets.shape[0], -1)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        design = _quadratic_features((x - mean) / scale)
        regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return cls(
            weights=weights.astype(np.float32),
            input_mean=mean.astype(np.float32),
            input_scale=scale.astype(np.float32),
            output_shape=tuple(targets.shape[1:]),
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32).reshape(inputs.shape[0], -1)
        normalized = (x - self.input_mean) / self.input_scale
        output = _quadratic_features(normalized.astype(np.float64)) @ self.weights
        return output.astype(np.float32).reshape((inputs.shape[0], *self.output_shape))

    def score_r2(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        target = np.asarray(targets, dtype=np.float64)
        residual = float(np.mean((self.predict(inputs) - target) ** 2))
        baseline = float(np.mean((target - target.mean(axis=0, keepdims=True)) ** 2))
        return float(1.0 - residual / max(baseline, 1e-12))

    def jacobian(self, inputs: np.ndarray, *, epsilon: float = 1e-3) -> np.ndarray:
        """Compute batched finite-difference Jacobians ``[sample, output, input]``."""

        x = np.asarray(inputs, dtype=np.float32).reshape(inputs.shape[0], -1)
        output_size = int(np.prod(self.output_shape)) if self.output_shape else 1
        jacobian = np.empty((x.shape[0], output_size, x.shape[1]), dtype=np.float32)
        for feature in range(x.shape[1]):
            plus = x.copy()
            minus = x.copy()
            plus[:, feature] += epsilon
            minus[:, feature] -= epsilon
            delta = (self.predict(plus) - self.predict(minus)) / (2.0 * epsilon)
            jacobian[:, :, feature] = delta.reshape(x.shape[0], output_size)
        return jacobian


def _quadratic_features(inputs: np.ndarray) -> np.ndarray:
    columns = [inputs]
    for left in range(inputs.shape[1]):
        columns.append(inputs[:, left : left + 1] * inputs[:, left:])
    columns.append(np.ones((inputs.shape[0], 1), dtype=inputs.dtype))
    return np.concatenate(columns, axis=1)


__all__ = ["QuadraticRidgeHead", "RidgeProbe", "random_label_control"]
