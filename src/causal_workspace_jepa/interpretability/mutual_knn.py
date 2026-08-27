"""Mutual k-nearest-neighbour overlap for paired latent geometries.

This is a neighborhood-overlap score, not a causal-circuit metric.
"""

from __future__ import annotations

import numpy as np


def pairwise_squared_distances(points: np.ndarray) -> np.ndarray:
    packed = np.asarray(points, dtype=np.float64)
    if packed.ndim != 2:
        raise ValueError("points must have shape [n, dim]")
    grams = packed @ packed.T
    squared_norms = np.diag(grams)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * grams
    np.fill_diagonal(distances, np.inf)
    return np.maximum(distances, 0.0)


def knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1")
    distances = pairwise_squared_distances(points)
    if points.shape[0] <= k:
        raise ValueError("need more points than k")
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


def mutual_knn(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    """Mean fraction of shared k-NN indices for paired rows."""

    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        raise ValueError("paired geometries must share shape")
    left_nn = knn_indices(left_arr, k)
    right_nn = knn_indices(right_arr, k)
    scores = [
        len(set(left_nn[index].tolist()) & set(right_nn[index].tolist())) / float(k)
        for index in range(left_arr.shape[0])
    ]
    return float(np.mean(scores))


def chance_reference(*, n_eval: int, k: int) -> float:
    return float(k) / float(n_eval - 1)
