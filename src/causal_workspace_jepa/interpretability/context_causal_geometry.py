"""Context-paired causal geometry with coordinate-gauge and pooling controls."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def causal_coupling(jacobian: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return the output-by-intervention matrix ``J D^T``.

    Rows of ``directions`` are tangent/intervention vectors and rows of ``jacobian`` are output
    covectors. Their contraction is invariant when vectors and covectors are transformed by the
    corresponding primal and dual coordinate maps.
    """

    jacobian = np.asarray(jacobian, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if jacobian.ndim != 2 or directions.ndim != 2:
        raise ValueError("jacobian and directions must both be matrices")
    if jacobian.shape[1] != directions.shape[1]:
        raise ValueError("jacobian and direction coordinates do not match")
    return jacobian @ directions.T


def coupling_spectrum(coupling: np.ndarray) -> dict[str, Any]:
    """Summarize singular energy without assigning semantic labels to modes."""

    singular = np.linalg.svd(np.asarray(coupling, dtype=np.float64), compute_uv=False)
    power = singular**2
    total = float(power.sum())
    probability = power / total if total > 1e-15 else np.zeros_like(power)
    effective_rank = (
        float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-15)))))
        if total > 1e-15
        else 0.0
    )
    return {
        "singular_values": singular.tolist(),
        "frobenius_energy": total,
        "effective_rank": effective_rank,
        "top_mode_energy_fraction": float(probability[0]) if len(probability) else 0.0,
    }


def euclidean_subspace_overlap(
    directions: np.ndarray, jacobian: np.ndarray, *, rank: int
) -> float:
    """Naive Euclidean overlap of top reachable vectors and observable covectors."""

    reachable = _top_row_basis(np.asarray(directions, dtype=np.float64), rank)
    observable = _top_row_basis(np.asarray(jacobian, dtype=np.float64), rank)
    count = min(reachable.shape[1], observable.shape[1])
    if count == 0:
        return 0.0
    singular = np.linalg.svd(reachable.T @ observable, compute_uv=False)
    return float(np.sum(singular[:count] ** 2) / count)


def pooled_euclidean_overlap(
    direction_sets: Sequence[np.ndarray],
    jacobians: Sequence[np.ndarray],
    *,
    rank: int,
) -> float:
    """Pool reachability and observability separately, intentionally discarding context pairing."""

    return euclidean_subspace_overlap(
        np.concatenate(direction_sets, axis=0),
        np.concatenate(jacobians, axis=0),
        rank=rank,
    )


def paired_euclidean_overlaps(
    direction_sets: Sequence[np.ndarray],
    jacobians: Sequence[np.ndarray],
    *,
    rank: int,
) -> np.ndarray:
    if len(direction_sets) != len(jacobians):
        raise ValueError("paired geometry requires the same number of contexts")
    return np.asarray(
        [
            euclidean_subspace_overlap(direction, jacobian, rank=rank)
            for direction, jacobian in zip(direction_sets, jacobians, strict=True)
        ],
        dtype=np.float64,
    )


def transform_vector_covector_pair(
    directions: np.ndarray, jacobian: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``h' = A h`` to vectors and the dual ``J' = J A^-1`` to covectors."""

    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    transformed_directions = np.asarray(directions, dtype=np.float64) @ transform.T
    transformed_jacobian = np.asarray(jacobian, dtype=np.float64) @ inverse
    return transformed_directions, transformed_jacobian


def analytic_pooling_counterexample() -> dict[str, float | bool]:
    """Two contexts where separate pooling invents perfect overlap from zero paired coupling."""

    directions = [np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 1.0]])]
    jacobians = [np.asarray([[0.0, 1.0]]), np.asarray([[1.0, 0.0]])]
    matched = np.asarray(
        [float(causal_coupling(jacobian, direction)[0, 0]) for direction, jacobian in zip(directions, jacobians, strict=True)]
    )
    pooled = pooled_euclidean_overlap(directions, jacobians, rank=2)
    swapped = np.asarray(
        [
            float(causal_coupling(jacobians[1 - index], directions[index])[0, 0])
            for index in range(2)
        ]
    )
    return {
        "matched_max_abs_coupling": float(np.abs(matched).max()),
        "pooled_overlap": pooled,
        "swapped_min_abs_coupling": float(np.abs(swapped).min()),
        "passed": bool(
            np.abs(matched).max() == 0.0
            and abs(pooled - 1.0) <= 1e-12
            and np.abs(swapped).min() == 1.0
        ),
    }


def _top_row_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        raise ValueError("rank must be positive")
    if matrix.size == 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float64)
    _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular):
        return np.zeros((matrix.shape[1], 0), dtype=np.float64)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    count = min(rank, int(np.sum(singular > tolerance)))
    return vectors[:count].T


__all__ = [
    "analytic_pooling_counterexample",
    "causal_coupling",
    "coupling_spectrum",
    "euclidean_subspace_overlap",
    "paired_euclidean_overlaps",
    "pooled_euclidean_overlap",
    "transform_vector_covector_pair",
]
