"""Workspace-candidate discovery and matched causal controls.

The helpers in this module identify a compact subspace that carries local
sensitivity for several downstream consumers.  That is necessary but not
sufficient evidence for a workspace: callers must separately test
controllability, direct causal mediation, selective necessity, and
generalization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharedSubspaceCandidate:
    """A compact subspace proposed from normalized consumer Jacobians."""

    basis: np.ndarray
    dimension: int
    representation_dim: int
    consumer_capture: Mapping[str, float]
    min_capture: float
    mean_capture: float
    compactness_ratio: float
    sensitivity_candidate_found: bool

    def to_metrics(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "representation_dim": self.representation_dim,
            "consumer_capture": dict(self.consumer_capture),
            "min_capture": self.min_capture,
            "mean_capture": self.mean_capture,
            "compactness_ratio": self.compactness_ratio,
            "sensitivity_candidate_found": self.sensitivity_candidate_found,
        }


def discover_shared_subspace(
    consumer_jacobians: Mapping[str, np.ndarray],
    *,
    max_dimension: int,
    min_consumer_capture: float = 0.75,
    max_compactness_ratio: float = 0.5,
    min_consumers: int = 3,
) -> SharedSubspaceCandidate:
    """Find the smallest subspace capturing each consumer's sensitivity.

    Each Jacobian must use its last axis for the candidate representation.
    Consumer Gram matrices are trace-normalized so a high-norm consumer cannot
    dominate discovery merely because of output scale.
    """

    if len(consumer_jacobians) < min_consumers:
        raise ValueError(f"at least {min_consumers} consumers are required")
    if not 0.0 < min_consumer_capture <= 1.0:
        raise ValueError("min_consumer_capture must be in (0, 1]")

    grams: dict[str, np.ndarray] = {}
    representation_dim: int | None = None
    for name, jacobian in consumer_jacobians.items():
        array = np.asarray(jacobian, dtype=np.float64)
        if array.ndim < 2:
            raise ValueError(f"consumer {name!r} Jacobian must have at least two dimensions")
        if representation_dim is None:
            representation_dim = int(array.shape[-1])
        elif array.shape[-1] != representation_dim:
            raise ValueError("all consumer Jacobians must share a representation dimension")
        rows = array.reshape(-1, array.shape[-1])
        gram = rows.T @ rows
        trace = float(np.trace(gram))
        if trace <= 1e-12:
            raise ValueError(f"consumer {name!r} has zero Jacobian sensitivity")
        grams[name] = gram / trace

    assert representation_dim is not None
    if max_dimension < 1 or max_dimension > representation_dim:
        raise ValueError("max_dimension must be between 1 and the representation dimension")
    mean_gram = np.mean(np.stack(list(grams.values()), axis=0), axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(mean_gram)
    order = np.argsort(eigenvalues)[::-1]
    ordered_vectors = eigenvectors[:, order]

    selected_dimension = max_dimension
    selected_capture: dict[str, float] = {}
    threshold_reached = False
    for dimension in range(1, max_dimension + 1):
        basis = ordered_vectors[:, :dimension]
        projector = basis @ basis.T
        capture = {name: float(np.trace(projector @ gram)) for name, gram in grams.items()}
        selected_dimension = dimension
        selected_capture = capture
        if min(capture.values()) >= min_consumer_capture:
            threshold_reached = True
            break

    basis = ordered_vectors[:, :selected_dimension].astype(np.float32)
    compactness = selected_dimension / representation_dim
    found = threshold_reached and compactness <= max_compactness_ratio
    return SharedSubspaceCandidate(
        basis=basis,
        dimension=selected_dimension,
        representation_dim=representation_dim,
        consumer_capture=selected_capture,
        min_capture=float(min(selected_capture.values())),
        mean_capture=float(np.mean(list(selected_capture.values()))),
        compactness_ratio=float(compactness),
        sensitivity_candidate_found=bool(found),
    )


def principal_subspace(activations: np.ndarray, dimension: int) -> np.ndarray:
    """Return a high-variance PCA control basis with vectors in columns."""

    array = np.asarray(activations, dtype=np.float64)
    flat = array.reshape(-1, array.shape[-1])
    _, _, right = np.linalg.svd(flat - flat.mean(axis=0, keepdims=True), full_matrices=False)
    return right[:dimension].T.astype(np.float32)


def random_subspace(representation_dim: int, dimension: int, seed: int) -> np.ndarray:
    """Sample a deterministic Haar-like orthonormal control basis."""

    if dimension > representation_dim:
        raise ValueError("dimension cannot exceed representation_dim")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(representation_dim, dimension))
    basis, _ = np.linalg.qr(matrix)
    return basis[:, :dimension].astype(np.float32)


def project_out_subspace(
    activations: np.ndarray,
    basis: np.ndarray,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Remove a column-basis subspace while preserving the chosen center."""

    array = np.asarray(activations)
    basis_array = np.asarray(basis, dtype=array.dtype)
    if basis_array.ndim != 2 or basis_array.shape[0] != array.shape[-1]:
        raise ValueError("basis must have shape [representation_dim, subspace_dim]")
    origin = (
        np.zeros(array.shape[-1], dtype=array.dtype)
        if center is None
        else np.asarray(center, dtype=array.dtype)
    )
    centered = array - origin
    return array - (centered @ basis_array) @ basis_array.T


def normalized_output_damage(clean: np.ndarray, intervened: np.ndarray) -> float:
    """Return intervention MSE normalized by clean output variance."""

    clean_array = np.asarray(clean, dtype=np.float64)
    intervened_array = np.asarray(intervened, dtype=np.float64)
    scale = float(np.mean((clean_array - clean_array.mean(axis=0, keepdims=True)) ** 2))
    return float(np.mean((intervened_array - clean_array) ** 2) / max(scale, 1e-12))


def matched_causal_subspace_audit(
    activations: np.ndarray,
    consumers: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    candidate_basis: np.ndarray,
    control_bases: Sequence[np.ndarray],
    *,
    center: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare direct candidate ablation with equal-dimensional controls."""

    clean = {name: np.asarray(consumer(activations)) for name, consumer in consumers.items()}
    candidate_activations = project_out_subspace(activations, candidate_basis, center=center)
    candidate_damage = {
        name: normalized_output_damage(output, consumers[name](candidate_activations))
        for name, output in clean.items()
    }
    control_means: list[float] = []
    for basis in control_bases:
        control_activations = project_out_subspace(activations, basis, center=center)
        damages = [
            normalized_output_damage(output, consumers[name](control_activations))
            for name, output in clean.items()
        ]
        control_means.append(float(np.mean(damages)))
    candidate_mean = float(np.mean(list(candidate_damage.values())))
    control_array = np.asarray(control_means, dtype=np.float64)
    return {
        "candidate_damage_by_consumer": candidate_damage,
        "candidate_mean_damage": candidate_mean,
        "random_control_mean_damage": float(control_array.mean()),
        "random_control_std_damage": float(control_array.std()),
        "random_control_max_damage": float(control_array.max()),
        "specificity_margin_vs_random_mean": float(candidate_mean - control_array.mean()),
        "specificity_exceeds_all_random_controls": bool(candidate_mean > control_array.max()),
        "control_count": len(control_bases),
    }


def selective_necessity_ratio(
    multistep_damage: float,
    one_step_damage: float,
    epsilon: float = 1e-12,
) -> float:
    return float(multistep_damage / (one_step_damage + epsilon))
