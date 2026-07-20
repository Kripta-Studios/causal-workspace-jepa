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


def conditional_resample_subspace(
    activations: np.ndarray,
    donor_bank: np.ndarray,
    basis: np.ndarray,
    *,
    neighbor_pool: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Patch basis coordinates from complement-matched in-manifold donors.

    For each recipient, donors are ranked by distance outside the candidate
    subspace. Among the closest ``neighbor_pool`` donors, the donor with the
    largest candidate-coordinate difference is selected. This creates a
    nontrivial interchange intervention while preserving the recipient's
    complement as closely as the empirical activation bank permits.
    """

    recipients = np.asarray(activations, dtype=np.float64)
    donors = np.asarray(donor_bank, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    if recipients.ndim != 2 or donors.ndim != 2:
        raise ValueError("activations and donor_bank must be rank-two arrays")
    if recipients.shape[1] != donors.shape[1] or basis_array.shape[0] != recipients.shape[1]:
        raise ValueError("activation and basis dimensions must match")
    if not 1 <= neighbor_pool <= len(donors):
        raise ValueError("neighbor_pool must be between one and the donor count")
    projector = basis_array @ basis_array.T
    complement = np.eye(recipients.shape[1], dtype=np.float64) - projector
    patched = np.empty_like(recipients)
    donor_ids = np.empty(len(recipients), dtype=np.int64)
    for index, recipient in enumerate(recipients):
        differences = donors - recipient
        complement_distance = np.sum((differences @ complement) ** 2, axis=1)
        pool = np.argpartition(complement_distance, neighbor_pool - 1)[:neighbor_pool]
        candidate_distance = np.sum((differences[pool] @ basis_array) ** 2, axis=1)
        donor_id = int(pool[int(np.argmax(candidate_distance))])
        donor_ids[index] = donor_id
        patched[index] = recipient + (differences[donor_id] @ basis_array) @ basis_array.T
    return patched.astype(np.float32), donor_ids


def activation_manifold_diagnostics(
    clean: np.ndarray,
    intervened: np.ndarray,
    donor_bank: np.ndarray,
) -> dict[str, float]:
    """Compare perturbation size and nearest-bank distance with clean inputs."""

    clean_array = np.asarray(clean, dtype=np.float64)
    intervened_array = np.asarray(intervened, dtype=np.float64)
    bank = np.asarray(donor_bank, dtype=np.float64)
    clean_distance = _nearest_distances(clean_array, bank)
    intervened_distance = _nearest_distances(intervened_array, bank)
    ratios = intervened_distance / np.maximum(clean_distance, 1e-8)
    activation_variance = float(np.mean((bank - bank.mean(axis=0, keepdims=True)) ** 2))
    perturbation_mse = float(np.mean((intervened_array - clean_array) ** 2))
    return {
        "median_density_distance_ratio": float(np.median(ratios)),
        "p95_density_distance_ratio": float(np.quantile(ratios, 0.95)),
        "perturbation_rms": float(np.sqrt(perturbation_mse)),
        "normalized_perturbation": float(perturbation_mse / max(activation_variance, 1e-12)),
        "clean_median_nearest_distance": float(np.median(clean_distance)),
        "intervened_median_nearest_distance": float(np.median(intervened_distance)),
    }


def conditional_resample_causal_audit(
    activations: np.ndarray,
    donor_bank: np.ndarray,
    consumers: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    candidate_basis: np.ndarray,
    control_bases: Sequence[np.ndarray],
    *,
    pca_basis: np.ndarray | None = None,
    neighbor_pool: int = 16,
    max_density_ratio: float = 3.0,
    match_factor: float = 2.0,
    min_matched_controls: int = 8,
) -> dict[str, object]:
    """Audit a candidate with conditional donor and matched control patches."""

    if match_factor < 1.0:
        raise ValueError("match_factor must be at least one")
    clean_outputs = {
        name: np.asarray(consumer(activations)) for name, consumer in consumers.items()
    }

    def evaluate(basis: np.ndarray) -> dict[str, object]:
        patched, donor_ids = conditional_resample_subspace(
            activations,
            donor_bank,
            basis,
            neighbor_pool=neighbor_pool,
        )
        damage = {
            name: normalized_output_damage(output, consumers[name](patched))
            for name, output in clean_outputs.items()
        }
        diagnostics = activation_manifold_diagnostics(activations, patched, donor_bank)
        return {
            "damage_by_consumer": damage,
            "mean_damage": float(np.mean(list(damage.values()))),
            "manifold": diagnostics,
            "unique_donor_fraction": float(len(np.unique(donor_ids)) / len(donor_ids)),
        }

    candidate = evaluate(candidate_basis)
    controls = [evaluate(basis) for basis in control_bases]
    candidate_manifold = candidate["manifold"]
    assert isinstance(candidate_manifold, dict)
    candidate_density = float(candidate_manifold["median_density_distance_ratio"])
    candidate_perturbation = float(candidate_manifold["perturbation_rms"])
    lower_density = candidate_density / match_factor
    upper_density = candidate_density * match_factor
    lower_perturbation = candidate_perturbation / match_factor
    upper_perturbation = candidate_perturbation * match_factor
    matched_controls = []
    matched_control_indices = []
    for control_index, control in enumerate(controls):
        manifold = control["manifold"]
        density = float(manifold["median_density_distance_ratio"])  # type: ignore[index]
        perturbation = float(manifold["perturbation_rms"])  # type: ignore[index]
        if (
            density <= max_density_ratio
            and lower_density <= density <= upper_density
            and lower_perturbation <= perturbation <= upper_perturbation
        ):
            matched_controls.append(control)
            matched_control_indices.append(control_index)
    matched_damage = np.asarray(
        [float(control["mean_damage"]) for control in matched_controls],
        dtype=np.float64,
    )
    candidate_damage = float(candidate["mean_damage"])
    candidate_density_valid = bool(candidate_density <= max_density_ratio)
    enough_controls = len(matched_controls) >= min_matched_controls
    control_p95 = float(np.quantile(matched_damage, 0.95)) if matched_damage.size else None
    pca = evaluate(pca_basis) if pca_basis is not None else None
    pca_matched = False
    if pca is not None:
        manifold = pca["manifold"]
        density = float(manifold["median_density_distance_ratio"])  # type: ignore[index]
        perturbation = float(manifold["perturbation_rms"])  # type: ignore[index]
        pca_matched = bool(
            density <= max_density_ratio
            and lower_density <= density <= upper_density
            and lower_perturbation <= perturbation <= upper_perturbation
        )
    return {
        "candidate": candidate,
        "controls_total": len(controls),
        "controls": controls,
        "matched_control_count": len(matched_controls),
        "matched_control_indices": matched_control_indices,
        "matched_control_mean_damage": (
            float(matched_damage.mean()) if matched_damage.size else None
        ),
        "matched_control_p95_damage": control_p95,
        "candidate_density_valid": candidate_density_valid,
        "enough_matched_controls": enough_controls,
        "candidate_exceeds_matched_control_p95": bool(
            enough_controls and control_p95 is not None and candidate_damage > control_p95
        ),
        "pca_control": pca,
        "pca_control_matched": pca_matched,
        "candidate_exceeds_pca": bool(
            pca_matched and pca is not None and candidate_damage > float(pca["mean_damage"])
        ),
        "thresholds": {
            "max_density_ratio": max_density_ratio,
            "match_factor": match_factor,
            "min_matched_controls": min_matched_controls,
        },
    }


def selective_necessity_ratio(
    multistep_damage: float,
    one_step_damage: float,
    epsilon: float = 1e-12,
) -> float:
    return float(multistep_damage / (one_step_damage + epsilon))


def _nearest_distances(samples: np.ndarray, bank: np.ndarray) -> np.ndarray:
    distances = np.empty(len(samples), dtype=np.float64)
    for index, sample in enumerate(samples):
        differences = bank - sample
        distances[index] = np.sqrt(float(np.min(np.sum(differences**2, axis=1))))
    return distances
