"""Replayable NumPy intervention operators."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from causal_workspace_jepa.common.types import InterventionSpec


def spec_to_json(spec: InterventionSpec) -> str:
    return json.dumps(spec.to_dict(), sort_keys=True)


def spec_from_json(payload: str) -> InterventionSpec:
    return InterventionSpec.from_dict(json.loads(payload))


def apply_intervention(
    activation: np.ndarray,
    spec: InterventionSpec,
    *,
    donor: np.ndarray | None = None,
    mean: np.ndarray | float | None = None,
    basis: np.ndarray | None = None,
    feature_values: np.ndarray | float | None = None,
) -> np.ndarray:
    """Apply an intervention to a copy of ``activation``.

    Positions index axis 1 for sequence-like tensors with at least three
    dimensions. Feature ids always index the final axis.
    """

    result = np.array(activation, copy=True)
    target = _extract(result, spec)
    if spec.operation in {"zero", "suppress_module"}:
        updated = np.zeros_like(target)
    elif spec.operation == "mean":
        replacement = mean if mean is not None else np.mean(activation, axis=0, keepdims=True)
        updated = _extract(np.broadcast_to(replacement, result.shape), spec)
    elif spec.operation == "resample":
        rng = np.random.default_rng(spec.seed)
        donor_values = np.array(activation, copy=True)
        rng.shuffle(donor_values, axis=0)
        updated = _extract(donor_values, spec)
    elif spec.operation in {"patch", "replace_feature"}:
        if donor is None and feature_values is None:
            raise ValueError(f"{spec.operation} requires donor or feature_values")
        if donor is not None:
            updated = _extract(np.asarray(donor), spec)
        else:
            updated = np.broadcast_to(feature_values, target.shape)
    elif spec.operation == "steer":
        updated = target + spec.magnitude
    elif spec.operation == "scale":
        updated = target * spec.magnitude
    elif spec.operation == "project_out":
        if basis is None:
            raise ValueError("project_out requires basis")
        updated = project_out_numpy(
            target, np.asarray(basis, dtype=target.dtype), spec.magnitude
        )
    else:
        raise ValueError(f"unsupported intervention operation: {spec.operation}")
    _assign(result, spec, updated)
    return result


def matched_random_feature_control(spec: InterventionSpec, hidden_size: int, seed: int) -> InterventionSpec:
    if spec.feature_ids is None:
        return spec
    rng = np.random.default_rng(seed)
    replacement = tuple(int(v) for v in rng.choice(hidden_size, size=len(spec.feature_ids), replace=False))
    return InterventionSpec(
        site=spec.site,
        operation=spec.operation,
        positions=spec.positions,
        feature_ids=replacement,
        magnitude=spec.magnitude,
        donor_example_id=spec.donor_example_id,
        seed=seed,
    )


def _extract(array: np.ndarray, spec: InterventionSpec) -> np.ndarray:
    if array.ndim == 3:
        positions = _positions(array, spec)
        features = _features(array, spec)
        return array[:, positions, :][..., features]
    if array.ndim == 2:
        features = _features(array, spec)
        return array[:, features]
    if spec.feature_ids is not None:
        return array[..., list(spec.feature_ids)]
    return array


def _assign(array: np.ndarray, spec: InterventionSpec, value: np.ndarray) -> None:
    if array.ndim == 3:
        positions = _positions(array, spec)
        features = _features(array, spec)
        view = array[:, positions, :]
        view[..., features] = value
        array[:, positions, :] = view
        return
    if array.ndim == 2:
        features = _features(array, spec)
        array[:, features] = value
        return
    if spec.feature_ids is not None:
        array[..., list(spec.feature_ids)] = value
    else:
        array[...] = value


def _positions(array: np.ndarray, spec: InterventionSpec) -> list[int]:
    if spec.positions is None:
        return list(range(array.shape[1]))
    return list(spec.positions)


def _features(array: np.ndarray, spec: InterventionSpec) -> list[int]:
    if spec.feature_ids is None:
        return list(range(array.shape[-1]))
    return list(spec.feature_ids)


def project_out_numpy(
    target: np.ndarray, basis: np.ndarray, magnitude: float = 1.0
) -> np.ndarray:
    """Remove a column-basis span from a NumPy activation.

    Two-dimensional bases use the repository-wide
    ``[representation_dim, subspace_dim]`` contract. A one-dimensional vector
    remains supported as an unambiguous single-column convenience. SVD turns
    non-orthogonal or redundant coordinates into an orthonormal span, so the
    intervention is invariant to invertible changes of basis within that span.
    """

    if not math.isfinite(magnitude):
        raise ValueError("project_out magnitude must be finite")
    target_array = np.asarray(target)
    if not np.issubdtype(target_array.dtype, np.floating):
        raise TypeError("project_out target must have a floating dtype")
    flat = target_array.reshape(-1, target.shape[-1])
    matrix = np.asarray(basis)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != flat.shape[-1] or matrix.shape[1] == 0:
        raise ValueError(
            "basis must have shape [representation_dim, subspace_dim] with a nonempty subspace"
        )
    if not np.all(np.isfinite(flat)) or not np.all(np.isfinite(matrix)):
        raise ValueError("project_out target and basis must be finite")

    work = flat.astype(np.float64, copy=False)
    basis_work = matrix.astype(np.float64, copy=False)
    left, singular_values, _ = np.linalg.svd(basis_work, full_matrices=False)
    input_dtype = matrix.dtype if np.issubdtype(matrix.dtype, np.floating) else np.dtype(np.float64)
    tolerance = (
        np.finfo(input_dtype).eps
        * max(basis_work.shape)
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == 0:
        return np.array(target_array, copy=True)
    orthonormal = left[:, :rank]
    projected = work - magnitude * (work @ orthonormal) @ orthonormal.T
    return projected.astype(target_array.dtype, copy=False).reshape(target_array.shape)


def project_out_torch(target: Any, basis: Any, magnitude: float) -> Any:
    """Torch equivalent of :func:`project_out_numpy` under the column-basis contract."""

    import torch

    if not math.isfinite(magnitude):
        raise ValueError("project_out magnitude must be finite")
    if not torch.is_tensor(target) or not torch.is_tensor(basis):
        raise TypeError("project_out_torch expects torch tensors")
    if not target.is_floating_point() or not basis.is_floating_point():
        raise TypeError("project_out target and basis must have floating dtypes")
    flat = target.reshape(-1, target.shape[-1])
    matrix = basis
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != flat.shape[-1] or matrix.shape[1] == 0:
        raise ValueError(
            "basis must have shape [representation_dim, subspace_dim] with a nonempty subspace"
        )
    if not bool(torch.isfinite(flat).all().item()) or not bool(torch.isfinite(matrix).all().item()):
        raise ValueError("project_out target and basis must be finite")

    work_dtype = torch.float64 if flat.dtype == torch.float64 else torch.float32
    work = flat.to(dtype=work_dtype)
    basis_work = matrix.to(device=flat.device, dtype=work_dtype)
    left, singular_values, _ = torch.linalg.svd(basis_work, full_matrices=False)
    tolerance = (
        torch.finfo(work_dtype).eps
        * max(basis_work.shape)
        * singular_values[0]
    )
    rank = int((singular_values > tolerance).sum().item())
    if rank == 0:
        return target.clone()
    orthonormal = left[:, :rank]
    projected = work - magnitude * (work @ orthonormal) @ orthonormal.T
    return projected.to(dtype=target.dtype).reshape_as(target)
