"""Replayable NumPy intervention operators."""

from __future__ import annotations

import json

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
        updated = _project_out(target, np.asarray(basis, dtype=target.dtype), spec.magnitude)
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


def _project_out(target: np.ndarray, basis: np.ndarray, magnitude: float) -> np.ndarray:
    flat = target.reshape(-1, target.shape[-1])
    basis_2d = basis.reshape(-1, basis.shape[-1])
    for vector in basis_2d:
        norm = float(np.dot(vector, vector))
        if norm <= 1e-12:
            continue
        projection = (flat @ vector[:, None]) * vector[None, :] / norm
        flat = flat - magnitude * projection
    return flat.reshape(target.shape)
