"""Train/validation-safe differential transports for Causal-Residual JEPA.

The baselines here are deliberately small, typed, and source-model agnostic.  They work on
already observed source/treated tensors and do not authorize any protected model execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from causal_workspace_jepa.models.causal_residual import (
    BaselineClass,
    BaselineEligibilityError,
    CausalEffectBatch,
    ProtectedSplitAccessError,
)


DifferentiableTransition = Callable[[Tensor, Tensor], Tensor]


def _flat(values: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError("effects must have a batch and at least one feature dimension")
    return values.reshape(values.shape[0], -1)


def _tensor_digest(values: Tensor) -> str:
    detached = values.detach().cpu().contiguous()
    raw = detached.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_summary(values: Tensor) -> dict[str, object]:
    return {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "sha256": _tensor_digest(values),
    }


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BaselineMetadata:
    """Declared information contract used by baseline selection."""

    name: str
    classification: BaselineClass
    declared_inputs: tuple[str, ...] = ("source", "intervention")


@dataclass(frozen=True)
class BaselineInputView:
    """Isolated online inputs supplied to arbitrary selector candidates.

    The view deliberately has no ``treated``, ``delta``, ``observed_prefix``, identity, or
    split attributes.  The selector clones tensors before handing them to candidate code so a
    candidate cannot mutate the held validation batch and thereby alter its own score.
    """

    source: Tensor
    intervention: Tensor

    @classmethod
    def from_train_batch(cls, batch: CausalEffectBatch) -> "BaselineInputView":
        batch.require_fit_access()
        return cls(
            source=batch.source.detach().clone(),
            intervention=batch.intervention.detach().clone(),
        )

    @classmethod
    def from_validation_batch(cls, batch: CausalEffectBatch) -> "BaselineInputView":
        batch.require_selection_access()
        return cls(
            source=batch.source.detach().clone(),
            intervention=batch.intervention.detach().clone(),
        )

    @classmethod
    def from_evaluation_batch(
        cls,
        batch: CausalEffectBatch,
        *,
        protected_execution_authorized: bool,
    ) -> "BaselineInputView":
        batch.require_evaluation_access(
            protected_execution_authorized=protected_execution_authorized
        )
        return cls(
            source=batch.source.detach().clone(),
            intervention=batch.intervention.detach().clone(),
        )


class DifferentialBaseline(Protocol):
    """Minimal protocol implemented by every differential transport."""

    metadata: BaselineMetadata

    def fit(self, train_inputs: BaselineInputView) -> "DifferentialBaseline": ...

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor: ...

    def artifact_payload(self) -> Mapping[str, object]: ...


class BaseDifferentialBaseline:
    """Shared split guards and stable artifact summaries."""

    metadata: BaselineMetadata

    def fit(self, train_inputs: BaselineInputView) -> "BaseDifferentialBaseline":
        if not isinstance(train_inputs, BaselineInputView):
            raise TypeError("selector candidates receive BaselineInputView, never outcome batches")
        return self

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def classification(self) -> BaselineClass:
        return self.metadata.classification

    def artifact_payload(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "classification": self.classification.value,
            "declared_inputs": list(self.metadata.declared_inputs),
        }

    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.artifact_payload()).encode("utf-8")).hexdigest()

    def predict_batch(
        self,
        batch: CausalEffectBatch,
        *,
        protected_execution_authorized: bool = False,
    ) -> Tensor:
        inputs = BaselineInputView.from_evaluation_batch(
            batch,
            protected_execution_authorized=protected_execution_authorized,
        )
        return self.predict(inputs.source, inputs.intervention)


class NoChangeBaseline(BaseDifferentialBaseline):
    """Deployable zero-effect residualizer."""

    def __init__(self, *, name: str = "no_change") -> None:
        self.metadata = BaselineMetadata(name, BaselineClass.DEPLOYABLE_RESIDUALIZER)

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if source.shape[0] != intervention.shape[0]:
            raise ValueError("source and intervention batch dimensions must match")
        return torch.zeros_like(source)


class MeanEffectBaseline(BaseDifferentialBaseline):
    """Train-only mean finite-effect baseline."""

    def __init__(self, *, name: str = "mean_effect") -> None:
        self.metadata = BaselineMetadata(name, BaselineClass.DEPLOYABLE_RESIDUALIZER)
        self.mean_effect: Tensor | None = None

    def _fit_train_target(self, target_delta: Tensor) -> None:
        self.mean_effect = target_delta.mean(dim=0, keepdim=True).detach().clone()

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if self.mean_effect is None:
            raise RuntimeError(
                "mean-effect baseline must be fitted on train data before prediction"
            )
        if (
            source.shape[0] != intervention.shape[0]
            or source.shape[1:] != self.mean_effect.shape[1:]
        ):
            raise ValueError("source/intervention shape does not match fitted mean effect")
        return self.mean_effect.expand_as(source)

    def artifact_payload(self) -> Mapping[str, object]:
        if self.mean_effect is None:
            return {**super().artifact_payload(), "fitted": False}
        return {**super().artifact_payload(), "mean_effect": _tensor_summary(self.mean_effect)}


class ExactJVPBaseline(BaseDifferentialBaseline):
    """Exact local first-order transport around the zero intervention."""

    def __init__(self, transition: DifferentiableTransition, *, name: str = "exact_jvp") -> None:
        self.transition = transition
        self.metadata = BaselineMetadata(name, BaselineClass.DEPLOYABLE_RESIDUALIZER)

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if source.shape[0] != intervention.shape[0]:
            raise ValueError("source and intervention batch dimensions must match")
        zero = torch.zeros_like(intervention)

        def transition_from_action(action: Tensor) -> Tensor:
            return self.transition(source, action)

        anchor = transition_from_action(zero) - source
        _, first_order = torch.func.jvp(transition_from_action, (zero,), (intervention,))
        if first_order.shape != source.shape:
            raise ValueError("transition must return a treated state shaped like source")
        return anchor + first_order


class QuadraticHVPBaseline(BaseDifferentialBaseline):
    """Directional HVP/Taylor transport through second order around zero intervention."""

    def __init__(
        self, transition: DifferentiableTransition, *, name: str = "quadratic_hvp"
    ) -> None:
        self.transition = transition
        self.metadata = BaselineMetadata(name, BaselineClass.DEPLOYABLE_RESIDUALIZER)

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if source.shape[0] != intervention.shape[0]:
            raise ValueError("source and intervention batch dimensions must match")
        zero = torch.zeros_like(intervention)

        def transition_from_action(action: Tensor) -> Tensor:
            return self.transition(source, action)

        anchor = transition_from_action(zero) - source
        _, first_order = torch.func.jvp(transition_from_action, (zero,), (intervention,))

        def first_directional_derivative(action: Tensor) -> Tensor:
            return torch.func.jvp(transition_from_action, (action,), (intervention,))[1]

        _, second_order = torch.func.jvp(first_directional_derivative, (zero,), (intervention,))
        if first_order.shape != source.shape or second_order.shape != source.shape:
            raise ValueError("transition must return a treated state shaped like source")
        return anchor + first_order + 0.5 * second_order


class ObservedPrefixRelinearizedBaseline(BaseDifferentialBaseline):
    """Oracle ceiling that requires an observed treated prefix and is never selectable."""

    def __init__(
        self,
        transition: DifferentiableTransition,
        *,
        name: str = "observed_prefix_relinearized_oracle",
    ) -> None:
        self.transition = transition
        self.metadata = BaselineMetadata(
            name,
            BaselineClass.ORACLE_CEILING,
            ("source", "intervention", "observed_treated_prefix"),
        )

    def predict(
        self,
        source: Tensor,
        intervention: Tensor,
        *,
        observed_prefix: Tensor | None = None,
    ) -> Tensor:
        if observed_prefix is None:
            raise ProtectedSplitAccessError(
                "observed-prefix relinearization is an oracle and requires an explicitly supplied "
                "treated prefix"
            )
        if source.shape != observed_prefix.shape or source.shape[0] != intervention.shape[0]:
            raise ValueError(
                "source, observed prefix, and intervention batch dimensions must match"
            )
        zero = torch.zeros_like(intervention)

        def transition_from_action(action: Tensor) -> Tensor:
            return self.transition(observed_prefix, action)

        anchor = transition_from_action(zero) - source
        _, first_order = torch.func.jvp(transition_from_action, (zero,), (intervention,))
        return anchor + first_order

    def predict_batch(
        self,
        batch: CausalEffectBatch,
        *,
        protected_execution_authorized: bool = False,
    ) -> Tensor:
        """Oracle-only batch interface; a prefix must be recorded explicitly on the batch."""

        inputs = BaselineInputView.from_evaluation_batch(
            batch,
            protected_execution_authorized=protected_execution_authorized,
        )
        if batch.observed_prefix is None:
            raise ProtectedSplitAccessError(
                "observed-prefix oracle batch prediction requires batch.observed_prefix"
            )
        return self.predict(
            inputs.source,
            inputs.intervention,
            observed_prefix=batch.observed_prefix.detach().clone(),
        )


@dataclass(frozen=True)
class _RidgeState:
    input_mean: Tensor
    input_scale: Tensor
    weights: Tensor
    output_shape: tuple[int, ...]


def _fit_ridge_state(features: Tensor, target: Tensor, *, ridge: float) -> _RidgeState:
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    features = _flat(features).to(dtype=torch.float64)
    target_shape = tuple(target.shape[1:])
    target = _flat(target).to(dtype=torch.float64)
    input_mean = features.mean(dim=0)
    input_scale = features.std(dim=0, unbiased=False).clamp_min(1e-8)
    normalized = (features - input_mean) / input_scale
    design = torch.cat(
        [
            normalized,
            torch.ones((normalized.shape[0], 1), dtype=normalized.dtype, device=normalized.device),
        ],
        dim=-1,
    )
    regularizer = torch.eye(design.shape[1], dtype=design.dtype, device=design.device) * ridge
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return _RidgeState(
        input_mean=input_mean.detach().clone(),
        input_scale=input_scale.detach().clone(),
        weights=weights.detach().clone(),
        output_shape=target_shape,
    )


def _predict_ridge(state: _RidgeState, features: Tensor, *, dtype: torch.dtype) -> Tensor:
    flat_features = _flat(features).to(dtype=state.weights.dtype)
    if flat_features.shape[-1] != state.input_mean.shape[-1]:
        raise ValueError("feature dimensionality does not match a fitted ridge transport")
    normalized = (flat_features - state.input_mean) / state.input_scale
    design = torch.cat(
        [
            normalized,
            torch.ones((normalized.shape[0], 1), dtype=normalized.dtype, device=normalized.device),
        ],
        dim=-1,
    )
    prediction = design @ state.weights
    return prediction.reshape(prediction.shape[0], *state.output_shape).to(dtype=dtype)


class PopulationDifferentialTransport(BaseDifferentialBaseline):
    """Train-fit population action-to-delta ridge transport."""

    def __init__(
        self,
        *,
        ridge: float = 1e-5,
        name: str = "population_differential",
        classification: BaselineClass = BaselineClass.DEPLOYABLE_RESIDUALIZER,
    ) -> None:
        self.ridge = ridge
        self.metadata = BaselineMetadata(name, classification, ("intervention",))
        self.state: _RidgeState | None = None

    def _fit_train_target(self, train_inputs: BaselineInputView, target_delta: Tensor) -> None:
        self.state = _fit_ridge_state(train_inputs.intervention, target_delta, ridge=self.ridge)

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if self.state is None:
            raise RuntimeError("population differential transport must be fitted on train data")
        if source.shape[0] != intervention.shape[0]:
            raise ValueError("source and intervention batch dimensions must match")
        prediction = _predict_ridge(self.state, intervention, dtype=source.dtype)
        if prediction.shape != source.shape:
            raise ValueError("fitted target state shape does not match source")
        return prediction

    def artifact_payload(self) -> Mapping[str, object]:
        if self.state is None:
            return {**super().artifact_payload(), "fitted": False, "ridge": self.ridge}
        return {
            **super().artifact_payload(),
            "ridge": self.ridge,
            "input_mean": _tensor_summary(self.state.input_mean),
            "input_scale": _tensor_summary(self.state.input_scale),
            "weights": _tensor_summary(self.state.weights),
            "output_shape": list(self.state.output_shape),
        }


class AffineRidgeTransport(BaseDifferentialBaseline):
    """Train-fit affine source/action causal-delta transport."""

    def __init__(
        self,
        *,
        ridge: float = 1e-5,
        name: str = "affine_ridge",
        classification: BaselineClass = BaselineClass.DEPLOYABLE_RESIDUALIZER,
    ) -> None:
        self.ridge = ridge
        self.metadata = BaselineMetadata(name, classification)
        self.state: _RidgeState | None = None

    @staticmethod
    def _features(source: Tensor, intervention: Tensor) -> Tensor:
        if source.shape[0] != intervention.shape[0]:
            raise ValueError("source and intervention batch dimensions must match")
        return torch.cat([_flat(source), _flat(intervention)], dim=-1)

    def _fit_train_target(self, train_inputs: BaselineInputView, target_delta: Tensor) -> None:
        self.state = _fit_ridge_state(
            self._features(train_inputs.source, train_inputs.intervention),
            target_delta,
            ridge=self.ridge,
        )

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if self.state is None:
            raise RuntimeError("affine ridge transport must be fitted on train data")
        prediction = _predict_ridge(
            self.state, self._features(source, intervention), dtype=source.dtype
        )
        if prediction.shape != source.shape:
            raise ValueError("fitted target state shape does not match source")
        return prediction

    def artifact_payload(self) -> Mapping[str, object]:
        if self.state is None:
            return {**super().artifact_payload(), "fitted": False, "ridge": self.ridge}
        return {
            **super().artifact_payload(),
            "ridge": self.ridge,
            "input_mean": _tensor_summary(self.state.input_mean),
            "input_scale": _tensor_summary(self.state.input_scale),
            "weights": _tensor_summary(self.state.weights),
            "output_shape": list(self.state.output_shape),
        }


class PCARidgeCausalDeltaTransport(AffineRidgeTransport):
    """Train-only PCA/ridge causal-delta transport with frozen target basis."""

    def __init__(
        self,
        *,
        rank: int | None = None,
        ridge: float = 1e-5,
        name: str = "pca_ridge_causal_delta",
        classification: BaselineClass = BaselineClass.DEPLOYABLE_RESIDUALIZER,
    ) -> None:
        super().__init__(ridge=ridge, name=name, classification=classification)
        self.rank = rank
        self.delta_mean: Tensor | None = None
        self.components: Tensor | None = None

    def _fit_train_target(self, train_inputs: BaselineInputView, target_delta: Tensor) -> None:
        target = _flat(target_delta).to(dtype=torch.float64)
        self.delta_mean = target.mean(dim=0, keepdim=True).detach().clone()
        centered = target - self.delta_mean
        _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
        max_rank = min(centered.shape)
        selected_rank = max_rank if self.rank is None else self.rank
        if selected_rank <= 0 or selected_rank > max_rank:
            raise ValueError(f"PCA rank must be in [1, {max_rank}]")
        self.components = vectors[:selected_rank].detach().clone()
        coefficients = centered @ self.components.T
        self.state = _fit_ridge_state(
            self._features(train_inputs.source, train_inputs.intervention),
            coefficients,
            ridge=self.ridge,
        )

    def predict(self, source: Tensor, intervention: Tensor) -> Tensor:
        if self.state is None or self.delta_mean is None or self.components is None:
            raise RuntimeError("PCA/ridge transport must be fitted on train data")
        coefficients = _predict_ridge(
            self.state,
            self._features(source, intervention),
            dtype=self.components.dtype,
        )
        flat_prediction = _flat(coefficients) @ self.components + self.delta_mean
        return flat_prediction.reshape_as(source).to(dtype=source.dtype)

    def artifact_payload(self) -> Mapping[str, object]:
        payload = dict(super().artifact_payload())
        payload["rank"] = self.rank
        if self.delta_mean is not None and self.components is not None:
            payload["delta_mean"] = _tensor_summary(self.delta_mean)
            payload["components"] = _tensor_summary(self.components)
        return payload


@dataclass(frozen=True)
class BaselineScore:
    name: str
    validation_score: float
    artifact_sha256: str

    @property
    def validation_mse(self) -> float:
        """Compatibility alias; use ``validation_score`` with record metric semantics."""

        return self.validation_score


@dataclass(frozen=True)
class BaselineSelectionRecord:
    """Canonical, self-hashed freeze record for the residual target selection."""

    _canonical_payload: str
    sha256: str

    def __post_init__(self) -> None:
        expected = hashlib.sha256(self._canonical_payload.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError(
                "baseline selection record SHA-256 does not match its canonical payload"
            )

    @classmethod
    def build(cls, payload: Mapping[str, object]) -> "BaselineSelectionRecord":
        canonical = _canonical_json(payload)
        return cls(canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    @property
    def payload(self) -> Mapping[str, object]:
        """A decoded copy; mutations cannot invalidate the frozen canonical record."""

        decoded = json.loads(self._canonical_payload)
        if not isinstance(decoded, dict):  # pragma: no cover - canonical records are mappings
            raise RuntimeError("baseline selection record payload is not a mapping")
        return decoded

    def canonical_json(self) -> str:
        return self._canonical_payload

    def save(self, path: str | Path) -> None:
        """Persist the exact canonical payload and self-hash for later audit/replay."""

        Path(path).write_text(
            _canonical_json({"canonical_payload": self._canonical_payload, "sha256": self.sha256})
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BaselineSelectionRecord":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("selection record file must contain a mapping")
        canonical_payload = payload.get("canonical_payload")
        sha256 = payload.get("sha256")
        if not isinstance(canonical_payload, str) or not isinstance(sha256, str):
            raise ValueError("selection record file is missing canonical_payload or sha256")
        return cls(canonical_payload, sha256)


@dataclass(frozen=True)
class BaselineSelection:
    baseline_star: DifferentialBaseline
    scores: tuple[BaselineScore, ...]
    record: BaselineSelectionRecord


_SUPPORTED_ENDPOINT_METRICS = frozenset({"mean_squared_error"})
_SUPPORTED_AGGREGATION_RULES = frozenset(
    {"mean_over_examples_and_coordinates", "normalized_ratio_of_sums"}
)


def _selection_score(
    prediction: Tensor,
    target_delta: Tensor,
    *,
    endpoint_metric: str,
    aggregation_rule: str,
) -> tuple[float, dict[str, object]]:
    if endpoint_metric not in _SUPPORTED_ENDPOINT_METRICS:
        raise BaselineEligibilityError(
            f"unsupported endpoint_metric {endpoint_metric!r}; "
            f"supported={sorted(_SUPPORTED_ENDPOINT_METRICS)}"
        )
    if aggregation_rule not in _SUPPORTED_AGGREGATION_RULES:
        raise BaselineEligibilityError(
            f"unsupported aggregation_rule {aggregation_rule!r}; "
            f"supported={sorted(_SUPPORTED_AGGREGATION_RULES)}"
        )
    squared_error = (prediction - target_delta).square()
    numerator = float(squared_error.sum().detach().cpu())
    if not torch.isfinite(torch.as_tensor(numerator)):
        raise RuntimeError("baseline produced a non-finite validation squared-error numerator")
    if aggregation_rule == "mean_over_examples_and_coordinates":
        denominator = float(squared_error.numel())
        if denominator <= 0.0:
            raise BaselineEligibilityError("mean aggregation requires a positive coordinate count")
        return numerator / denominator, {
            "numerator_semantics": "sum_squared_prediction_error_over_validation_rows",
            "denominator_semantics": "count_validation_effect_coordinates",
            "denominator_value": denominator,
        }
    denominator = float(target_delta.square().sum().detach().cpu())
    if not torch.isfinite(torch.as_tensor(denominator)) or denominator <= 0.0:
        raise BaselineEligibilityError(
            "normalized_ratio_of_sums requires a finite, strictly positive validation "
            "sum_squared_observed_delta denominator"
        )
    return numerator / denominator, {
        "numerator_semantics": "sum_squared_prediction_error_over_validation_rows",
        "denominator_semantics": "sum_squared_observed_delta_over_validation_rows",
        "denominator_value": denominator,
    }


def _fit_deployable_candidate(
    candidate: DifferentialBaseline,
    train_inputs: BaselineInputView,
    train_delta: Tensor,
) -> None:
    """Give outcome targets only to audited built-in fitters, never arbitrary candidates.

    Candidate ``fit`` always receives the online-only view.  Exact type checks intentionally
    prevent a subclass or a third-party candidate from advertising the deployable class and
    silently acquiring train outcomes through this selector.
    """

    candidate.fit(train_inputs)
    trusted_target = train_delta.detach().clone()
    if type(candidate) is MeanEffectBaseline:
        candidate._fit_train_target(trusted_target)
    elif type(candidate) is PopulationDifferentialTransport:
        candidate._fit_train_target(train_inputs, trusted_target)
    elif type(candidate) is AffineRidgeTransport:
        candidate._fit_train_target(train_inputs, trusted_target)
    elif type(candidate) is PCARidgeCausalDeltaTransport:
        candidate._fit_train_target(train_inputs, trusted_target)


def select_train_validation_baseline(
    candidates: Sequence[DifferentialBaseline],
    train: CausalEffectBatch,
    validation: CausalEffectBatch,
    *,
    deployment_inputs: tuple[str, ...] = ("source", "intervention"),
    endpoint_metric: str = "mean_squared_error",
    aggregation_rule: str = "mean_over_examples_and_coordinates",
    tie_break_rule: str = "ascending_validation_mse_then_lexicographic_name",
    normalization: str = "none",
    dimensionality_reduction: str = "none",
) -> BaselineSelection:
    """Fit deployable baselines on train and select exactly once on validation.

    Protected/test rows fail before any candidate can see their tensors.  Oracle and fair
    methods are recorded but cannot define ``baseline_star``.
    """

    train.require_fit_access()
    validation.require_selection_access()
    if endpoint_metric not in _SUPPORTED_ENDPOINT_METRICS:
        raise BaselineEligibilityError(
            f"unsupported endpoint_metric {endpoint_metric!r}; "
            f"supported={sorted(_SUPPORTED_ENDPOINT_METRICS)}"
        )
    if aggregation_rule not in _SUPPORTED_AGGREGATION_RULES:
        raise BaselineEligibilityError(
            f"unsupported aggregation_rule {aggregation_rule!r}; "
            f"supported={sorted(_SUPPORTED_AGGREGATION_RULES)}"
        )
    if not candidates:
        raise BaselineEligibilityError("at least one deployable baseline is required")
    names = [candidate.metadata.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise BaselineEligibilityError("baseline names must be unique for deterministic selection")
    available_inputs = set(deployment_inputs)
    eligible: list[DifferentialBaseline] = []
    excluded: dict[str, str] = {}
    for candidate in candidates:
        metadata = candidate.metadata
        if metadata.classification is not BaselineClass.DEPLOYABLE_RESIDUALIZER:
            excluded[metadata.name] = f"classification={metadata.classification.value}"
            continue
        missing_inputs = sorted(set(metadata.declared_inputs) - available_inputs)
        if missing_inputs:
            excluded[metadata.name] = f"unavailable_inputs={','.join(missing_inputs)}"
            continue
        eligible.append(candidate)
    if not eligible:
        raise BaselineEligibilityError("no deployable baseline has the declared deployment inputs")

    scores: list[BaselineScore] = []
    fitted_artifacts: dict[str, object] = {}
    train_inputs = BaselineInputView.from_train_batch(train)
    validation_inputs = BaselineInputView.from_validation_batch(validation)
    metric_semantics: dict[str, object] | None = None
    for candidate in eligible:
        _fit_deployable_candidate(candidate, train_inputs, train.delta)
        prediction = candidate.predict(
            validation_inputs.source.detach().clone(),
            validation_inputs.intervention.detach().clone(),
        )
        if prediction.shape != validation.delta.shape:
            raise ValueError(
                f"baseline {candidate.metadata.name!r} returned an invalid effect shape"
            )
        score, semantics = _selection_score(
            prediction,
            validation.delta,
            endpoint_metric=endpoint_metric,
            aggregation_rule=aggregation_rule,
        )
        if not torch.isfinite(torch.as_tensor(score)):
            raise RuntimeError(
                f"baseline {candidate.metadata.name!r} produced a non-finite validation score"
            )
        artifact_sha256 = _artifact_digest(candidate)
        scores.append(BaselineScore(candidate.metadata.name, score, artifact_sha256))
        if metric_semantics is None:
            metric_semantics = semantics
        elif metric_semantics != semantics:
            raise RuntimeError("baseline selection metric denominator changed across candidates")
        fitted_artifacts[candidate.metadata.name] = {
            "sha256": artifact_sha256,
            "payload": candidate.artifact_payload(),
        }

    by_name = {candidate.metadata.name: candidate for candidate in eligible}
    ordered_scores = tuple(sorted(scores, key=lambda score: (score.validation_score, score.name)))
    selected_name = ordered_scores[0].name
    payload: dict[str, object] = {
        "schema_version": "causal_residual_baseline_selection_v1",
        "eligible_baselines": sorted(candidate.metadata.name for candidate in eligible),
        "candidate_classifications": {
            candidate.metadata.name: candidate.metadata.classification.value
            for candidate in sorted(candidates, key=lambda item: item.metadata.name)
        },
        "excluded_from_residual_target": dict(sorted(excluded.items())),
        "deployment_inputs": list(deployment_inputs),
        "endpoint_metric": endpoint_metric,
        "aggregation_rule": aggregation_rule,
        "metric_denominator": metric_semantics,
        "tie_break_rule": tie_break_rule,
        "normalization": normalization,
        "dimensionality_reduction": dimensionality_reduction,
        "split_identities": {
            "train": list(train.identities),
            "validation": list(validation.identities),
        },
        "fitted_baseline_artifacts": dict(sorted(fitted_artifacts.items())),
        "validation_scores": {
            score.name: score.validation_score
            for score in sorted(scores, key=lambda item: item.name)
        },
        "selected_baseline": selected_name,
    }
    return BaselineSelection(
        baseline_star=by_name[selected_name],
        scores=ordered_scores,
        record=BaselineSelectionRecord.build(payload),
    )


def _artifact_digest(candidate: DifferentialBaseline) -> str:
    artifact_method = getattr(candidate, "artifact_sha256", None)
    if callable(artifact_method):
        return str(artifact_method())
    return hashlib.sha256(_canonical_json(candidate.artifact_payload()).encode("utf-8")).hexdigest()


__all__ = [
    "AffineRidgeTransport",
    "BaselineInputView",
    "BaselineMetadata",
    "BaselineScore",
    "BaselineSelection",
    "BaselineSelectionRecord",
    "BaseDifferentialBaseline",
    "DifferentiableTransition",
    "DifferentialBaseline",
    "ExactJVPBaseline",
    "MeanEffectBaseline",
    "NoChangeBaseline",
    "ObservedPrefixRelinearizedBaseline",
    "PCARidgeCausalDeltaTransport",
    "PopulationDifferentialTransport",
    "QuadraticHVPBaseline",
    "select_train_validation_baseline",
]
