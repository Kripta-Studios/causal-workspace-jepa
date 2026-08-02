"""Typed PyTorch primitives for Causal-Residual Intervention JEPA.

The module deliberately separates an observed finite effect from a learned residual.
It contains no model-specific execution code: callers supply directly observed ``treated``
states and a baseline predicted from deployable inputs.  In particular, a predictor in this
module never receives a treated state as an online input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TensorMapping: TypeAlias = Mapping[str, Tensor]


class BaselineClass(str, Enum):
    """Scientific role of a comparison method.

    Only ``DEPLOYABLE_RESIDUALIZER`` methods are eligible to define the frozen residual
    target.  Fair comparators remain reportable but cannot change that target, while oracle
    ceilings must never enter baseline selection.
    """

    DEPLOYABLE_RESIDUALIZER = "deployable_residualizer"
    FAIR_COMPARATOR = "fair_comparator"
    ORACLE_CEILING = "oracle_ceiling"


class SplitRole(str, Enum):
    """Role of an effect batch in the preregistered split contract."""

    TRAIN = "train"
    VALIDATION = "validation"
    PROTECTED = "protected"
    TEST = "test"


class ProtectedSplitAccessError(RuntimeError):
    """Raised when a fit/selection path attempts to consume protected rows."""


class BaselineEligibilityError(ValueError):
    """Raised when an ineligible baseline is offered as a residualizer."""


@dataclass(frozen=True)
class CausalEffectBatch:
    """A named split of directly observed intervention outcomes.

    ``source`` and ``treated`` have the same shape ``[batch, ...state]``.  The exact
    finite effect is always ``treated - source``; it is never inferred from labels or
    representations.  ``observed_prefix`` is retained only to make oracle use explicit.
    It cannot be read by train/validation selection helpers.
    """

    source: Tensor
    treated: Tensor
    intervention: Tensor
    split: SplitRole
    example_ids: tuple[str, ...] = ()
    observed_prefix: Tensor | None = None

    def __post_init__(self) -> None:
        if self.source.ndim < 2:
            raise ValueError("source must be shaped [batch, ...state]")
        if self.source.shape != self.treated.shape:
            raise ValueError("source and treated tensors must have identical shapes")
        if self.intervention.ndim < 2:
            raise ValueError("intervention must be shaped [batch, ...intervention]")
        if self.source.shape[0] != self.intervention.shape[0]:
            raise ValueError("source and intervention batch dimensions must match")
        if self.example_ids and len(self.example_ids) != self.source.shape[0]:
            raise ValueError("example_ids must be empty or have one entry per batch row")
        if (
            self.observed_prefix is not None
            and self.observed_prefix.shape[0] != self.source.shape[0]
        ):
            raise ValueError("observed_prefix batch dimension must match source")

    @property
    def delta(self) -> Tensor:
        """Exact observed finite effect ``x_u - x``."""

        return exact_delta(self.source, self.treated)

    @property
    def identities(self) -> tuple[str, ...]:
        if self.example_ids:
            return self.example_ids
        return tuple(f"{self.split.value}:{index}" for index in range(self.source.shape[0]))

    def require_fit_access(self) -> None:
        if self.split is not SplitRole.TRAIN:
            raise ProtectedSplitAccessError(
                f"fitting is train-only; received a {self.split.value!r} effect batch"
            )

    def require_selection_access(self) -> None:
        if self.split is not SplitRole.VALIDATION:
            raise ProtectedSplitAccessError(
                "baseline selection is validation-only after train fitting; "
                f"received a {self.split.value!r} effect batch"
            )

    def require_evaluation_access(self, *, protected_execution_authorized: bool = False) -> None:
        if (
            self.split in {SplitRole.PROTECTED, SplitRole.TEST}
            and not protected_execution_authorized
        ):
            raise ProtectedSplitAccessError(
                f"{self.split.value} outcomes are fail-closed until explicit authorization"
            )


def _require_same_shape(*tensors: Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"expected matching tensor shapes, received {sorted(shapes)}")


def exact_delta(source: Tensor, treated: Tensor) -> Tensor:
    """Return the non-negotiable finite-effect identity ``delta = x_u - x``."""

    _require_same_shape(source, treated)
    return treated - source


def residual_from_baseline(delta: Tensor, baseline_star: Tensor) -> Tensor:
    """Return ``delta - baseline_star`` with shape checking rather than broadcasting."""

    _require_same_shape(delta, baseline_star)
    return delta - baseline_star


def reconstruct_delta(baseline_star: Tensor, predicted_residual: Tensor) -> Tensor:
    """Return the complete finite effect reconstructed from baseline plus residual."""

    _require_same_shape(baseline_star, predicted_residual)
    return baseline_star + predicted_residual


def reconstruct_treated(
    source: Tensor, baseline_star: Tensor, predicted_residual: Tensor
) -> Tensor:
    """Return ``x_hat_u = x + baseline_star + predicted_residual`` exactly."""

    _require_same_shape(source, baseline_star, predicted_residual)
    return source + reconstruct_delta(baseline_star, predicted_residual)


@dataclass(frozen=True)
class CausalResidualTarget:
    """Frozen baseline target and its residual decomposition for one effect batch."""

    source: Tensor
    treated: Tensor
    delta: Tensor
    baseline_star: Tensor
    residual: Tensor

    @classmethod
    def from_effect(cls, batch: CausalEffectBatch, baseline_star: Tensor) -> "CausalResidualTarget":
        delta = batch.delta
        residual = residual_from_baseline(delta, baseline_star)
        return cls(
            source=batch.source,
            treated=batch.treated,
            delta=delta,
            baseline_star=baseline_star,
            residual=residual,
        )

    def reconstruct(self, predicted_residual: Tensor) -> Tensor:
        return reconstruct_treated(self.source, self.baseline_star, predicted_residual)

    def replay_error(self, predicted_residual: Tensor) -> Tensor:
        return F.mse_loss(self.reconstruct(predicted_residual), self.treated)


@dataclass(frozen=True)
class AffineStateTransform:
    """An affine intervention map used for identity/inverse/composition controls.

    The row-major convention is ``apply(x) = x @ matrix.T + bias``.  This makes the
    sequential order explicit and gives an exact noncommuting control when matrices do not
    commute.
    """

    matrix: Tensor
    bias: Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be square")
        if self.bias.ndim != 1 or self.bias.shape[0] != self.matrix.shape[0]:
            raise ValueError("bias must have the state dimension")

    @property
    def state_dim(self) -> int:
        return int(self.matrix.shape[0])

    def apply(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError("state dimension does not match affine transform")
        return state @ self.matrix.T + self.bias

    def delta(self, state: Tensor) -> Tensor:
        return self.apply(state) - state

    def inverse(self) -> "AffineStateTransform":
        inverse_matrix = torch.linalg.inv(self.matrix)
        inverse_bias = -self.bias @ inverse_matrix.T
        return AffineStateTransform(inverse_matrix, inverse_bias)

    def then(self, later: "AffineStateTransform") -> "AffineStateTransform":
        """Return ``later(self(state))``."""

        if self.state_dim != later.state_dim:
            raise ValueError("affine transforms must share a state dimension")
        return AffineStateTransform(
            matrix=later.matrix @ self.matrix,
            bias=self.bias @ later.matrix.T + later.bias,
        )


def sequential_delta(
    state: Tensor, first: AffineStateTransform, second: AffineStateTransform
) -> Tensor:
    """Exact finite effect of applying ``first`` and then ``second``."""

    return first.then(second).delta(state)


def commutator_delta(
    state: Tensor, first: AffineStateTransform, second: AffineStateTransform
) -> Tensor:
    """Return the observable interaction ``second∘first - first∘second``."""

    return first.then(second).apply(state) - second.then(first).apply(state)


def normalized_residual_reconstruction_loss(
    predicted_residual: Tensor,
    target_residual: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Scale-normalized residual MSE, stable for zero-residual no-go cases."""

    _require_same_shape(predicted_residual, target_residual)
    denominator = target_residual.square().mean().detach().clamp_min(epsilon)
    return F.mse_loss(predicted_residual, target_residual) / denominator


def direct_replay_endpoint_loss(
    source: Tensor,
    baseline_star: Tensor,
    predicted_residual: Tensor,
    treated: Tensor,
) -> Tensor:
    """Complete-state replay loss while retaining residual-only learned output."""

    _require_same_shape(source, baseline_star, predicted_residual, treated)
    return F.mse_loss(reconstruct_treated(source, baseline_star, predicted_residual), treated)


def identity_residual_loss(predicted_identity_residual: Tensor) -> Tensor:
    """Penalty for assigning a nonzero residual to an identity intervention."""

    return predicted_identity_residual.square().mean()


def inverse_restoration_loss(
    source: Tensor,
    forward: AffineStateTransform,
    inverse: AffineStateTransform,
) -> Tensor:
    """Penalty for failing to restore source state after an inverse intervention."""

    restored = inverse.apply(forward.apply(source))
    return F.mse_loss(restored, source)


def sequential_composition_loss(
    source: Tensor,
    first: AffineStateTransform,
    second: AffineStateTransform,
    predicted_delta: Tensor,
) -> Tensor:
    """Penalty against the exact sequential finite-effect target."""

    target = sequential_delta(source, first, second)
    _require_same_shape(target, predicted_delta)
    return F.mse_loss(predicted_delta, target)


def commutator_interaction_loss(
    source: Tensor,
    first: AffineStateTransform,
    second: AffineStateTransform,
    predicted_commutator: Tensor,
) -> Tensor:
    """Penalty against a noncommuting composition interaction target."""

    target = commutator_delta(source, first, second)
    _require_same_shape(target, predicted_commutator)
    return F.mse_loss(predicted_commutator, target)


def matched_control_specificity_loss(
    targeted_error: Tensor,
    matched_control_error: Tensor,
    *,
    required_margin: float = 0.0,
) -> Tensor:
    """Hinge loss requiring the targeted condition to beat its matched corruption control."""

    return torch.relu(targeted_error - matched_control_error + required_margin)


def uncertainty_calibration_nll(
    predicted_mean: Tensor,
    predicted_log_variance: Tensor,
    target: Tensor,
) -> Tensor:
    """Diagonal Gaussian NLL for an optional residual uncertainty head."""

    _require_same_shape(predicted_mean, predicted_log_variance, target)
    inverse_variance = torch.exp(-predicted_log_variance)
    return (
        0.5
        * (predicted_log_variance + (target - predicted_mean).square() * inverse_variance).mean()
    )


@dataclass(frozen=True)
class ResidualLossBundle:
    """Named losses that can be independently weighted by a preregistered objective."""

    terms: TensorMapping

    def total(self, weights: Mapping[str, float] | None = None) -> Tensor:
        if not self.terms:
            raise ValueError("at least one residual loss term is required")
        total = next(iter(self.terms.values())).new_zeros(())
        for name, value in self.terms.items():
            total = total + value * (1.0 if weights is None else weights.get(name, 1.0))
        return total


@dataclass(frozen=True)
class MatchedActionControl:
    """A batch permutation that preserves the complete action norm/covariance distribution."""

    actions: Tensor
    permutation: Tensor
    max_norm_difference: float
    max_covariance_difference: float


@dataclass(frozen=True)
class NormMatchedActionControl:
    """Semantically irrelevant random directions with each action's original L2 norm."""

    actions: Tensor
    max_norm_difference: float


def norm_matched_random_direction_control(
    actions: Tensor, *, seed: int
) -> NormMatchedActionControl:
    """Replace directions while preserving every example's action norm exactly up to roundoff."""

    if actions.ndim != 2 or actions.shape[-1] < 2:
        raise ValueError("actions must be rank-two with at least two action coordinates")
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(seed)
    direction = torch.randn(
        actions.shape,
        dtype=actions.dtype,
        device=actions.device,
        generator=generator,
    )
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    controlled = direction * torch.linalg.vector_norm(actions, dim=-1, keepdim=True)
    difference = torch.linalg.vector_norm(controlled, dim=-1) - torch.linalg.vector_norm(
        actions, dim=-1
    )
    return NormMatchedActionControl(
        actions=controlled,
        max_norm_difference=float(difference.abs().max().detach().cpu()),
    )


def matched_norm_covariance_action_control(actions: Tensor, *, seed: int) -> MatchedActionControl:
    """Permute actions across examples without changing their empirical norms or covariance.

    A permutation is intentionally used instead of a random Euclidean direction: it preserves
    the exact sample covariance and multiset of norms, so it is a valid matched action control.
    """

    if actions.ndim != 2 or actions.shape[0] < 2:
        raise ValueError("actions must be a rank-two tensor with at least two examples")
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(actions.shape[0], generator=generator, device=actions.device)
    if bool(torch.equal(permutation, torch.arange(actions.shape[0], device=actions.device))):
        permutation = torch.roll(permutation, shifts=1)
    controlled = actions[permutation]
    norm_difference = (
        torch.sort(torch.linalg.vector_norm(actions, dim=-1)).values
        - torch.sort(torch.linalg.vector_norm(controlled, dim=-1)).values
    )
    centered = actions - actions.mean(dim=0, keepdim=True)
    controlled_centered = controlled - controlled.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(actions.shape[0] - 1, 1)
    controlled_covariance = (
        controlled_centered.T @ controlled_centered / max(actions.shape[0] - 1, 1)
    )
    return MatchedActionControl(
        actions=controlled,
        permutation=permutation,
        max_norm_difference=float(norm_difference.abs().max().detach().cpu()),
        max_covariance_difference=float(
            (covariance - controlled_covariance).abs().max().detach().cpu()
        ),
    )


class ResidualPredictor(Protocol):
    """A learned component that maps deployable inputs to a residual only."""

    def __call__(self, context: Tensor, intervention: Tensor, **kwargs: object) -> Tensor: ...


class CausalResidualJEPA(nn.Module):
    """Small wrapper that enforces residual-only online prediction and replay reconstruction."""

    target_kind = "residual"

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, context: Tensor, intervention: Tensor, **kwargs: object) -> Tensor:
        return self.predictor(context, intervention, **kwargs)

    def replay(
        self,
        source: Tensor,
        baseline_star: Tensor,
        context: Tensor,
        intervention: Tensor,
        **kwargs: object,
    ) -> Tensor:
        predicted_residual = self(context, intervention, **kwargs)
        return reconstruct_treated(source, baseline_star, predicted_residual)


@dataclass(frozen=True)
class RoutingOutput:
    """Inspectable transport output; weights alone are never a causal claim."""

    prediction: Tensor
    routing_weights: Tensor
    transported_values: Tensor


def _sequence_context(context: Tensor) -> Tensor:
    if context.ndim == 2:
        return context.unsqueeze(1)
    if context.ndim != 3:
        raise ValueError("context must be [batch, state] or [batch, sites, state]")
    return context


def _flat_intervention(intervention: Tensor) -> Tensor:
    if intervention.ndim < 2:
        raise ValueError("intervention must have a batch and at least one feature dimension")
    return intervention.reshape(intervention.shape[0], -1)


class NoRoutingLocalResidualPredictor(nn.Module):
    """Local residual predictor that cannot transport information across sites."""

    def __init__(self, state_dim: int, intervention_dim: int, *, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.intervention_dim = intervention_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim + intervention_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        context: Tensor,
        intervention: Tensor,
        *,
        routing_ablation: bool = False,
        value_ablation: bool = False,
    ) -> Tensor:
        del routing_ablation
        sequence = _sequence_context(context)
        local = sequence[:, 0]
        intervention = _flat_intervention(intervention)
        if local.shape[-1] != self.state_dim or intervention.shape[-1] != self.intervention_dim:
            raise ValueError("local predictor input dimensions do not match its construction")
        if value_ablation:
            local = torch.zeros_like(local)
        return self.network(torch.cat([local, intervention], dim=-1))


class ParameterMatchedMLPResidualPredictor(NoRoutingLocalResidualPredictor):
    """A no-routing MLP whose width can be selected to match a reference parameter budget."""

    @classmethod
    def matched_to(
        cls,
        reference: nn.Module,
        state_dim: int,
        intervention_dim: int,
        *,
        minimum_hidden_dim: int = 4,
        maximum_hidden_dim: int = 4096,
    ) -> "ParameterMatchedMLPResidualPredictor":
        budget = sum(parameter.numel() for parameter in reference.parameters())
        input_dim = state_dim + intervention_dim

        def parameter_count(hidden_dim: int) -> int:
            return (input_dim + 1) * hidden_dim + (hidden_dim + 1) * state_dim

        candidates = range(minimum_hidden_dim, maximum_hidden_dim + 1)
        hidden_dim = min(
            candidates, key=lambda width: (abs(parameter_count(width) - budget), width)
        )
        return cls(state_dim, intervention_dim, hidden_dim=hidden_dim)


class StandardCrossAttentionResidualPredictor(nn.Module):
    """Standard cross-attention residual candidate with visible routing/value ablations."""

    def __init__(
        self,
        state_dim: int,
        intervention_dim: int,
        *,
        hidden_dim: int = 64,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if state_dim % num_heads:
            raise ValueError("state_dim must be divisible by num_heads")
        self.state_dim = state_dim
        self.intervention_dim = intervention_dim
        self.query = nn.Linear(intervention_dim, state_dim)
        self.attention = nn.MultiheadAttention(state_dim, num_heads, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(state_dim + intervention_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        context: Tensor,
        intervention: Tensor,
        *,
        routing_ablation: bool = False,
        value_ablation: bool = False,
        return_routing: bool = False,
    ) -> Tensor | RoutingOutput:
        sequence = _sequence_context(context)
        intervention = _flat_intervention(intervention)
        if sequence.shape[-1] != self.state_dim or intervention.shape[-1] != self.intervention_dim:
            raise ValueError("cross-attention input dimensions do not match its construction")
        values = torch.zeros_like(sequence) if value_ablation else sequence
        if routing_ablation:
            weights = torch.full(
                (sequence.shape[0], 1, sequence.shape[1]),
                1.0 / sequence.shape[1],
                dtype=sequence.dtype,
                device=sequence.device,
            )
            transport = torch.bmm(weights, values).squeeze(1)
        else:
            query = self.query(intervention).unsqueeze(1)
            transport, weights = self.attention(
                query,
                sequence,
                values,
                need_weights=True,
                average_attn_weights=True,
            )
            transport = transport.squeeze(1)
        prediction = self.readout(torch.cat([transport, intervention], dim=-1))
        if return_routing:
            return RoutingOutput(prediction, weights, transport)
        return prediction


class ConditionalLowRankRouter(nn.Module):
    """Conditional low-rank routing with full-width transported source values.

    Only keys/queries are low rank.  The transported values stay in ``state_dim`` and can be
    ablated independently from routing weights, preventing a low-rank value bottleneck from
    being mistaken for a routing result.
    """

    def __init__(
        self,
        state_dim: int,
        intervention_dim: int,
        coordinate_dim: int,
        *,
        routing_rank: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if routing_rank <= 0:
            raise ValueError("routing_rank must be positive")
        self.state_dim = state_dim
        self.intervention_dim = intervention_dim
        self.coordinate_dim = coordinate_dim
        self.routing_rank = routing_rank
        self.coordinate_key = nn.Linear(coordinate_dim, routing_rank, bias=False)
        self.condition_key = nn.Linear(intervention_dim, routing_rank, bias=False)
        self.condition_query = nn.Linear(intervention_dim, routing_rank, bias=False)
        self.value_gate = nn.Sequential(nn.Linear(intervention_dim, state_dim), nn.Sigmoid())
        self.readout = nn.Sequential(
            nn.Linear(state_dim + intervention_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        context: Tensor,
        intervention: Tensor,
        coordinates: Tensor,
        *,
        routing_ablation: bool = False,
        value_ablation: bool = False,
        return_routing: bool = False,
    ) -> Tensor | RoutingOutput:
        sequence = _sequence_context(context)
        intervention = _flat_intervention(intervention)
        if sequence.shape[-1] != self.state_dim or intervention.shape[-1] != self.intervention_dim:
            raise ValueError("router input dimensions do not match its construction")
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0).expand(sequence.shape[0], -1, -1)
        if coordinates.ndim != 3 or coordinates.shape[:2] != sequence.shape[:2]:
            raise ValueError("coordinates must be [sites, coord] or [batch, sites, coord]")
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError("coordinate dimensionality does not match router construction")
        keys = self.coordinate_key(coordinates) + self.condition_key(intervention).unsqueeze(1)
        query = self.condition_query(intervention).unsqueeze(1)
        logits = (keys * query).sum(dim=-1) / math.sqrt(self.routing_rank)
        if routing_ablation:
            weights = torch.full_like(logits, 1.0 / logits.shape[-1])
        else:
            weights = torch.softmax(logits, dim=-1)
        full_width_values = sequence * self.value_gate(intervention).unsqueeze(1)
        if value_ablation:
            full_width_values = torch.zeros_like(full_width_values)
        transport = torch.bmm(weights.unsqueeze(1), full_width_values).squeeze(1)
        prediction = self.readout(torch.cat([transport, intervention], dim=-1))
        if return_routing:
            return RoutingOutput(prediction, weights.unsqueeze(1), transport)
        return prediction


__all__ = [
    "AffineStateTransform",
    "BaselineClass",
    "BaselineEligibilityError",
    "CausalEffectBatch",
    "CausalResidualJEPA",
    "CausalResidualTarget",
    "ConditionalLowRankRouter",
    "MatchedActionControl",
    "NormMatchedActionControl",
    "NoRoutingLocalResidualPredictor",
    "ParameterMatchedMLPResidualPredictor",
    "ProtectedSplitAccessError",
    "ResidualLossBundle",
    "RoutingOutput",
    "SplitRole",
    "StandardCrossAttentionResidualPredictor",
    "commutator_delta",
    "commutator_interaction_loss",
    "direct_replay_endpoint_loss",
    "exact_delta",
    "identity_residual_loss",
    "inverse_restoration_loss",
    "matched_control_specificity_loss",
    "matched_norm_covariance_action_control",
    "norm_matched_random_direction_control",
    "normalized_residual_reconstruction_loss",
    "reconstruct_delta",
    "reconstruct_treated",
    "residual_from_baseline",
    "sequential_composition_loss",
    "sequential_delta",
    "uncertainty_calibration_nll",
]
