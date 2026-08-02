"""Deterministic Stage-0 falsification benchmark for Causal-Residual JEPA.

The benchmark uses observable physical-state targets only.  It intentionally contains two
no-go systems where analytical residualizers are exact, one system with a finite nonlinear
compositional residual, and one predictable nuisance trap for complete-state objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn

from causal_workspace_jepa.interpretability.differential_baselines import (
    AffineRidgeTransport,
    BaselineSelection,
    ExactJVPBaseline,
    MeanEffectBaseline,
    NoChangeBaseline,
    PCARidgeCausalDeltaTransport,
    PopulationDifferentialTransport,
    QuadraticHVPBaseline,
    select_train_validation_baseline,
)
from causal_workspace_jepa.models.causal_residual import (
    CausalEffectBatch,
    CausalResidualTarget,
    SplitRole,
    matched_norm_covariance_action_control,
    norm_matched_random_direction_control,
    reconstruct_treated,
)


MetricValue = bool | float | int | str


@dataclass(frozen=True)
class Stage0CaseResult:
    """Outcome of one preregistered deterministic falsification case."""

    case_id: str
    status: str
    metrics: Mapping[str, MetricValue]
    selection_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "selection_sha256": self.selection_sha256,
        }


@dataclass(frozen=True)
class Stage0BenchmarkResult:
    """All four Stage-0 cases, with no protected model execution."""

    seed: int
    status: str
    cases: tuple[Stage0CaseResult, ...]

    def by_case(self, case_id: str) -> Stage0CaseResult:
        for result in self.cases:
            if result.case_id == case_id:
                return result
        raise KeyError(case_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "status": self.status,
            "cases": [result.to_dict() for result in self.cases],
        }


def _tensor_sha256(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TrainOnlyPolynomialResidualHead(nn.Module):
    """Serializable train-only residual head for the known Stage-0 composition system."""

    def __init__(self, feature_dim: int, state_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.readout = nn.Linear(feature_dim, state_dim, dtype=torch.float64)
        self.ridge = 1e-10
        self.fit_count = 0
        self.train_input_sha256 = ""
        self.target_sha256 = ""

    def forward(self, source: Tensor, intervention: Tensor) -> Tensor:
        return self.readout(_compositional_features(source, intervention))

    @classmethod
    def fit(
        cls,
        train: CausalEffectBatch,
        residual: Tensor,
        *,
        ridge: float = 1e-10,
    ) -> "TrainOnlyPolynomialResidualHead":
        if residual.shape != train.delta.shape:
            raise ValueError("residual learner target must match the train effect shape")
        model = cls(
            _compositional_features(train.source, train.intervention).shape[-1], residual.shape[-1]
        )
        return model.resume_fit(train, residual, ridge=ridge)

    def resume_fit(
        self,
        train: CausalEffectBatch,
        residual: Tensor,
        *,
        ridge: float | None = None,
    ) -> "TrainOnlyPolynomialResidualHead":
        train.require_fit_access()
        if residual.shape != train.delta.shape:
            raise ValueError("residual learner target must match the train effect shape")
        features = _compositional_features(train.source, train.intervention)
        if features.shape[-1] != self.feature_dim or residual.shape[-1] != self.state_dim:
            raise ValueError("resume target dimensions do not match the saved residual head")
        effective_ridge = self.ridge if ridge is None else ridge
        if effective_ridge <= 0.0:
            raise ValueError("ridge must be positive")
        design = torch.cat(
            [features, torch.ones((features.shape[0], 1), dtype=features.dtype)], dim=-1
        )
        regularizer = torch.eye(design.shape[1], dtype=features.dtype) * effective_ridge
        regularizer[-1, -1] = 0.0
        weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ residual)
        with torch.no_grad():
            self.readout.weight.copy_(weights[:-1].T)
            self.readout.bias.copy_(weights[-1])
        self.ridge = effective_ridge
        self.fit_count += 1
        self.train_input_sha256 = _tensor_sha256(train.source, train.intervention)
        self.target_sha256 = _tensor_sha256(residual)
        return self

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0_polynomial_residual_head_v1",
            "feature_dim": self.feature_dim,
            "state_dim": self.state_dim,
            "ridge": self.ridge,
            "fit_count": self.fit_count,
            "train_input_sha256": self.train_input_sha256,
            "target_sha256": self.target_sha256,
            "state_sha256": _tensor_sha256(self.readout.weight, self.readout.bias),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_dim": self.feature_dim,
            "state_dim": self.state_dim,
            "ridge": self.ridge,
            "fit_count": self.fit_count,
            "train_input_sha256": self.train_input_sha256,
            "target_sha256": self.target_sha256,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "TrainOnlyPolynomialResidualHead":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("residual-head checkpoint must be a mapping")
        model = cls(int(payload["feature_dim"]), int(payload["state_dim"]))
        state = payload["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("residual-head checkpoint state_dict must be a mapping")
        model.load_state_dict(state)
        model.ridge = float(payload["ridge"])
        model.fit_count = int(payload["fit_count"])
        model.train_input_sha256 = str(payload["train_input_sha256"])
        model.target_sha256 = str(payload["target_sha256"])
        return model


def _compositional_features(source: Tensor, action: Tensor) -> Tensor:
    """Source/action monomials available before treatment, not a treated-state feature."""

    action_0 = action[:, 0:1]
    action_1 = action[:, 1:2]
    quadratic = torch.cat([action_0.square(), action_0 * action_1, action_1.square()], dim=-1)
    cubic = torch.cat(
        [
            action_0.pow(3),
            action_0.square() * action_1,
            action_0 * action_1.square(),
            action_1.pow(3),
        ],
        dim=-1,
    )
    source_cubic = (source.unsqueeze(-1) * cubic.unsqueeze(1)).reshape(source.shape[0], -1)
    return torch.cat([source, action, quadratic, cubic, source_cubic], dim=-1)


def _split_effects(
    source: Tensor,
    intervention: Tensor,
    transition: Callable[[Tensor, Tensor], Tensor],
    *,
    prefix: str,
) -> tuple[CausalEffectBatch, CausalEffectBatch, CausalEffectBatch]:
    total = source.shape[0]
    train_end = int(total * 0.6)
    validation_end = int(total * 0.8)
    treated = transition(source, intervention)

    def batch(start: int, end: int, split: SplitRole) -> CausalEffectBatch:
        return CausalEffectBatch(
            source=source[start:end],
            treated=treated[start:end],
            intervention=intervention[start:end],
            split=split,
            example_ids=tuple(f"{prefix}-{index:04d}" for index in range(start, end)),
        )

    return (
        batch(0, train_end, SplitRole.TRAIN),
        batch(train_end, validation_end, SplitRole.VALIDATION),
        batch(validation_end, total, SplitRole.TEST),
    )


def _eligible_selection(
    transition: Callable[[Tensor, Tensor], Tensor],
    train: CausalEffectBatch,
    validation: CausalEffectBatch,
) -> BaselineSelection:
    """Fixed Stage-0 deployable set; the oracle is intentionally absent."""

    pca_rank = min(3, train.delta.reshape(train.delta.shape[0], -1).shape[-1])
    return select_train_validation_baseline(
        (
            NoChangeBaseline(),
            MeanEffectBaseline(),
            ExactJVPBaseline(transition),
            QuadraticHVPBaseline(transition),
            PopulationDifferentialTransport(ridge=1e-7),
            AffineRidgeTransport(ridge=1e-7),
            PCARidgeCausalDeltaTransport(rank=pca_rank, ridge=1e-7),
        ),
        train,
        validation,
        normalization="none_observable_frozen_state",
        dimensionality_reduction="pca_train_only_for_registered_pca_ridge_candidate",
    )


def _test_baseline(selection: BaselineSelection, test: CausalEffectBatch) -> Tensor:
    """Stage-0 is local observable data; test evaluation is explicit rather than implicit."""

    return selection.baseline_star.predict_batch(test, protected_execution_authorized=True)


def _mse(left: Tensor, right: Tensor) -> float:
    return float(torch.mean((left - right).square()).detach().cpu())


def _relative_power(residual: Tensor, delta: Tensor) -> float:
    return float((residual.square().mean() / delta.square().mean().clamp_min(1e-18)).detach().cpu())


def _linear_case(generator: torch.Generator) -> Stage0CaseResult:
    source = torch.randn((180, 3), generator=generator, dtype=torch.float64)
    action = torch.rand((180, 2), generator=generator, dtype=torch.float64) * 1.6 - 0.8
    linear = torch.tensor([[0.7, -0.2], [0.15, 0.9], [-0.4, 0.25]], dtype=torch.float64)

    def transition(state: Tensor, control: Tensor) -> Tensor:
        return state + control @ linear.T

    train, validation, test = _split_effects(source, action, transition, prefix="linear")
    selection = _eligible_selection(transition, train, validation)
    baseline = _test_baseline(selection, test)
    target = CausalResidualTarget.from_effect(test, baseline)
    replay = reconstruct_treated(test.source, baseline, torch.zeros_like(target.residual))
    residual_power = _relative_power(target.residual, target.delta)
    max_abs_residual = float(target.residual.abs().max().detach().cpu())
    passed = (
        selection.baseline_star.metadata.name == "exact_jvp"
        and max_abs_residual < 1e-10
        and _mse(replay, test.treated) < 1e-20
    )
    return Stage0CaseResult(
        case_id="linear_zero_residual",
        status="NEGATIVE_RESULT",
        metrics={
            "selected_baseline": selection.baseline_star.metadata.name,
            "baseline_replay_mse": _mse(replay, test.treated),
            "max_abs_residual": max_abs_residual,
            "residual_power_fraction": residual_power,
            "learned_discovery_permitted": False,
            "passed_zero_residual_gate": passed,
        },
        selection_sha256=selection.record.sha256,
    )


def _quadratic_case(generator: torch.Generator) -> Stage0CaseResult:
    source = torch.randn((180, 3), generator=generator, dtype=torch.float64)
    action = torch.rand((180, 2), generator=generator, dtype=torch.float64) * 1.6 - 0.8
    linear = torch.tensor([[0.55, -0.3], [0.2, 0.7], [-0.1, 0.4]], dtype=torch.float64)
    quadratic = torch.tensor([[0.22, -0.11], [-0.16, 0.31], [0.08, 0.18]], dtype=torch.float64)

    def transition(state: Tensor, control: Tensor) -> Tensor:
        return state + control @ linear.T + control.square() @ quadratic.T

    train, validation, test = _split_effects(source, action, transition, prefix="quadratic")
    selection = _eligible_selection(transition, train, validation)
    baseline = _test_baseline(selection, test)
    target = CausalResidualTarget.from_effect(test, baseline)
    replay = reconstruct_treated(test.source, baseline, torch.zeros_like(target.residual))
    max_abs_residual = float(target.residual.abs().max().detach().cpu())
    passed = (
        selection.baseline_star.metadata.name == "quadratic_hvp"
        and max_abs_residual < 1e-10
        and _mse(replay, test.treated) < 1e-20
    )
    return Stage0CaseResult(
        case_id="quadratic_hvp_zero_false_discovery",
        status="NEGATIVE_RESULT",
        metrics={
            "selected_baseline": selection.baseline_star.metadata.name,
            "baseline_replay_mse": _mse(replay, test.treated),
            "max_abs_residual": max_abs_residual,
            "residual_power_fraction": _relative_power(target.residual, target.delta),
            "learned_discovery_permitted": False,
            "passed_zero_false_discovery_gate": passed,
        },
        selection_sha256=selection.record.sha256,
    )


def _nonlinear_compositional_case(generator: torch.Generator) -> Stage0CaseResult:
    source = torch.randn((240, 3), generator=generator, dtype=torch.float64) * 0.6
    action = torch.rand((240, 2), generator=generator, dtype=torch.float64) * 1.6 - 0.8
    linear = torch.tensor([[0.6, -0.2], [0.1, 0.75], [-0.35, 0.3]], dtype=torch.float64)
    quadratic = torch.tensor([[0.24, -0.08], [-0.12, 0.2], [0.1, 0.15]], dtype=torch.float64)
    cubic_weights = torch.tensor(
        [[0.16, 0.52, -0.38, 0.10], [-0.12, -0.26, 0.44, -0.08], [0.08, 0.32, 0.18, -0.14]],
        dtype=torch.float64,
    )
    source_cubic_weights = torch.tensor(
        [[0.00, 0.17, 0.00, 0.00], [0.00, -0.09, 0.00, 0.00], [0.00, 0.05, 0.00, 0.00]],
        dtype=torch.float64,
    )

    def transition(state: Tensor, control: Tensor) -> Tensor:
        c0 = control[:, 0:1]
        c1 = control[:, 1:2]
        cubic = torch.cat([c0.pow(3), c0.square() * c1, c0 * c1.square(), c1.pow(3)], dim=-1)
        finite_residual = cubic @ cubic_weights.T + state * (cubic @ source_cubic_weights.T)
        return state + control @ linear.T + control.square() @ quadratic.T + finite_residual

    train, validation, test = _split_effects(source, action, transition, prefix="nonlinear")
    selection = _eligible_selection(transition, train, validation)
    baseline_train = selection.baseline_star.predict(train.source, train.intervention)
    train_target = CausalResidualTarget.from_effect(train, baseline_train)
    learner = TrainOnlyPolynomialResidualHead.fit(train, train_target.residual)
    baseline_test = _test_baseline(selection, test)
    predicted_residual = learner(test.source, test.intervention)
    baseline_replay = reconstruct_treated(
        test.source, baseline_test, torch.zeros_like(predicted_residual)
    )
    learned_replay = reconstruct_treated(test.source, baseline_test, predicted_residual)
    baseline_mse = _mse(baseline_replay, test.treated)
    learned_mse = _mse(learned_replay, test.treated)
    improvement = 1.0 - learned_mse / max(baseline_mse, 1e-18)
    passed = baseline_mse > 1e-4 and learned_mse < baseline_mse * 0.05 and improvement > 0.95
    return Stage0CaseResult(
        case_id="nonlinear_compositional_residual",
        status="SMOKE_VALIDATED" if passed else "NEGATIVE_RESULT",
        metrics={
            "selected_baseline": selection.baseline_star.metadata.name,
            "eligible_baseline_replay_mse": baseline_mse,
            "residual_learner_replay_mse": learned_mse,
            "direct_replay_improvement_fraction": improvement,
            "passed_nonlinear_replay_gate": passed,
        },
        selection_sha256=selection.record.sha256,
    )


@dataclass(frozen=True)
class _ObsessedNuisanceSuite:
    """Observable Stage-0 sequences with planted, semantically irrelevant shortcuts."""

    observations: Tensor
    next_observations: Tensor
    physical: Tensor
    next_physical: Tensor
    actions: Tensor
    episode_ids: Tensor
    timestamps: Tensor
    template_ids: Tensor
    bit_strings: Tensor
    train_indices: Tensor
    validation_indices: Tensor
    test_indices: Tensor


class _NaiveNextStateJEPA(nn.Module):
    """Fixed-budget next-state bottleneck that has no action input or nuisance guard."""

    def __init__(self, observation_dim: int, *, latent_dim: int = 2) -> None:
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(observation_dim, 16),
            nn.GELU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, observation_dim),
        )

    def encode(self, observation: Tensor) -> Tensor:
        return self.encoder_net(observation)

    def decode(self, representation: Tensor) -> Tensor:
        return self.decoder_net(representation)

    def forward(self, observation: Tensor) -> Tensor:
        return self.decode(self.encode(observation))


class _GuardedResidualActionJEPA(nn.Module):
    """Stage-0 positive control with a frozen physical-input mask and action conditioning."""

    def __init__(self, *, physical_dim: int = 2, action_dim: int = 2) -> None:
        super().__init__()
        self.physical_dim = physical_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(physical_dim, 8),
            nn.GELU(),
            nn.Linear(8, 2),
        )
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, 2), nn.Tanh())
        self.residual_predictor = nn.Sequential(
            nn.Linear(4, 8),
            nn.GELU(),
            nn.Linear(8, physical_dim),
        )

    def encode(self, observation: Tensor, action: Tensor) -> Tensor:
        return torch.cat(
            [self.state_encoder(observation[:, : self.physical_dim]), self.action_encoder(action)],
            dim=-1,
        )

    def decode(self, representation: Tensor) -> Tensor:
        return self.residual_predictor(representation)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        return self.decode(self.encode(observation, action))


@dataclass(frozen=True)
class _LinearMap:
    mean: Tensor
    scale: Tensor
    weights: Tensor

    def predict(self, features: Tensor) -> Tensor:
        normalized = (features - self.mean) / self.scale
        design = torch.cat(
            [
                normalized,
                torch.ones(
                    (normalized.shape[0], 1), dtype=normalized.dtype, device=normalized.device
                ),
            ],
            dim=-1,
        )
        return design @ self.weights


def _fit_linear_map(features: Tensor, target: Tensor, *, ridge: float = 1e-4) -> _LinearMap:
    features = features.to(dtype=torch.float64)
    target = target.to(dtype=torch.float64)
    mean = features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / scale
    design = torch.cat(
        [normalized, torch.ones((normalized.shape[0], 1), dtype=normalized.dtype)], dim=-1
    )
    regularizer = torch.eye(design.shape[1], dtype=design.dtype) * ridge
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return _LinearMap(mean, scale, weights)


def _linear_r2(
    train_features: Tensor, train_target: Tensor, test_features: Tensor, test_target: Tensor
) -> float:
    predictor = _fit_linear_map(train_features, train_target)
    prediction = predictor.predict(test_features)
    baseline = train_target.to(dtype=torch.float64).mean(dim=0, keepdim=True)
    denominator = (test_target.to(dtype=torch.float64) - baseline).square().sum().clamp_min(1e-12)
    return float((1.0 - (prediction - test_target).square().sum() / denominator).detach().cpu())


def _class_probe_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    classes: int,
) -> float:
    target = torch.nn.functional.one_hot(train_labels.to(torch.long), num_classes=classes).to(
        dtype=torch.float64
    )
    prediction = _fit_linear_map(train_features, target).predict(test_features)
    return float((prediction.argmax(dim=-1) == test_labels.to(torch.long)).float().mean().cpu())


def _bit_probe_accuracy(
    train_features: Tensor,
    train_bits: Tensor,
    test_features: Tensor,
    test_bits: Tensor,
) -> float:
    prediction = _fit_linear_map(train_features, train_bits).predict(test_features)
    return float(((prediction >= 0.5) == (test_bits >= 0.5)).float().mean().cpu())


def _effective_rank(representation: Tensor) -> float:
    centered = representation.to(dtype=torch.float64) - representation.to(dtype=torch.float64).mean(
        dim=0, keepdim=True
    )
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if float(total) <= 1e-12:
        return 0.0
    probability = energy / total
    return float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()).cpu())


def _low_dim_nuisance_dominance(
    train_representation: Tensor,
    test_representation: Tensor,
    train_nuisance: Tensor,
    test_nuisance: Tensor,
) -> float:
    centered_train = train_representation.to(dtype=torch.float64) - train_representation.to(
        dtype=torch.float64
    ).mean(dim=0, keepdim=True)
    _, _, vectors = torch.linalg.svd(centered_train, full_matrices=False)
    dimensions = min(2, vectors.shape[0])
    components = vectors[:dimensions].T
    train_projection = centered_train @ components
    test_projection = (
        test_representation.to(dtype=torch.float64)
        - train_representation.to(dtype=torch.float64).mean(dim=0, keepdim=True)
    ) @ components
    return _linear_r2(train_nuisance, train_projection, test_nuisance, test_projection)


def _fixed_cpu_train(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    target: Tensor,
    train_indices: Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*(value[train_indices] for value in inputs))
        loss = torch.mean((prediction - target[train_indices]).square())
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu())


def _episode_bits(episode: int, time: int) -> Tensor:
    value = (episode * 5 + time) % 16
    return torch.tensor([(value >> bit) & 1 for bit in range(4)], dtype=torch.float32)


def _build_obsessed_nuisance_suite(generator: torch.Generator) -> _ObsessedNuisanceSuite:
    """Generate episode IDs, time, templates, and bit strings alongside real action dynamics."""

    episode_count = 8
    horizon = 11
    action_map = torch.tensor([[0.42, -0.12], [0.08, 0.37]], dtype=torch.float32)
    template_codes = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32
    )
    observations: list[Tensor] = []
    next_observations: list[Tensor] = []
    physical: list[Tensor] = []
    next_physical: list[Tensor] = []
    actions: list[Tensor] = []
    episode_ids: list[int] = []
    timestamps: list[Tensor] = []
    template_ids: list[int] = []
    bit_strings: list[Tensor] = []
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    def observation(state: Tensor, episode: int, time: int) -> Tensor:
        episode_scalar = torch.tensor([(episode / (episode_count - 1)) * 16.0 - 8.0])
        timestamp = torch.tensor([(time / horizon) * 6.0])
        template = template_codes[episode % len(template_codes)] * 9.0
        bits = _episode_bits(episode, time) * 6.0 - 3.0
        return torch.cat([state, episode_scalar, timestamp, template, bits])

    row = 0
    for episode in range(episode_count):
        state = torch.randn((2,), generator=generator, dtype=torch.float32) * 0.35
        for time in range(horizon):
            action = torch.rand((2,), generator=generator, dtype=torch.float32) * 1.4 - 0.7
            successor = 0.88 * state + action @ action_map.T
            if time < horizon - 1:
                observations.append(observation(state, episode, time))
                next_observations.append(observation(successor, episode, time + 1))
                physical.append(state)
                next_physical.append(successor)
                actions.append(action)
                episode_ids.append(episode)
                timestamps.append(torch.tensor([time / horizon], dtype=torch.float32))
                template_ids.append(episode % len(template_codes))
                bit_strings.append(_episode_bits(episode, time))
                if time < 6:
                    train_indices.append(row)
                elif time < 8:
                    validation_indices.append(row)
                else:
                    test_indices.append(row)
                row += 1
            state = successor
    return _ObsessedNuisanceSuite(
        observations=torch.stack(observations),
        next_observations=torch.stack(next_observations),
        physical=torch.stack(physical),
        next_physical=torch.stack(next_physical),
        actions=torch.stack(actions),
        episode_ids=torch.tensor(episode_ids, dtype=torch.long),
        timestamps=torch.stack(timestamps),
        template_ids=torch.tensor(template_ids, dtype=torch.long),
        bit_strings=torch.stack(bit_strings),
        train_indices=torch.tensor(train_indices, dtype=torch.long),
        validation_indices=torch.tensor(validation_indices, dtype=torch.long),
        test_indices=torch.tensor(test_indices, dtype=torch.long),
    )


def _predictable_nuisance_case(generator: torch.Generator) -> Stage0CaseResult:
    """Run the preregistered obsessed-encoder shortcut suite under fixed CPU budgets."""

    suite = _build_obsessed_nuisance_suite(generator)
    action_map = torch.tensor([[0.42, -0.12], [0.08, 0.37]], dtype=torch.float32)

    def physical_transition(state: Tensor, control: Tensor) -> Tensor:
        return 0.88 * state + control @ action_map.T

    def effect_batch(indices: Tensor, split: SplitRole, prefix: str) -> CausalEffectBatch:
        return CausalEffectBatch(
            source=suite.physical[indices],
            treated=suite.next_physical[indices],
            intervention=suite.actions[indices],
            split=split,
            example_ids=tuple(f"{prefix}-{int(index)}" for index in indices),
        )

    train = effect_batch(suite.train_indices, SplitRole.TRAIN, "nuisance-train")
    validation = effect_batch(suite.validation_indices, SplitRole.VALIDATION, "nuisance-validation")
    test = effect_batch(suite.test_indices, SplitRole.TEST, "nuisance-test")
    selection = _eligible_selection(physical_transition, train, validation)
    baseline = _test_baseline(selection, test)

    training_steps = 220
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7001)
        naive = _NaiveNextStateJEPA(suite.observations.shape[-1])
        naive_train_loss = _fixed_cpu_train(
            naive,
            (suite.observations,),
            suite.next_observations,
            suite.train_indices,
            steps=training_steps,
            learning_rate=3e-3,
        )
        torch.manual_seed(7002)
        guarded = _GuardedResidualActionJEPA()
        guarded_train_loss = _fixed_cpu_train(
            guarded,
            (suite.observations, suite.actions),
            suite.next_physical - suite.physical,
            suite.train_indices,
            steps=training_steps,
            learning_rate=3e-3,
        )

    with torch.no_grad():
        naive_representation = naive.encode(suite.observations)
        guarded_representation = guarded.encode(suite.observations, suite.actions)
        naive_prediction = naive.decode(naive_representation)
        guarded_prediction = guarded.decode(guarded_representation)

    nuisance_features = torch.cat(
        [
            torch.nn.functional.one_hot(suite.episode_ids, num_classes=8).to(torch.float32),
            suite.timestamps,
            torch.nn.functional.one_hot(suite.template_ids, num_classes=4).to(torch.float32),
            suite.bit_strings,
        ],
        dim=-1,
    )
    nuisance_inputs = {
        "episode_id": torch.nn.functional.one_hot(suite.episode_ids, num_classes=8).to(
            torch.float32
        ),
        "timestamp": suite.timestamps,
        "camera_template": torch.nn.functional.one_hot(suite.template_ids, num_classes=4).to(
            torch.float32
        ),
        "bit_string": suite.bit_strings,
    }
    train_index = suite.train_indices
    validation_index = suite.validation_indices
    test_index = suite.test_indices

    def variance_explained(representation: Tensor) -> dict[str, float]:
        return {
            name: _linear_r2(
                value[train_index],
                representation[train_index],
                value[test_index],
                representation[test_index],
            )
            for name, value in nuisance_inputs.items()
        }

    naive_variance = variance_explained(naive_representation)
    guarded_variance = variance_explained(guarded_representation)
    naive_nuisance_scores = [
        _class_probe_accuracy(
            naive_representation[train_index],
            suite.episode_ids[train_index],
            naive_representation[test_index],
            suite.episode_ids[test_index],
            classes=8,
        ),
        _class_probe_accuracy(
            naive_representation[train_index],
            suite.template_ids[train_index],
            naive_representation[test_index],
            suite.template_ids[test_index],
            classes=4,
        ),
        _bit_probe_accuracy(
            naive_representation[train_index],
            suite.bit_strings[train_index],
            naive_representation[test_index],
            suite.bit_strings[test_index],
        ),
    ]
    guarded_nuisance_scores = [
        _class_probe_accuracy(
            guarded_representation[train_index],
            suite.episode_ids[train_index],
            guarded_representation[test_index],
            suite.episode_ids[test_index],
            classes=8,
        ),
        _class_probe_accuracy(
            guarded_representation[train_index],
            suite.template_ids[train_index],
            guarded_representation[test_index],
            suite.template_ids[test_index],
            classes=4,
        ),
        _bit_probe_accuracy(
            guarded_representation[train_index],
            suite.bit_strings[train_index],
            guarded_representation[test_index],
            suite.bit_strings[test_index],
        ),
    ]
    physical_delta = suite.next_physical - suite.physical
    naive_effect_r2 = _linear_r2(
        naive_representation[train_index],
        physical_delta[train_index],
        naive_representation[test_index],
        physical_delta[test_index],
    )
    guarded_effect_r2 = _linear_r2(
        guarded_representation[train_index],
        physical_delta[train_index],
        guarded_representation[test_index],
        physical_delta[test_index],
    )
    naive_timestamp_probe_r2 = _linear_r2(
        naive_representation[train_index],
        suite.timestamps[train_index],
        naive_representation[test_index],
        suite.timestamps[test_index],
    )
    guarded_timestamp_probe_r2 = _linear_r2(
        guarded_representation[train_index],
        suite.timestamps[train_index],
        guarded_representation[test_index],
        suite.timestamps[test_index],
    )
    effect_classes = (physical_delta[:, 0] >= 0.0).to(torch.long) * 2 + (
        physical_delta[:, 1] >= 0.0
    ).to(torch.long)
    naive_effect_accuracy = _class_probe_accuracy(
        naive_representation[train_index],
        effect_classes[train_index],
        naive_representation[test_index],
        effect_classes[test_index],
        classes=4,
    )
    guarded_effect_accuracy = _class_probe_accuracy(
        guarded_representation[train_index],
        effect_classes[train_index],
        guarded_representation[test_index],
        effect_classes[test_index],
        classes=4,
    )
    nuisance_removed_observation = suite.observations.clone()
    nuisance_removed_observation[:, 2:] = suite.observations[train_index, 2:].mean(
        dim=0, keepdim=True
    )
    naive_test_mse = _mse(naive_prediction[test_index], suite.next_observations[test_index])
    naive_validation_mse = _mse(
        naive_prediction[validation_index], suite.next_observations[validation_index]
    )
    guarded_test_mse = _mse(guarded_prediction[test_index], physical_delta[test_index])
    guarded_validation_mse = _mse(
        guarded_prediction[validation_index], physical_delta[validation_index]
    )
    with torch.no_grad():
        naive_removed_mse = _mse(
            naive(nuisance_removed_observation[test_index]), suite.next_observations[test_index]
        )
        guarded_removed_mse = _mse(
            guarded(nuisance_removed_observation[test_index], suite.actions[test_index]),
            physical_delta[test_index],
        )

    norm_control = norm_matched_random_direction_control(suite.actions[test_index], seed=917)
    covariance_control = matched_norm_covariance_action_control(suite.actions[test_index], seed=918)
    with torch.no_grad():
        guarded_norm_prediction = guarded(suite.observations[test_index], norm_control.actions)
        guarded_covariance_prediction = guarded(
            suite.observations[test_index], covariance_control.actions
        )
    naive_action_sensitivity = 0.0
    guarded_norm_sensitivity = float(
        torch.linalg.vector_norm(guarded_prediction[test_index] - guarded_norm_prediction, dim=-1)
        .mean()
        .cpu()
    )
    guarded_covariance_sensitivity = float(
        torch.linalg.vector_norm(
            guarded_prediction[test_index] - guarded_covariance_prediction, dim=-1
        )
        .mean()
        .cpu()
    )
    baseline_replay_mse = _mse(
        reconstruct_treated(test.source, baseline, torch.zeros_like(baseline)), test.treated
    )
    naive_low_dim = _low_dim_nuisance_dominance(
        naive_representation[train_index],
        naive_representation[test_index],
        nuisance_features[train_index],
        nuisance_features[test_index],
    )
    guarded_low_dim = _low_dim_nuisance_dominance(
        guarded_representation[train_index],
        guarded_representation[test_index],
        nuisance_features[train_index],
        nuisance_features[test_index],
    )
    naive_nuisance_probe = float(sum(naive_nuisance_scores) / len(naive_nuisance_scores))
    guarded_nuisance_probe = float(sum(guarded_nuisance_scores) / len(guarded_nuisance_scores))
    passed = (
        selection.baseline_star.metadata.name == "exact_jvp"
        and baseline_replay_mse < 1e-12
        and naive_nuisance_probe > 0.60
        and naive_low_dim > 0.55
        and naive_action_sensitivity == 0.0
        and guarded_test_mse < 0.02
        and guarded_effect_r2 > 0.65
        and guarded_effect_accuracy > naive_effect_accuracy + 0.15
        and guarded_norm_sensitivity > 0.10
        and guarded_covariance_sensitivity > 0.10
        and naive_removed_mse > naive_test_mse * 1.05
        and guarded_removed_mse < guarded_test_mse * 2.5 + 1e-4
        and norm_control.max_norm_difference < 1e-6
        and covariance_control.max_norm_difference < 1e-6
        and covariance_control.max_covariance_difference < 1e-6
    )
    return Stage0CaseResult(
        case_id="predictable_nuisance_guard",
        status="SMOKE_VALIDATED" if passed else "NEGATIVE_RESULT",
        metrics={
            "selected_baseline": selection.baseline_star.metadata.name,
            "observable_target_contract": "physical_state_delta_and_frozen_known_nuisance_mask",
            "planted_nuisances": "episode_prompt_id|slow_timestamp|persistent_camera_template|predictable_bit_string",
            "naive_training_steps": training_steps,
            "guarded_training_steps": training_steps,
            "naive_train_loss": naive_train_loss,
            "guarded_train_loss": guarded_train_loss,
            "baseline_replay_mse": baseline_replay_mse,
            "naive_next_state_mse": naive_test_mse,
            "naive_validation_next_state_mse": naive_validation_mse,
            "naive_next_state_mse_after_nuisance_removal": naive_removed_mse,
            "guarded_residual_mse": guarded_test_mse,
            "guarded_validation_residual_mse": guarded_validation_mse,
            "guarded_residual_mse_after_nuisance_removal": guarded_removed_mse,
            "nuisance_removal_mode": "replace_frozen_nuisance_channels_with_train_mean",
            "naive_latent_variance": float(
                naive_representation[test_index].var(dim=0, unbiased=False).mean().cpu()
            ),
            "guarded_latent_variance": float(
                guarded_representation[test_index].var(dim=0, unbiased=False).mean().cpu()
            ),
            "naive_effective_rank": _effective_rank(naive_representation[test_index]),
            "guarded_effective_rank": _effective_rank(guarded_representation[test_index]),
            "naive_nuisance_probe_accuracy": naive_nuisance_probe,
            "guarded_nuisance_probe_accuracy": guarded_nuisance_probe,
            "naive_timestamp_probe_r2": naive_timestamp_probe_r2,
            "guarded_timestamp_probe_r2": guarded_timestamp_probe_r2,
            "naive_causal_effect_probe_r2": naive_effect_r2,
            "guarded_causal_effect_probe_r2": guarded_effect_r2,
            "naive_causal_effect_probe_accuracy": naive_effect_accuracy,
            "guarded_causal_effect_probe_accuracy": guarded_effect_accuracy,
            "naive_variance_explained_episode_id": naive_variance["episode_id"],
            "naive_variance_explained_timestamp": naive_variance["timestamp"],
            "naive_variance_explained_camera_template": naive_variance["camera_template"],
            "naive_variance_explained_bit_string": naive_variance["bit_string"],
            "guarded_variance_explained_episode_id": guarded_variance["episode_id"],
            "guarded_variance_explained_timestamp": guarded_variance["timestamp"],
            "guarded_variance_explained_camera_template": guarded_variance["camera_template"],
            "guarded_variance_explained_bit_string": guarded_variance["bit_string"],
            "naive_low_dim_nuisance_dominance_r2": naive_low_dim,
            "guarded_low_dim_nuisance_dominance_r2": guarded_low_dim,
            "naive_conditional_action_sensitivity": naive_action_sensitivity,
            "guarded_norm_matched_action_sensitivity": guarded_norm_sensitivity,
            "guarded_covariance_matched_action_sensitivity": guarded_covariance_sensitivity,
            "norm_matched_control_max_norm_difference": norm_control.max_norm_difference,
            "covariance_matched_control_max_norm_difference": covariance_control.max_norm_difference,
            "covariance_matched_control_max_covariance_difference": covariance_control.max_covariance_difference,
            "passed_nuisance_guard": passed,
        },
        selection_sha256=selection.record.sha256,
    )


def run_stage0_benchmark(*, seed: int = 20260802) -> Stage0BenchmarkResult:
    """Run all four small deterministic, CPU-scale Stage-0 cases.

    No target encoder, human label, model download, or protected Qwen execution is involved.
    ``SMOKE_VALIDATED`` means the benchmark's expected falsification behavior occurred; it is
    not a mechanism claim about either research track.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cases = (
        _linear_case(generator),
        _quadratic_case(generator),
        _nonlinear_compositional_case(generator),
        _predictable_nuisance_case(generator),
    )
    expected = {
        "linear_zero_residual": "NEGATIVE_RESULT",
        "quadratic_hvp_zero_false_discovery": "NEGATIVE_RESULT",
        "nonlinear_compositional_residual": "SMOKE_VALIDATED",
        "predictable_nuisance_guard": "SMOKE_VALIDATED",
    }
    status = (
        "SMOKE_VALIDATED"
        if all(result.status == expected[result.case_id] for result in cases)
        else "NEGATIVE_RESULT"
    )
    return Stage0BenchmarkResult(seed=seed, status=status, cases=cases)


__all__ = [
    "Stage0BenchmarkResult",
    "Stage0CaseResult",
    "TrainOnlyPolynomialResidualHead",
    "run_stage0_benchmark",
]
