"""Intervention-conditioned JEPA-style predictors for CPU smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LayerTransitionInterventionJEPA:
    weights: np.ndarray
    target_mean: np.ndarray

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        intervention_features: np.ndarray,
        targets: np.ndarray,
        *,
        ridge: float = 1e-4,
    ) -> "LayerTransitionInterventionJEPA":
        design = _design(context, intervention_features)
        target = targets.reshape(targets.shape[0], -1).astype(np.float64)
        xtx = design.T @ design
        weights = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), design.T @ target)
        return cls(weights=weights.astype(np.float32), target_mean=target.mean(axis=0).astype(np.float32))

    def predict(self, context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
        return (_design(context, intervention_features) @ self.weights).astype(np.float32)

    def mse(self, context: np.ndarray, intervention_features: np.ndarray, targets: np.ndarray) -> float:
        target = targets.reshape(targets.shape[0], -1)
        return float(np.mean((self.predict(context, intervention_features) - target) ** 2))

    def mean_baseline_mse(self, targets: np.ndarray) -> float:
        target = targets.reshape(targets.shape[0], -1)
        prediction = np.broadcast_to(self.target_mean, target.shape)
        return float(np.mean((prediction - target) ** 2))


@dataclass
class LinearContextBaseline:
    weights: np.ndarray

    @classmethod
    def fit(cls, context: np.ndarray, targets: np.ndarray, ridge: float = 1e-4) -> "LinearContextBaseline":
        design = _with_bias(context.reshape(context.shape[0], -1).astype(np.float64))
        target = targets.reshape(targets.shape[0], -1).astype(np.float64)
        xtx = design.T @ design
        weights = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), design.T @ target)
        return cls(weights=weights.astype(np.float32))

    def predict(self, context: np.ndarray) -> np.ndarray:
        design = _with_bias(context.reshape(context.shape[0], -1).astype(np.float64))
        return (design @ self.weights).astype(np.float32)

    def mse(self, context: np.ndarray, targets: np.ndarray) -> float:
        target = targets.reshape(targets.shape[0], -1)
        return float(np.mean((self.predict(context) - target) ** 2))


@dataclass
class RidgeInterventionPredictor:
    """Standardized linear or bilinear intervention-effect predictor."""

    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    mode: str

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        intervention_features: np.ndarray,
        targets: np.ndarray,
        *,
        mode: str,
        ridge: float = 1e-3,
    ) -> "RidgeInterventionPredictor":
        features = _intervention_design(context, intervention_features, mode=mode)
        target = targets.reshape(targets.shape[0], -1).astype(np.float64)
        feature_mean = features.mean(axis=0)
        feature_scale = features.std(axis=0)
        feature_scale[feature_scale < 1e-6] = 1.0
        target_mean = target.mean(axis=0)
        target_scale = target.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
        design = _with_bias((features - feature_mean) / feature_scale)
        normalized_target = (target - target_mean) / target_scale
        regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        if design.shape[1] > design.shape[0]:
            weights = design.T @ np.linalg.solve(
                design @ design.T + ridge * np.eye(design.shape[0]),
                normalized_target,
            )
        else:
            weights = np.linalg.solve(
                design.T @ design + regularizer,
                design.T @ normalized_target,
            )
        return cls(
            weights=weights.astype(np.float32),
            feature_mean=feature_mean.astype(np.float32),
            feature_scale=feature_scale.astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            target_scale=target_scale.astype(np.float32),
            mode=mode,
        )

    def predict(self, context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
        features = _intervention_design(context, intervention_features, mode=self.mode)
        design = _with_bias((features - self.feature_mean) / self.feature_scale)
        normalized = design @ self.weights
        return (normalized * self.target_scale + self.target_mean).astype(np.float32)


@dataclass
class TinyMLPInterventionPredictor:
    """A trained one-hidden-layer CPU MLP for intervention effects."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    input_weights: np.ndarray
    input_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        intervention_features: np.ndarray,
        targets: np.ndarray,
        *,
        hidden_dim: int = 64,
        steps: int = 400,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> "TinyMLPInterventionPredictor":
        import torch

        inputs = np.concatenate(
            [context.reshape(context.shape[0], -1), intervention_features],
            axis=1,
        ).astype(np.float32)
        target = targets.reshape(targets.shape[0], -1).astype(np.float32)
        input_mean = inputs.mean(axis=0)
        input_scale = inputs.std(axis=0)
        input_scale[input_scale < 1e-6] = 1.0
        target_mean = target.mean(axis=0)
        target_scale = target.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
        x = torch.from_numpy((inputs - input_mean) / input_scale)
        y = torch.from_numpy((target - target_mean) / target_scale)
        torch.manual_seed(seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, y.shape[1]),
        )
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((network(x) - y) ** 2)
            loss.backward()
            optimizer.step()
        first = network[0]
        second = network[2]
        return cls(
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            input_weights=first.weight.detach().cpu().numpy().T,
            input_bias=first.bias.detach().cpu().numpy(),
            output_weights=second.weight.detach().cpu().numpy().T,
            output_bias=second.bias.detach().cpu().numpy(),
        )

    def predict(self, context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
        inputs = np.concatenate(
            [context.reshape(context.shape[0], -1), intervention_features],
            axis=1,
        ).astype(np.float32)
        normalized = (inputs - self.input_mean) / self.input_scale
        hidden = np.tanh(normalized @ self.input_weights + self.input_bias)
        output = hidden @ self.output_weights + self.output_bias
        return (output * self.target_scale + self.target_mean).astype(np.float32)


@dataclass
class NeuralInterventionJEPA:
    """Two-branch nonlinear JEPA predictor for downstream intervention effects."""

    context_mean: np.ndarray
    context_scale: np.ndarray
    intervention_mean: np.ndarray
    intervention_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    state: dict[str, np.ndarray]
    context_dim: int
    intervention_dim: int
    target_dim: int
    hidden_dim: int
    meta_dim: int
    validation_mse: float

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        interventions: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        hidden_dim: int = 64,
        meta_dim: int = 32,
        steps: int = 800,
        learning_rate: float = 3e-3,
        seed: int = 0,
        device: str = "cpu",
    ) -> "NeuralInterventionJEPA":
        import torch

        context = context.reshape(context.shape[0], -1).astype(np.float32)
        interventions = interventions.reshape(interventions.shape[0], -1).astype(np.float32)
        targets = targets.reshape(targets.shape[0], -1).astype(np.float32)
        val_context, val_interventions, val_targets = validation
        val_context = val_context.reshape(val_context.shape[0], -1).astype(np.float32)
        val_interventions = val_interventions.reshape(val_interventions.shape[0], -1).astype(np.float32)
        val_targets = val_targets.reshape(val_targets.shape[0], -1).astype(np.float32)
        context_mean, context_scale = _standardizer(context)
        intervention_mean, intervention_scale = _standardizer(interventions)
        target_mean, target_scale = _standardizer(targets)
        x_context = torch.as_tensor((context - context_mean) / context_scale, device=device)
        x_intervention = torch.as_tensor(
            (interventions - intervention_mean) / intervention_scale, device=device
        )
        y = torch.as_tensor((targets - target_mean) / target_scale, device=device)
        vx_context = torch.as_tensor((val_context - context_mean) / context_scale, device=device)
        vx_intervention = torch.as_tensor(
            (val_interventions - intervention_mean) / intervention_scale, device=device
        )
        vy = torch.as_tensor((val_targets - target_mean) / target_scale, device=device)
        torch.manual_seed(seed)
        network = _InterventionJEPANetwork(
            context.shape[1], interventions.shape[1], targets.shape[1], hidden_dim, meta_dim
        ).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience = max(50, steps // 8)
        stale = 0
        for _ in range(steps):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = network(x_context, x_intervention)
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            optimizer.step()
            network.eval()
            with torch.no_grad():
                val_prediction, _ = network(vx_context, vx_intervention)
                val_loss = float(torch.mean((val_prediction - vy) ** 2).cpu())
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Intervention-JEPA training produced no finite validation state")
        return cls(
            context_mean=context_mean,
            context_scale=context_scale,
            intervention_mean=intervention_mean,
            intervention_scale=intervention_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            state={name: value.numpy() for name, value in best_state.items()},
            context_dim=context.shape[1],
            intervention_dim=interventions.shape[1],
            target_dim=targets.shape[1],
            hidden_dim=hidden_dim,
            meta_dim=meta_dim,
            validation_mse=best_loss,
        )

    def predict(self, context: np.ndarray, interventions: np.ndarray) -> np.ndarray:
        import torch

        network = self._network()
        with torch.no_grad():
            prediction, _ = network(
                torch.from_numpy(
                    (context.reshape(context.shape[0], -1) - self.context_mean) / self.context_scale
                ).float(),
                torch.from_numpy(
                    (interventions.reshape(interventions.shape[0], -1) - self.intervention_mean)
                    / self.intervention_scale
                ).float(),
            )
        return (prediction.numpy() * self.target_scale + self.target_mean).astype(np.float32)

    def encode_meta(self, context: np.ndarray, interventions: np.ndarray) -> np.ndarray:
        import torch

        network = self._network()
        with torch.no_grad():
            _, meta = network(
                torch.from_numpy(
                    (context.reshape(context.shape[0], -1) - self.context_mean) / self.context_scale
                ).float(),
                torch.from_numpy(
                    (interventions.reshape(interventions.shape[0], -1) - self.intervention_mean)
                    / self.intervention_scale
                ).float(),
            )
        return meta.numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "context_mean": self.context_mean,
            "context_scale": self.context_scale,
            "intervention_mean": self.intervention_mean,
            "intervention_scale": self.intervention_scale,
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "dimensions": np.asarray(
                [self.context_dim, self.intervention_dim, self.target_dim, self.hidden_dim, self.meta_dim],
                dtype=np.int64,
            ),
            "validation_mse": np.asarray([self.validation_mse], dtype=np.float64),
        }
        payload.update({f"state::{name}": value for name, value in self.state.items()})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralInterventionJEPA":
        with np.load(path) as payload:
            dimensions = payload["dimensions"].tolist()
            state = {
                name.removeprefix("state::"): payload[name].copy()
                for name in payload.files
                if name.startswith("state::")
            }
            return cls(
                context_mean=payload["context_mean"].copy(),
                context_scale=payload["context_scale"].copy(),
                intervention_mean=payload["intervention_mean"].copy(),
                intervention_scale=payload["intervention_scale"].copy(),
                target_mean=payload["target_mean"].copy(),
                target_scale=payload["target_scale"].copy(),
                state=state,
                context_dim=int(dimensions[0]),
                intervention_dim=int(dimensions[1]),
                target_dim=int(dimensions[2]),
                hidden_dim=int(dimensions[3]),
                meta_dim=int(dimensions[4]),
                validation_mse=float(payload["validation_mse"][0]),
            )

    def _network(self) -> Any:
        import torch

        network = _InterventionJEPANetwork(
            self.context_dim,
            self.intervention_dim,
            self.target_dim,
            self.hidden_dim,
            self.meta_dim,
        )
        network.load_state_dict({name: torch.from_numpy(value) for name, value in self.state.items()})
        network.eval()
        return network


@dataclass
class TrajectoryInterventionJEPA:
    """Trajectory variant using the same leakage-safe predictor on flattened sequences."""

    model: NeuralInterventionJEPA
    context_shape: tuple[int, ...]
    intervention_shape: tuple[int, ...]
    target_shape: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        context: np.ndarray,
        interventions: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray],
        **kwargs: Any,
    ) -> "TrajectoryInterventionJEPA":
        model = NeuralInterventionJEPA.fit(context, interventions, targets, validation, **kwargs)
        return cls(model, context.shape[1:], interventions.shape[1:], targets.shape[1:])

    def predict(self, context: np.ndarray, interventions: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(context, interventions)
        return prediction.reshape((prediction.shape[0], *self.target_shape))


def _InterventionJEPANetwork(
    context_dim: int,
    intervention_dim: int,
    target_dim: int,
    hidden_dim: int,
    meta_dim: int,
) -> Any:
    import torch

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_encoder = torch.nn.Sequential(
                torch.nn.Linear(context_dim, hidden_dim), torch.nn.GELU()
            )
            self.intervention_encoder = torch.nn.Sequential(
                torch.nn.Linear(intervention_dim, hidden_dim), torch.nn.GELU()
            )
            self.meta = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 3, meta_dim), torch.nn.Tanh()
            )
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(meta_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, target_dim),
            )

        def forward(self, context: Any, intervention: Any) -> tuple[Any, Any]:
            context_state = self.context_encoder(context)
            intervention_state = self.intervention_encoder(intervention)
            meta = self.meta(
                torch.cat(
                    [context_state, intervention_state, context_state * intervention_state], dim=-1
                )
            )
            return self.predictor(meta), meta

    return Network()


def _standardizer(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array.mean(axis=0).astype(np.float32)
    scale = array.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def no_change_mse(targets: np.ndarray) -> float:
    target = targets.reshape(targets.shape[0], -1)
    return float(np.mean(target**2))


def effect_correlation(predicted: np.ndarray, observed: np.ndarray) -> float:
    pred = predicted.reshape(-1)
    obs = observed.reshape(-1)
    if float(np.std(pred)) <= 1e-12 or float(np.std(obs)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(pred, obs)[0, 1])


def _design(context: np.ndarray, intervention_features: np.ndarray) -> np.ndarray:
    context_flat = context.reshape(context.shape[0], -1).astype(np.float64)
    intervention = intervention_features.reshape(intervention_features.shape[0], -1).astype(
        np.float64
    )
    magnitude = intervention[:, -1:]
    interactions = context_flat * magnitude
    return _with_bias(np.concatenate([context_flat, intervention, interactions], axis=-1))


def _intervention_design(
    context: np.ndarray,
    intervention_features: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    context_flat = context.reshape(context.shape[0], -1).astype(np.float64)
    intervention = intervention_features.reshape(intervention_features.shape[0], -1).astype(
        np.float64
    )
    if mode == "linear":
        return np.concatenate([context_flat, intervention], axis=1)
    if mode == "bilinear":
        interactions = np.einsum("ni,nj->nij", context_flat, intervention).reshape(
            context_flat.shape[0], -1
        )
        return np.concatenate([context_flat, intervention, interactions], axis=1)
    raise ValueError(f"unknown intervention predictor mode: {mode}")


def _with_bias(array: np.ndarray) -> np.ndarray:
    return np.concatenate([array, np.ones((array.shape[0], 1), dtype=array.dtype)], axis=-1)
