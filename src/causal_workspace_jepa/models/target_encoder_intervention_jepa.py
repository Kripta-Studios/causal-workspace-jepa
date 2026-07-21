"""Target-encoder Intervention-JEPA for residual-state transition prediction.

Unlike the legacy supervised conditional bottleneck, this model is trained to predict an
EMA/stop-gradient embedding of the directly observed downstream residual state. Direct activation
and logit effects are used only by a separately fitted linear decoder after JEPA training.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayStandardizer:
    """Training-split-only coordinate standardizer."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray]) -> "ArrayStandardizer":
        values = np.concatenate([value.astype(np.float32) for value in arrays], axis=0)
        mean = values.mean(axis=0).astype(np.float32)
        scale = values.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values.astype(np.float32) - self.mean) / self.scale).astype(np.float32)


@dataclass
class StandardizedRidgeDecoder:
    """Small linear decoder fitted after JEPA representation training."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(
        cls, inputs: np.ndarray, targets: np.ndarray, *, ridge: float = 1.0
    ) -> "StandardizedRidgeDecoder":
        x = inputs.reshape(inputs.shape[0], -1).astype(np.float64)
        y = targets.reshape(targets.shape[0], -1).astype(np.float64)
        input_mean = x.mean(axis=0)
        input_scale = x.std(axis=0)
        input_scale[input_scale < 1e-6] = 1.0
        target_mean = y.mean(axis=0)
        target_scale = y.std(axis=0)
        target_scale[target_scale < 1e-6] = 1.0
        design = np.concatenate(
            [(x - input_mean) / input_scale, np.ones((x.shape[0], 1))], axis=1
        )
        normalized_target = (y - target_mean) / target_scale
        regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(
            design.T @ design + regularizer, design.T @ normalized_target
        )
        return cls(
            input_mean=input_mean.astype(np.float32),
            input_scale=input_scale.astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            target_scale=target_scale.astype(np.float32),
            weights=weights.astype(np.float32),
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs.reshape(inputs.shape[0], -1).astype(np.float32)
        design = np.concatenate(
            [(x - self.input_mean) / self.input_scale, np.ones((x.shape[0], 1), np.float32)],
            axis=1,
        )
        normalized = design @ self.weights
        return (normalized * self.target_scale + self.target_mean).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            target_mean=self.target_mean,
            target_scale=self.target_scale,
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StandardizedRidgeDecoder":
        with np.load(path) as payload:
            return cls(**{name: payload[name].copy() for name in payload.files})


@dataclass
class TargetEncoderInterventionJEPA:
    """EMA target-encoder JEPA over Qwen residual-state interventions."""

    source_standardizer: ArrayStandardizer
    target_standardizer: ArrayStandardizer
    delta_standardizer: ArrayStandardizer
    state: dict[str, np.ndarray]
    input_dim: int
    hidden_dim: int
    meta_dim: int
    training_metrics: dict[str, float | int]

    @classmethod
    def fit(
        cls,
        clean_source: np.ndarray,
        donor_source: np.ndarray,
        clean_target: np.ndarray,
        intervened_target: np.ndarray,
        source_delta: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        hidden_dim: int = 192,
        meta_dim: int = 32,
        steps: int = 1200,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.99,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.01,
        target_consistency_weight: float = 0.25,
        std_floor: float = 0.5,
        validation_interval: int = 10,
        patience: int = 200,
        seed: int = 0,
        device: str = "cpu",
    ) -> "TargetEncoderInterventionJEPA":
        import torch

        arrays = [clean_source, donor_source, clean_target, intervened_target, source_delta]
        if any(value.ndim != 2 for value in arrays):
            raise ValueError("target-encoder JEPA inputs must all be rank-two arrays")
        dimensions = {value.shape[1] for value in arrays}
        if len(dimensions) != 1:
            raise ValueError("all residual-state inputs must share their final dimension")
        source_standardizer = ArrayStandardizer.fit([clean_source, donor_source])
        target_standardizer = ArrayStandardizer.fit([clean_target, intervened_target])
        delta_standardizer = ArrayStandardizer.fit([source_delta])
        train_tensors = _standardized_tensors(
            source_standardizer,
            target_standardizer,
            delta_standardizer,
            clean_source,
            donor_source,
            clean_target,
            intervened_target,
            source_delta,
            device,
        )
        val_tensors = _standardized_tensors(
            source_standardizer,
            target_standardizer,
            delta_standardizer,
            *validation,
            device,
        )
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        network = _target_encoder_network(
            input_dim=int(next(iter(dimensions))), hidden_dim=hidden_dim, meta_dim=meta_dim
        ).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in network.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        best_loss = float("inf")
        best_step = 0
        best_state: dict[str, Any] | None = None
        stale = 0
        last_losses: dict[str, float] = {}
        for step in range(1, steps + 1):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = network(*train_tensors)
            alignment = torch.mean((outputs["prediction"] - outputs["target"]) ** 2)
            consistency = torch.mean(
                (outputs["online_target"] - outputs["target"]) ** 2
            )
            variance, covariance = _anti_collapse_loss(
                [
                    outputs["prediction"],
                    outputs["recipient"],
                    outputs["donor"],
                    outputs["clean_target"],
                    outputs["online_target"],
                ],
                std_floor=std_floor,
            )
            loss = (
                alignment
                + target_consistency_weight * consistency
                + variance_weight * variance
                + covariance_weight * covariance
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("target-encoder JEPA produced a non-finite training loss")
            loss.backward()
            optimizer.step()
            network.update_target(ema_decay)
            last_losses = {
                "alignment_loss": float(alignment.detach().cpu()),
                "target_consistency_loss": float(consistency.detach().cpu()),
                "variance_loss": float(variance.detach().cpu()),
                "covariance_loss": float(covariance.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
            if step % validation_interval != 0 and step != steps:
                continue
            network.eval()
            with torch.no_grad():
                val_outputs = network(*val_tensors)
                val_loss = float(
                    torch.mean(
                        (val_outputs["prediction"] - val_outputs["target"]) ** 2
                    ).cpu()
                )
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += validation_interval
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("target-encoder JEPA produced no finite validation checkpoint")
        training_metrics: dict[str, float | int] = {
            **last_losses,
            "best_validation_alignment": best_loss,
            "best_step": best_step,
            "steps_completed": step,
        }
        return cls(
            source_standardizer=source_standardizer,
            target_standardizer=target_standardizer,
            delta_standardizer=delta_standardizer,
            state={name: value.numpy() for name, value in best_state.items()},
            input_dim=int(next(iter(dimensions))),
            hidden_dim=hidden_dim,
            meta_dim=meta_dim,
            training_metrics=training_metrics,
        )

    def predict_latent(
        self,
        clean_source: np.ndarray,
        donor_source: np.ndarray,
        clean_target: np.ndarray,
        source_delta: np.ndarray,
    ) -> np.ndarray:
        import torch

        network = self._network()
        tensors = (
            torch.from_numpy(self.source_standardizer.transform(clean_source)),
            torch.from_numpy(self.source_standardizer.transform(donor_source)),
            torch.from_numpy(self.target_standardizer.transform(clean_target)),
            torch.from_numpy(self.delta_standardizer.transform(source_delta)),
        )
        with torch.no_grad():
            prediction = network.predict(*tensors)
        return prediction.cpu().numpy().astype(np.float32)

    def target_latent(self, intervened_target: np.ndarray) -> np.ndarray:
        import torch

        network = self._network()
        values = torch.from_numpy(self.target_standardizer.transform(intervened_target))
        with torch.no_grad():
            target = network.target_encoder(values)
        return target.cpu().numpy().astype(np.float32)

    def collapse_metrics(
        self,
        clean_source: np.ndarray,
        donor_source: np.ndarray,
        clean_target: np.ndarray,
        intervened_target: np.ndarray,
        source_delta: np.ndarray,
    ) -> dict[str, float]:
        predicted = self.predict_latent(
            clean_source, donor_source, clean_target, source_delta
        )
        target = self.target_latent(intervened_target)
        return {
            "predicted_mean_dimension_std": float(predicted.std(axis=0).mean()),
            "target_mean_dimension_std": float(target.std(axis=0).mean()),
            "predicted_effective_rank": _effective_rank(predicted),
            "target_effective_rank": _effective_rank(target),
        }

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "source_mean": self.source_standardizer.mean,
            "source_scale": self.source_standardizer.scale,
            "target_mean": self.target_standardizer.mean,
            "target_scale": self.target_standardizer.scale,
            "delta_mean": self.delta_standardizer.mean,
            "delta_scale": self.delta_standardizer.scale,
            "dimensions": np.asarray(
                [self.input_dim, self.hidden_dim, self.meta_dim], dtype=np.int64
            ),
            "training_metrics_json": np.frombuffer(
                json.dumps(self.training_metrics, sort_keys=True).encode("utf-8"), dtype=np.uint8
            ),
        }
        payload.update({f"state::{name}": value for name, value in self.state.items()})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "TargetEncoderInterventionJEPA":
        with np.load(path) as payload:
            dimensions = payload["dimensions"].tolist()
            training_metrics = json.loads(
                payload["training_metrics_json"].tobytes().decode("utf-8")
            )
            state = {
                name.removeprefix("state::"): payload[name].copy()
                for name in payload.files
                if name.startswith("state::")
            }
            return cls(
                source_standardizer=ArrayStandardizer(
                    payload["source_mean"].copy(), payload["source_scale"].copy()
                ),
                target_standardizer=ArrayStandardizer(
                    payload["target_mean"].copy(), payload["target_scale"].copy()
                ),
                delta_standardizer=ArrayStandardizer(
                    payload["delta_mean"].copy(), payload["delta_scale"].copy()
                ),
                state=state,
                input_dim=int(dimensions[0]),
                hidden_dim=int(dimensions[1]),
                meta_dim=int(dimensions[2]),
                training_metrics=training_metrics,
            )

    def _network(self) -> Any:
        import torch

        network = _target_encoder_network(self.input_dim, self.hidden_dim, self.meta_dim)
        network.load_state_dict(
            {name: torch.from_numpy(value) for name, value in self.state.items()}
        )
        network.eval()
        return network


def _standardized_tensors(
    source_standardizer: ArrayStandardizer,
    target_standardizer: ArrayStandardizer,
    delta_standardizer: ArrayStandardizer,
    clean_source: np.ndarray,
    donor_source: np.ndarray,
    clean_target: np.ndarray,
    intervened_target: np.ndarray,
    source_delta: np.ndarray,
    device: str,
) -> tuple[Any, ...]:
    import torch

    return (
        torch.as_tensor(source_standardizer.transform(clean_source), device=device),
        torch.as_tensor(source_standardizer.transform(donor_source), device=device),
        torch.as_tensor(target_standardizer.transform(clean_target), device=device),
        torch.as_tensor(target_standardizer.transform(intervened_target), device=device),
        torch.as_tensor(delta_standardizer.transform(source_delta), device=device),
    )


def _target_encoder_network(input_dim: int, hidden_dim: int, meta_dim: int) -> Any:
    import torch

    class StateEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, meta_dim),
                torch.nn.LayerNorm(meta_dim),
            )

        def forward(self, values: Any) -> Any:
            return self.layers(values)

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = StateEncoder()
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.intervention_encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, meta_dim),
                torch.nn.LayerNorm(meta_dim),
            )
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(meta_dim * 6, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, meta_dim),
                torch.nn.LayerNorm(meta_dim),
            )

        def predict(
            self, recipient: Any, donor: Any, clean_target: Any, source_delta: Any
        ) -> Any:
            recipient_latent = self.online_encoder(recipient)
            donor_latent = self.online_encoder(donor)
            clean_target_latent = self.online_encoder(clean_target)
            delta_latent = self.intervention_encoder(source_delta)
            features = torch.cat(
                [
                    recipient_latent,
                    donor_latent,
                    clean_target_latent,
                    delta_latent,
                    donor_latent - recipient_latent,
                    clean_target_latent * delta_latent,
                ],
                dim=-1,
            )
            return self.predictor(features)

        def forward(
            self,
            recipient: Any,
            donor: Any,
            clean_target: Any,
            intervened_target: Any,
            source_delta: Any,
        ) -> dict[str, Any]:
            recipient_latent = self.online_encoder(recipient)
            donor_latent = self.online_encoder(donor)
            clean_target_latent = self.online_encoder(clean_target)
            delta_latent = self.intervention_encoder(source_delta)
            prediction = self.predictor(
                torch.cat(
                    [
                        recipient_latent,
                        donor_latent,
                        clean_target_latent,
                        delta_latent,
                        donor_latent - recipient_latent,
                        clean_target_latent * delta_latent,
                    ],
                    dim=-1,
                )
            )
            online_target = self.online_encoder(intervened_target)
            with torch.no_grad():
                target = self.target_encoder(intervened_target)
            return {
                "prediction": prediction,
                "target": target,
                "online_target": online_target,
                "recipient": recipient_latent,
                "donor": donor_latent,
                "clean_target": clean_target_latent,
            }

        @torch.no_grad()
        def update_target(self, decay: float) -> None:
            for online, target in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters(), strict=True
            ):
                target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)

    return Network()


def _anti_collapse_loss(latents: list[Any], *, std_floor: float) -> tuple[Any, Any]:
    import torch

    variance_losses = []
    covariance_losses = []
    for latent in latents:
        centered = latent - latent.mean(dim=0, keepdim=True)
        standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        variance_losses.append(torch.relu(std_floor - standard_deviation).mean())
        covariance = centered.T @ centered / max(latent.shape[0] - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        covariance_losses.append(off_diagonal.square().sum() / latent.shape[1])
    return torch.stack(variance_losses).mean(), torch.stack(covariance_losses).mean()


def _effective_rank(values: np.ndarray) -> float:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    power = singular_values**2
    if float(power.sum()) <= 1e-12:
        return 0.0
    probability = power / power.sum()
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
    return float(np.exp(entropy))


__all__ = [
    "ArrayStandardizer",
    "StandardizedRidgeDecoder",
    "TargetEncoderInterventionJEPA",
]
