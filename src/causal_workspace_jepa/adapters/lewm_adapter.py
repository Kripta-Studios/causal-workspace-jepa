"""Typed NumPy-facing adapter for the small LeWorldModel reproduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from causal_workspace_jepa.common.types import (
    InterventionSpec,
    LatentState,
    WorldModelOutput,
)
from causal_workspace_jepa.models.lewm import SmallLeWorldModel

STATUS = "FAITHFUL_SMALL_REPRODUCTION"


class LeWorldModelAdapter:
    """Expose a trained small LeWM through the repository world-model protocol."""

    def __init__(self, model: SmallLeWorldModel, *, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._decoder: np.ndarray | None = None
        self._means: dict[str, torch.Tensor] = {}
        self._donors: dict[tuple[str, str], torch.Tensor] = {}
        self._bases: dict[str, torch.Tensor] = {}

    def fit_state_decoder(self, embeddings: np.ndarray, states: np.ndarray, ridge: float = 1e-3) -> None:
        design = np.concatenate(
            [np.asarray(embeddings), np.ones((len(embeddings), 1), dtype=np.float32)], axis=1
        ).astype(np.float64)
        target = np.asarray(states, dtype=np.float64)
        regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        self._decoder = np.linalg.solve(
            design.T @ design + regularizer, design.T @ target
        ).astype(np.float32)

    def register_mean(self, site: str, value: np.ndarray | torch.Tensor) -> None:
        self._validate_site(site)
        self._means[site] = torch.as_tensor(value, device=self.device)

    def register_donor(
        self, donor_example_id: str, site: str, value: np.ndarray | torch.Tensor
    ) -> None:
        self._validate_site(site)
        self._donors[(donor_example_id, site)] = torch.as_tensor(value, device=self.device)

    def register_basis(self, site: str, value: np.ndarray | torch.Tensor) -> None:
        self._validate_site(site)
        basis = torch.as_tensor(value, device=self.device)
        if basis.ndim != 2:
            raise ValueError("basis must have shape [representation, dimension]")
        self._bases[site] = basis

    def encode(self, observation: np.ndarray) -> LatentState:
        pixels = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        if pixels.ndim == 3:
            pixels = pixels[None]
        with torch.inference_mode():
            embedding, _ = self.model.encode_pixels(pixels)
        return LatentState(embedding.cpu().numpy(), names=("projector.latent",))

    def predict(
        self,
        latent: LatentState,
        actions: np.ndarray,
        *,
        return_intermediates: bool = False,
    ) -> WorldModelOutput:
        return self._predict(latent, actions, None, return_intermediates)

    def predict_with_intervention(
        self,
        latent: LatentState,
        actions: np.ndarray,
        intervention: InterventionSpec,
        *,
        return_intermediates: bool = False,
    ) -> WorldModelOutput:
        self._validate_site(intervention.site)
        return self._predict(latent, actions, intervention, return_intermediates)

    def _predict(
        self,
        latent: LatentState,
        actions: np.ndarray,
        intervention: InterventionSpec | None,
        return_intermediates: bool,
    ) -> WorldModelOutput:
        embedding = torch.as_tensor(latent.tensor, device=self.device, dtype=torch.float32)
        if embedding.ndim == 1:
            embedding = embedding[None]
        action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if action_tensor.ndim == 2:
            action_tensor = action_tensor[None]
        callbacks = None
        if intervention is not None:
            callbacks = {
                intervention.site: lambda value: self._apply_intervention(value, intervention)
            }
        with torch.inference_mode():
            predicted, sites = self.model.rollout(
                embedding, action_tensor, interventions=callbacks
            )
        predicted_np = predicted.cpu().numpy()
        decoded = None
        cost_features = None
        if self._decoder is not None:
            decoded_state = self.decode_state(LatentState(predicted_np))["state"]
            decoded = {"state": decoded_state}
            cost_features = decoded_state[..., :2]
        return WorldModelOutput(
            predicted_latents=predicted_np,
            uncertainty=None,
            decoded_state=decoded,
            intermediates=(
                {name: value.cpu().numpy() for name, value in sites.items()}
                if return_intermediates
                else {}
            ),
            action_embeddings=action_tensor.cpu().numpy(),
            cost_features=cost_features,
        )

    def decode_state(self, latent: LatentState) -> Mapping[str, np.ndarray]:
        if self._decoder is None:
            raise RuntimeError("fit_state_decoder must be called before decode_state")
        values = np.asarray(latent.tensor, dtype=np.float32)
        flat = values.reshape(-1, values.shape[-1])
        design = np.concatenate(
            [flat, np.ones((len(flat), 1), dtype=np.float32)], axis=1
        )
        decoded = design @ self._decoder
        return {"state": decoded.reshape(*values.shape[:-1], decoded.shape[-1])}

    def named_activation_points(self) -> Sequence[str]:
        return self.model.named_activation_points()

    def _apply_intervention(self, activation: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        target = activation.clone()
        features = list(spec.feature_ids) if spec.feature_ids is not None else slice(None)
        selected = target[..., features]
        if spec.operation in {"zero", "suppress_module"}:
            updated = torch.zeros_like(selected)
        elif spec.operation == "scale":
            updated = selected * spec.magnitude
        elif spec.operation == "steer":
            updated = selected + spec.magnitude
        elif spec.operation == "mean":
            if spec.site not in self._means:
                raise ValueError("mean intervention requires a registered training mean")
            updated = self._means[spec.site].to(selected).expand_as(activation)[..., features]
        elif spec.operation in {"patch", "replace_feature", "resample"}:
            if spec.donor_example_id is None:
                raise ValueError(f"{spec.operation} requires donor_example_id")
            key = (spec.donor_example_id, spec.site)
            if key not in self._donors:
                raise KeyError(f"unregistered donor {spec.donor_example_id!r} at {spec.site}")
            updated = self._donors[key].to(selected).expand_as(activation)[..., features]
        elif spec.operation == "project_out":
            if spec.site not in self._bases:
                raise ValueError("project_out requires a registered basis")
            basis = self._bases[spec.site].to(selected)
            flat = selected.reshape(-1, selected.shape[-1])
            for vector in basis.T:
                denominator = vector.square().sum().clamp_min(1e-12)
                flat = flat - spec.magnitude * ((flat @ vector) / denominator)[:, None] * vector
            updated = flat.reshape_as(selected)
        else:  # pragma: no cover - InterventionSpec constrains operations
            raise ValueError(f"unsupported intervention operation: {spec.operation}")
        target[..., features] = updated
        return target

    def _validate_site(self, site: str) -> None:
        if site not in self.named_activation_points():
            raise KeyError(f"unknown LeWM activation site: {site}")
