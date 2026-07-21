"""Adapter and exact one-step GRU decomposition for official EB-JEPA.

The upstream action-conditioned example represents a frame with an Impala
encoder and advances the representation with a one-layer ``torch.nn.GRU``.
This module deliberately depends on that public object contract rather than
vendoring upstream code.  It can therefore be unit-tested with a small object
that has the same ``encoder``/``action_encoder``/``predictor`` attributes.

No published checkpoint result is implied by importing this adapter.  A local
checkout and checkpoint must be revision-checked separately before use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from causal_workspace_jepa.common.types import (
    InterventionSpec,
    LatentState,
    WorldModelOutput,
)

EB_JEPA_UPSTREAM_URL = "https://github.com/facebookresearch/eb_jepa.git"
EB_JEPA_UPSTREAM_REVISION = "966e61e9285b3a876f49b9774e9720d9a99a7925"
STATUS = "OFFICIAL_SOURCE_CONTRACT_VALIDATED"

_ACTIVATION_POINTS = (
    "encoder.state",
    "action.encoded",
    "predictor.reset_gate",
    "predictor.update_gate",
    "predictor.candidate_state",
    "predictor.hidden_pre_norm",
    "predictor.hidden_post_norm",
    "prediction.latent",
)


def verify_eb_jepa_checkout(
    source_root: str | Path,
    *,
    expected_revision: str = EB_JEPA_UPSTREAM_REVISION,
    require_clean: bool = True,
) -> Mapping[str, Any]:
    """Validate the immutable upstream source boundary without importing it."""

    root = Path(source_root).resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "eb_jepa").is_dir():
        raise FileNotFoundError(f"not an EB-JEPA source checkout: {root}")
    resolved = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_lines = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if resolved != expected_revision:
        raise RuntimeError(
            f"EB-JEPA revision mismatch: expected {expected_revision}, resolved {resolved}"
        )
    if require_clean and dirty_lines:
        raise RuntimeError("EB-JEPA checkout has uncommitted changes")
    return {
        "source_url": EB_JEPA_UPSTREAM_URL,
        "expected_revision": expected_revision,
        "resolved_revision": resolved,
        "clean": not dirty_lines,
        "source_root": str(root),
    }


class EBJEPAAdapter:
    """NumPy-facing adapter for the official one-layer recurrent AC Video JEPA.

    ``model`` must expose ``encoder``, ``action_encoder``, and ``predictor``.
    The predictor must expose a one-layer ``torch.nn.GRU`` as ``rnn`` and may
    expose a final normalization module as ``final_ln``.  Latents are returned
    as ``[batch, time, feature]`` arrays; upstream singleton spatial axes are
    checked and removed explicitly.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str = "cpu",
        upstream_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.upstream_metadata = dict(upstream_metadata or {})
        self._decoder: np.ndarray | None = None
        self._means: dict[str, torch.Tensor] = {}
        self._donors: dict[tuple[str, str], torch.Tensor] = {}
        self._bases: dict[str, torch.Tensor] = {}
        self._validate_model_contract()

    def _validate_model_contract(self) -> None:
        for name in ("encoder", "action_encoder", "predictor"):
            if not hasattr(self.model, name):
                raise TypeError(f"EB-JEPA model is missing {name!r}")
        predictor = self.model.predictor
        if not isinstance(getattr(predictor, "rnn", None), torch.nn.GRU):
            raise TypeError("EB-JEPA predictor.rnn must be torch.nn.GRU")
        if predictor.rnn.num_layers != 1 or predictor.rnn.bidirectional:
            raise ValueError("gate decomposition currently requires a one-layer unidirectional GRU")

    def fit_state_decoder(
        self, embeddings: np.ndarray, states: np.ndarray, ridge: float = 1e-3
    ) -> None:
        values = np.asarray(embeddings, dtype=np.float32)
        flat = values.reshape(-1, values.shape[-1])
        targets = np.asarray(states, dtype=np.float32).reshape(len(flat), -1)
        design = np.concatenate(
            [flat, np.ones((len(flat), 1), dtype=np.float32)], axis=1
        ).astype(np.float64)
        regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        self._decoder = np.linalg.solve(
            design.T @ design + regularizer, design.T @ targets.astype(np.float64)
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
            pixels = pixels[None, :, None]
        elif pixels.ndim == 4:
            pixels = pixels[:, :, None]
        if pixels.ndim != 5:
            raise ValueError("observation must have shape [C,H,W], [B,C,H,W], or [B,C,T,H,W]")
        with torch.inference_mode():
            encoded = self.model.encoder(pixels)
        latent = self._from_upstream_state(encoded)
        return LatentState(latent.cpu().numpy(), names=("encoder.state",))

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

    @torch.inference_mode()
    def _predict(
        self,
        latent: LatentState,
        actions: np.ndarray,
        intervention: InterventionSpec | None,
        return_intermediates: bool,
    ) -> WorldModelOutput:
        hidden = self._initial_hidden(latent)
        action_time = self._actions_time_major(actions)
        if action_time.shape[0] != hidden.shape[0]:
            if hidden.shape[0] == 1:
                hidden = hidden.expand(action_time.shape[0], -1).clone()
            else:
                raise ValueError("latent and action batch dimensions differ")
        encoded_actions = self.model.action_encoder(action_time.transpose(1, 2))
        if encoded_actions.ndim != 3:
            raise ValueError("action_encoder must return [B,A,T]")
        encoded_actions = encoded_actions.transpose(1, 2).contiguous()

        if intervention is not None and intervention.site == "encoder.state":
            hidden = self._apply_intervention(hidden, intervention)
        initial_hidden = hidden.clone()

        traces: dict[str, list[torch.Tensor]] = {
            name: [] for name in _ACTIVATION_POINTS if name != "encoder.state"
        }
        for step in range(encoded_actions.shape[1]):
            action = encoded_actions[:, step]
            if self._applies_at_step(intervention, "action.encoded", step):
                action = self._apply_intervention(action, intervention)
            traces["action.encoded"].append(action)
            gates = self._gru_step(hidden, action)
            for site in (
                "predictor.reset_gate",
                "predictor.update_gate",
                "predictor.candidate_state",
            ):
                if self._applies_at_step(intervention, site, step):
                    gates[site] = self._apply_intervention(gates[site], intervention)

            # Recompute the recurrent update after any gate edit.
            reset = gates["predictor.reset_gate"]
            update = gates["predictor.update_gate"]
            candidate = gates["predictor.candidate_state"]
            if self._applies_at_step(intervention, "predictor.reset_gate", step):
                candidate = self._candidate_from_reset(hidden, action, reset)
                gates["predictor.candidate_state"] = candidate
            hidden_pre_norm = (1.0 - update) * candidate + update * hidden
            if self._applies_at_step(intervention, "predictor.hidden_pre_norm", step):
                hidden_pre_norm = self._apply_intervention(hidden_pre_norm, intervention)
            final_ln = getattr(self.model.predictor, "final_ln", None)
            hidden_post_norm = final_ln(hidden_pre_norm) if final_ln is not None else hidden_pre_norm
            if self._applies_at_step(intervention, "predictor.hidden_post_norm", step):
                hidden_post_norm = self._apply_intervention(hidden_post_norm, intervention)
            if self._applies_at_step(intervention, "prediction.latent", step):
                hidden_post_norm = self._apply_intervention(hidden_post_norm, intervention)
            hidden = hidden_post_norm

            for site in (
                "predictor.reset_gate",
                "predictor.update_gate",
                "predictor.candidate_state",
            ):
                traces[site].append(gates[site])
            traces["predictor.hidden_pre_norm"].append(hidden_pre_norm)
            traces["predictor.hidden_post_norm"].append(hidden_post_norm)
            traces["prediction.latent"].append(hidden_post_norm)

        stacked = {name: torch.stack(values, dim=1) for name, values in traces.items()}
        predicted = stacked["prediction.latent"]
        decoded = None
        cost_features = None
        if self._decoder is not None:
            state = self.decode_state(LatentState(predicted.detach().cpu().numpy()))["state"]
            decoded = {"state": state}
            cost_features = state[..., :2]
        intermediates: dict[str, np.ndarray] = {}
        if return_intermediates:
            intermediates["encoder.state"] = initial_hidden.detach().cpu().numpy()
            intermediates.update(
                {name: value.detach().cpu().numpy() for name, value in stacked.items()}
            )
        return WorldModelOutput(
            predicted_latents=predicted.detach().cpu().numpy(),
            uncertainty=None,
            decoded_state=decoded,
            intermediates=intermediates,
            action_embeddings=encoded_actions.detach().cpu().numpy(),
            cost_features=cost_features,
        )

    def validate_predictor_reconstruction(
        self,
        latent: LatentState,
        actions: np.ndarray,
        *,
        atol: float = 1e-6,
    ) -> float:
        """Compare the decomposed recurrence against the unmodified upstream predictor."""

        hidden = self._initial_hidden(latent)
        action_time = self._actions_time_major(actions)
        if hidden.shape[0] == 1 and action_time.shape[0] != 1:
            hidden = hidden.expand(action_time.shape[0], -1).clone()
        encoded = self.model.action_encoder(action_time.transpose(1, 2)).transpose(1, 2)
        maximum = 0.0
        with torch.inference_mode():
            for step in range(encoded.shape[1]):
                upstream_state = hidden[:, :, None, None, None]
                upstream_action = encoded[:, step, :, None]
                expected = self.model.predictor(upstream_state, upstream_action).flatten(1)
                observed = self._gru_step(hidden, encoded[:, step])["predictor.hidden_pre_norm"]
                final_ln = getattr(self.model.predictor, "final_ln", None)
                if final_ln is not None:
                    observed = final_ln(observed)
                error = float((expected - observed).abs().max().item())
                maximum = max(maximum, error)
                hidden = expected
        if maximum > atol:
            raise RuntimeError(
                f"EB-JEPA GRU decomposition mismatch {maximum:.3e} exceeds atol={atol:.3e}"
            )
        return maximum

    def _gru_step(self, hidden: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        rnn = self.model.predictor.rnn
        input_terms = F.linear(action, rnn.weight_ih_l0, rnn.bias_ih_l0)
        hidden_terms = F.linear(hidden, rnn.weight_hh_l0, rnn.bias_hh_l0)
        input_reset, input_update, input_candidate = input_terms.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_candidate = hidden_terms.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset * hidden_candidate)
        hidden_pre_norm = (1.0 - update) * candidate + update * hidden
        return {
            "predictor.reset_gate": reset,
            "predictor.update_gate": update,
            "predictor.candidate_state": candidate,
            "predictor.hidden_pre_norm": hidden_pre_norm,
        }

    def _candidate_from_reset(
        self, hidden: torch.Tensor, action: torch.Tensor, reset: torch.Tensor
    ) -> torch.Tensor:
        rnn = self.model.predictor.rnn
        input_terms = F.linear(action, rnn.weight_ih_l0, rnn.bias_ih_l0)
        hidden_terms = F.linear(hidden, rnn.weight_hh_l0, rnn.bias_hh_l0)
        input_candidate = input_terms.chunk(3, dim=-1)[2]
        hidden_candidate = hidden_terms.chunk(3, dim=-1)[2]
        return torch.tanh(input_candidate + reset * hidden_candidate)

    def _initial_hidden(self, latent: LatentState) -> torch.Tensor:
        value = torch.as_tensor(latent.tensor, device=self.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value[None]
        elif value.ndim == 3:
            value = value[:, -1]
        elif value.ndim == 5:
            value = self._from_upstream_state(value)[:, -1]
        if value.ndim != 2:
            raise ValueError("latent must have shape [D], [B,D], [B,T,D], or upstream [B,D,T,1,1]")
        expected = self.model.predictor.rnn.hidden_size
        if value.shape[-1] != expected:
            raise ValueError(f"latent width {value.shape[-1]} does not match GRU hidden size {expected}")
        return value

    def _actions_time_major(self, actions: np.ndarray) -> torch.Tensor:
        value = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value[None, None]
        elif value.ndim == 2:
            value = value[None]
        if value.ndim != 3:
            raise ValueError("actions must have shape [A], [T,A], [B,T,A], or [B,A,T]")
        action_dim = self.model.predictor.rnn.input_size
        if value.shape[-1] == action_dim:
            return value
        if value.shape[1] == action_dim:
            return value.transpose(1, 2).contiguous()
        raise ValueError(f"no action axis matches predictor input size {action_dim}")

    @staticmethod
    def _from_upstream_state(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("encoder must return [B,D,T,H,W]")
        if value.shape[-2:] != (1, 1):
            raise ValueError("official recurrent adapter requires singleton encoded spatial axes")
        return value[:, :, :, 0, 0].transpose(1, 2).contiguous()

    @staticmethod
    def _applies_at_step(
        intervention: InterventionSpec | None, site: str, step: int
    ) -> bool:
        return bool(
            intervention is not None
            and intervention.site == site
            and (intervention.positions is None or step in intervention.positions)
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
        return _ACTIVATION_POINTS

    def _apply_intervention(
        self, activation: torch.Tensor, spec: InterventionSpec
    ) -> torch.Tensor:
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
            raise KeyError(f"unknown EB-JEPA activation site: {site}")
