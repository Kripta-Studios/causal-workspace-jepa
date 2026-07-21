"""A bounded, source-traceable small LeWorldModel reproduction.

The implementation retains the defining published recipe: an end-to-end pixel
encoder, an action embedder, an autoregressive action-conditioned predictor,
and exactly two training losses (next-embedding MSE plus SIGReg). Dimensions
are deliberately reduced for deterministic tests and a single 12 GB GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

OFFICIAL_REPOSITORY = "https://github.com/lucas-maes/le-wm"
OFFICIAL_REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


@dataclass(frozen=True)
class SmallLeWMConfig:
    image_size: int = 20
    patch_size: int = 4
    channels: int = 3
    action_dim: int = 4
    latent_dim: int = 32
    encoder_depth: int = 2
    predictor_depth: int = 2
    heads: int = 4
    mlp_dim: int = 96
    max_history: int = 8
    dropout: float = 0.0
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_projections: int = 64

    def validate(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.latent_dim % self.heads:
            raise ValueError("latent_dim must be divisible by heads")
        if min(self.encoder_depth, self.predictor_depth, self.max_history) < 1:
            raise ValueError("depths and max_history must be positive")


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by official LeWM."""

    def __init__(self, knots: int = 17, num_proj: int = 64) -> None:
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("SIGReg requires at least two knots and one projection")
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Evaluate embeddings shaped ``[time, batch, dimension]``."""

        if embeddings.ndim != 3:
            raise ValueError("SIGReg embeddings must have shape [time, batch, dimension]")
        directions = torch.randn(
            embeddings.size(-1), self.num_proj, device=embeddings.device, dtype=embeddings.dtype
        )
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        projected = (embeddings @ directions).unsqueeze(-1) * self.t.to(embeddings.dtype)
        error = (
            (projected.cos().mean(dim=-3) - self.phi.to(embeddings.dtype)).square()
            + projected.sin().mean(dim=-3).square()
        )
        statistic = (error @ self.weights.to(embeddings.dtype)) * embeddings.size(-2)
        return statistic.mean()


class _PatchEncoder(nn.Module):
    def __init__(self, config: SmallLeWMConfig) -> None:
        super().__init__()
        self.patch = nn.Conv2d(
            config.channels,
            config.latent_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        patches_per_side = config.image_size // config.patch_size
        tokens = patches_per_side**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))
        self.position = nn.Parameter(torch.randn(1, tokens + 1, config.latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.latent_dim,
                    nhead=config.heads,
                    dim_feedforward=config.mlp_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, pixels: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        tokens = self.patch(pixels).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position
        captured: dict[str, Tensor] = {"encoder.patch_tokens": tokens}
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            captured[f"encoder.block{index}"] = tokens
        latent = self.norm(tokens)[:, 0]
        captured["encoder.cls"] = latent
        return latent, captured


class _MLP(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class _ActionEmbedder(nn.Module):
    def __init__(self, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.smooth = nn.Conv1d(action_dim, latent_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.SiLU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        smoothed = self.smooth(actions.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(smoothed)


def _modulate(values: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return values * (1 + scale) + shift


class _ConditionalBlock(nn.Module):
    """Scaled AdaLN-zero predictor block matching the official topology."""

    def __init__(self, dimension: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dimension, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dimension),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 6 * dimension))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        normalized = _modulate(self.norm1(values), shift_attn, scale_attn)
        length = values.size(1)
        causal_mask = torch.triu(
            torch.ones(length, length, device=values.device, dtype=torch.bool), diagonal=1
        )
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
        )
        values = values + gate_attn * attended
        values = values + gate_mlp * self.mlp(
            _modulate(self.norm2(values), shift_mlp, scale_mlp)
        )
        return values


Intervention = Callable[[Tensor], Tensor]


class _ARPredictor(nn.Module):
    def __init__(self, config: SmallLeWMConfig) -> None:
        super().__init__()
        self.position = nn.Parameter(
            torch.randn(1, config.max_history, config.latent_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                _ConditionalBlock(
                    config.latent_dim, config.heads, config.mlp_dim, config.dropout
                )
                for _ in range(config.predictor_depth)
            ]
        )
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(
        self,
        values: Tensor,
        condition: Tensor,
        interventions: Mapping[str, Intervention] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if values.size(1) > self.position.size(1):
            raise ValueError("sequence exceeds configured max_history")
        values = values + self.position[:, : values.size(1)]
        captured: dict[str, Tensor] = {}
        for index, block in enumerate(self.blocks):
            values = block(values, condition)
            site = f"predictor.block{index}"
            if interventions is not None and site in interventions:
                values = interventions[site](values)
            captured[site] = values
        values = self.norm(values)
        return values, captured


class SmallLeWorldModel(nn.Module):
    """End-to-end pixel JEPA using the LeWM loss and conditioning pattern."""

    def __init__(self, config: SmallLeWMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = _PatchEncoder(config)
        self.projector = _MLP(config.latent_dim, config.mlp_dim)
        self.action_encoder = _ActionEmbedder(config.action_dim, config.latent_dim)
        self.predictor = _ARPredictor(config)
        self.pred_projector = _MLP(config.latent_dim, config.mlp_dim)
        self.sigreg = SIGReg(config.sigreg_knots, config.sigreg_projections)

    def encode_pixels(self, pixels: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [batch, channels, height, width]")
        latent, captured = self.encoder(pixels.float())
        latent = self.projector(latent)
        captured["projector.latent"] = latent
        return latent, captured

    def predict_embeddings(
        self,
        embeddings: Tensor,
        actions: Tensor,
        *,
        interventions: Mapping[str, Intervention] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if embeddings.ndim != 3 or actions.ndim != 3:
            raise ValueError("embeddings and actions must have [batch, time, dimension] shape")
        action_embeddings = self.action_encoder(actions)
        if interventions is not None and "action.embedding" in interventions:
            action_embeddings = interventions["action.embedding"](action_embeddings)
        predicted, captured = self.predictor(embeddings, action_embeddings, interventions)
        predicted = self.pred_projector(predicted.reshape(-1, predicted.size(-1))).reshape_as(
            predicted
        )
        if interventions is not None and "predictor.latent" in interventions:
            predicted = interventions["predictor.latent"](predicted)
        captured["action.embedding"] = action_embeddings
        captured["predictor.latent"] = predicted
        return predicted, captured

    def forward_sequence(
        self,
        pixels: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if pixels.ndim != 5 or actions.ndim != 3:
            raise ValueError("pixels/actions must have [batch,time,...] shape")
        if pixels.shape[:2] != (actions.size(0), actions.size(1) + 1):
            raise ValueError("actions must align with consecutive pixel frames")
        batch, time = pixels.shape[:2]
        embeddings, encoder_sites = self.encode_pixels(pixels.reshape(-1, *pixels.shape[2:]))
        embeddings = embeddings.reshape(batch, time, -1)
        predicted, predictor_sites = self.predict_embeddings(embeddings[:, :-1], actions)
        return {
            "embeddings": embeddings,
            "predicted_embeddings": predicted,
            "encoder_sites": encoder_sites,
            "predictor_sites": predictor_sites,
        }

    def loss(self, pixels: Tensor, actions: Tensor) -> dict[str, Tensor]:
        output = self.forward_sequence(pixels, actions)
        embeddings = output["embeddings"]
        predicted = output["predicted_embeddings"]
        assert isinstance(embeddings, Tensor) and isinstance(predicted, Tensor)
        prediction_loss = F.mse_loss(predicted, embeddings[:, 1:])
        sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
        return {
            "loss": prediction_loss + self.config.sigreg_weight * sigreg_loss,
            "prediction_loss": prediction_loss,
            "sigreg_loss": sigreg_loss,
        }

    def predict_step(
        self,
        embedding: Tensor,
        action: Tensor,
        *,
        interventions: Mapping[str, Intervention] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if embedding.ndim == 2:
            embedding = embedding[:, None, :]
        if action.ndim == 2:
            action = action[:, None, :]
        predicted, captured = self.predict_embeddings(
            embedding[:, -self.config.max_history :],
            action[:, -self.config.max_history :],
            interventions=interventions,
        )
        return predicted[:, -1], {name: value[:, -1] for name, value in captured.items()}

    def rollout(
        self,
        initial_embedding: Tensor,
        actions: Tensor,
        *,
        interventions: Mapping[str, Intervention] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if initial_embedding.ndim != 2 or actions.ndim != 3:
            raise ValueError("rollout expects [batch,latent] and [batch,horizon,action]")
        current = initial_embedding
        trajectory: list[Tensor] = []
        site_values: dict[str, list[Tensor]] = {}
        for step in range(actions.size(1)):
            current, captured = self.predict_step(
                current, actions[:, step], interventions=interventions
            )
            trajectory.append(current)
            for name, value in captured.items():
                site_values.setdefault(name, []).append(value)
        return torch.stack(trajectory, dim=1), {
            name: torch.stack(values, dim=1) for name, values in site_values.items()
        }

    def named_activation_points(self) -> Sequence[str]:
        return (
            "encoder.patch_tokens",
            *(f"encoder.block{i}" for i in range(self.config.encoder_depth)),
            "encoder.cls",
            "projector.latent",
            "action.embedding",
            *(f"predictor.block{i}" for i in range(self.config.predictor_depth)),
            "predictor.latent",
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": asdict(self.config), "state_dict": self.state_dict()}, destination
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "SmallLeWorldModel":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(SmallLeWMConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model
