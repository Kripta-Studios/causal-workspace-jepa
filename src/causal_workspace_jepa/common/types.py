"""Core typed interfaces shared by world-model and LLM tracks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

Tensor = np.ndarray


@dataclass(frozen=True)
class LatentState:
    tensor: Tensor
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldModelOutput:
    predicted_latents: Tensor
    uncertainty: Tensor | None
    decoded_state: Mapping[str, Tensor] | None
    intermediates: Mapping[str, Tensor]
    action_embeddings: Tensor
    cost_features: Tensor | None


class ActionConditionedWorldModel(Protocol):
    def encode(self, observation: Tensor) -> LatentState: ...

    def predict(
        self,
        latent: LatentState,
        actions: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> WorldModelOutput: ...

    def named_activation_points(self) -> Sequence[str]: ...

    def decode_state(self, latent: LatentState) -> Mapping[str, Tensor]: ...


InterventionOperation = Literal[
    "zero",
    "mean",
    "resample",
    "patch",
    "replace_feature",
    "steer",
    "project_out",
    "scale",
    "suppress_module",
]


@dataclass(frozen=True)
class InterventionSpec:
    site: str
    operation: InterventionOperation
    positions: tuple[int, ...] | None = None
    feature_ids: tuple[int, ...] | None = None
    magnitude: float = 1.0
    donor_example_id: str | None = None
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionSpec":
        return cls(
            site=str(payload["site"]),
            operation=payload["operation"],  # type: ignore[arg-type]
            positions=tuple(payload["positions"]) if payload.get("positions") is not None else None,
            feature_ids=tuple(payload["feature_ids"]) if payload.get("feature_ids") is not None else None,
            magnitude=float(payload.get("magnitude", 1.0)),
            donor_example_id=payload.get("donor_example_id"),
            seed=int(payload.get("seed", 0)),
        )


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor
    prompts: tuple[str, ...]
    tokens: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class LLMRun:
    logits: Tensor
    activations: Mapping[str, Tensor]
    token_batch: TokenBatch
    metadata: Mapping[str, Any]


class InstrumentedCausalLM(Protocol):
    def tokenize(self, prompts: Sequence[str]) -> TokenBatch: ...

    def forward_with_cache(self, batch: TokenBatch, sites: Sequence[str]) -> LLMRun: ...

    def forward_with_intervention(
        self,
        batch: TokenBatch,
        intervention: InterventionSpec,
        sites: Sequence[str],
    ) -> LLMRun: ...

    def named_activation_points(self) -> Sequence[str]: ...
