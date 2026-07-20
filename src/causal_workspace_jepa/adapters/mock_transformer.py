"""Deterministic mock CausalLM with known activation dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causal_workspace_jepa.common.types import InterventionSpec, LLMRun, TokenBatch
from causal_workspace_jepa.hooks.capture import ActivationCache
from causal_workspace_jepa.hooks.interventions import apply_intervention
from causal_workspace_jepa.hooks.names import transformer_site


@dataclass(frozen=True)
class MockTransformerConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    layers: int = 3
    max_length: int = 16
    seed: int = 0


class MockInstrumentedCausalLM:
    """Tiny transformer-like model for intervention pipeline validation."""

    def __init__(self, config: MockTransformerConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.hidden_size)
        self.embedding = rng.normal(0.0, scale, size=(config.vocab_size, config.hidden_size)).astype(np.float32)
        self.attn = rng.normal(0.0, scale, size=(config.layers, config.hidden_size, config.hidden_size)).astype(np.float32)
        self.mlp = rng.normal(0.0, scale, size=(config.layers, config.hidden_size, config.hidden_size)).astype(np.float32)
        self.unembed = rng.normal(0.0, scale, size=(config.hidden_size, config.vocab_size)).astype(np.float32)

    def tokenize(self, prompts: Sequence[str]) -> TokenBatch:
        tokenized = [prompt.split()[: self.config.max_length] for prompt in prompts]
        max_len = max(1, max(len(tokens) for tokens in tokenized))
        input_ids = np.zeros((len(prompts), max_len), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids, dtype=np.float32)
        padded_tokens: list[tuple[str, ...]] = []
        for row, tokens in enumerate(tokenized):
            padded = tokens + ["<pad>"] * (max_len - len(tokens))
            padded_tokens.append(tuple(padded))
            for column, token in enumerate(padded):
                input_ids[row, column] = _stable_token_id(token, self.config.vocab_size)
                attention_mask[row, column] = 0.0 if token == "<pad>" else 1.0
        return TokenBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompts=tuple(prompts),
            tokens=tuple(padded_tokens),
        )

    def forward_with_cache(self, batch: TokenBatch, sites: Sequence[str]) -> LLMRun:
        return self._forward(batch, sites, intervention=None)

    def forward_with_intervention(
        self,
        batch: TokenBatch,
        intervention: InterventionSpec,
        sites: Sequence[str],
    ) -> LLMRun:
        return self._forward(batch, sites, intervention=intervention)

    def named_activation_points(self) -> Sequence[str]:
        names: list[str] = []
        for layer in range(self.config.layers):
            names.extend(
                [
                    transformer_site(layer, "resid_pre"),
                    transformer_site(layer, "attn_out"),
                    transformer_site(layer, "mlp_out"),
                    transformer_site(layer, "resid_post"),
                ]
            )
        names.append("logits")
        return tuple(names)

    def _forward(
        self,
        batch: TokenBatch,
        sites: Sequence[str],
        intervention: InterventionSpec | None,
    ) -> LLMRun:
        requested = set(sites)
        cache = ActivationCache()
        resid = self.embedding[batch.input_ids] * batch.attention_mask[..., None]
        for layer in range(self.config.layers):
            resid_pre_site = transformer_site(layer, "resid_pre")
            resid = self._maybe_intervene(resid_pre_site, resid, intervention)
            if resid_pre_site in requested:
                cache.add(resid_pre_site, resid)
            causal_context = np.cumsum(resid, axis=1) / np.maximum(
                np.cumsum(batch.attention_mask, axis=1)[..., None],
                1.0,
            )
            attn_out = np.tanh(causal_context @ self.attn[layer])
            attn_site = transformer_site(layer, "attn_out")
            attn_out = self._maybe_intervene(attn_site, attn_out, intervention)
            if attn_site in requested:
                cache.add(attn_site, attn_out)
            mlp_out = np.tanh((resid + 0.5 * attn_out) @ self.mlp[layer])
            mlp_site = transformer_site(layer, "mlp_out")
            mlp_out = self._maybe_intervene(mlp_site, mlp_out, intervention)
            if mlp_site in requested:
                cache.add(mlp_site, mlp_out)
            resid = resid + 0.35 * attn_out + 0.25 * mlp_out
            resid_post_site = transformer_site(layer, "resid_post")
            resid = self._maybe_intervene(resid_post_site, resid, intervention)
            if resid_post_site in requested:
                cache.add(resid_post_site, resid)
        logits = resid @ self.unembed
        logits = self._maybe_intervene("logits", logits, intervention)
        if "logits" in requested:
            cache.add("logits", logits)
        return LLMRun(
            logits=logits,
            activations=cache.activations,
            token_batch=batch,
            metadata={
                "model": "mock_transformer",
                "hidden_activations_available": True,
                "autograd_available": False,
            },
        )

    @staticmethod
    def _maybe_intervene(
        site: str,
        activation: np.ndarray,
        intervention: InterventionSpec | None,
    ) -> np.ndarray:
        if intervention is None or intervention.site != site:
            return activation
        return apply_intervention(activation, intervention)


def build_mock_prompts(count: int) -> list[str]:
    prompts = []
    for index in range(count):
        entity = index % 8
        value = (index * 3) % 11
        filler = "." * (1 + (index % 4))
        prompts.append(f"entity_{entity} {filler} has value_{value} answer")
    return prompts


def _stable_token_id(token: str, vocab_size: int) -> int:
    value = 0
    for char in token:
        value = (value * 131 + ord(char)) % (2**32)
    return value % vocab_size
