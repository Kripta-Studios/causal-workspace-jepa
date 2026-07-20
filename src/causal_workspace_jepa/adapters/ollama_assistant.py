"""Ollama assistant boundary."""

from __future__ import annotations


class OllamaAssistant:
    hidden_activations_available = False
    autograd_available = False

    def capability_statement(self) -> str:
        return (
            "Ollama may assist with tentative labels or code review, but it does not expose "
            "hidden activations or autograd for mechanistic evidence."
        )
