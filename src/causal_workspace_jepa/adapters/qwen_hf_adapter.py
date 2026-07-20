"""Hugging Face Qwen adapter boundary.

Real Qwen instrumentation is blocked on the CPU VPS because it requires model
weights, Transformers, selected-layer activation storage, and suitable GPU
memory. This module intentionally fails loudly instead of pretending Ollama or
mock models expose Qwen hidden states.
"""

from __future__ import annotations


class QwenHFAdapterUnavailable(RuntimeError):
    pass


class QwenHFAdapter:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise QwenHFAdapterUnavailable(
            "BLOCKED_RESOURCE: Hugging Face Qwen instrumentation requires gpu_12gb or larger"
        )
