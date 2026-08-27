"""GPU-mandatory collection rules.

Qwen tokenizer/model tests must not run on CPU GitHub runners and must never
download weights. Missing CUDA is ``SKIPPED_RESOURCE``, never a silent CPU
fallback.
"""

from __future__ import annotations

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("gpu") is None:
        return
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SKIPPED_RESOURCE: CUDA GPU required")
