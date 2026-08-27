from __future__ import annotations

from pathlib import Path


def test_gpu_tests_have_no_cpu_fallback() -> None:
    root = Path(__file__).resolve().parents[2]
    conftest = (root / "tests/conftest.py").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in conftest
    assert "SKIPPED_RESOURCE" in conftest
    cpu_ci = (root / ".github/workflows/cpu-ci.yml").read_text(encoding="utf-8")
    assert "-m \"not gpu and not slow\"" in cpu_ci or "-m 'not gpu and not slow'" in cpu_ci
    assert "HF_HUB_OFFLINE" in cpu_ci
    gpu_ci = (root / ".github/workflows/gpu-ci.yml").read_text(encoding="utf-8")
    assert "cuda.is_available()" in gpu_ci
    assert "download.pytorch.org/whl/cpu" not in gpu_ci
