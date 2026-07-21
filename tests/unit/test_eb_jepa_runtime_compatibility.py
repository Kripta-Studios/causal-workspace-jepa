from __future__ import annotations

import unittest

from causal_workspace_jepa.experiments.world_model.eb_jepa_runtime_compatibility import (
    evaluate_runtime_pair,
)


class EBJEPARuntimeCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _runtime(*, compatible: bool):
        checks = {}
        for name in ("matmul", "conv2d", "gru"):
            checks[name] = (
                {"passed": True, "finite": True}
                if compatible
                else {
                    "passed": False,
                    "error": "CUDA error: no kernel image is available for execution on the device",
                }
            )
        return {
            "python": "3.12.13",
            "torch": "2.10.0+cu128" if compatible else "2.6.0+cu126",
            "cuda_runtime": "12.8" if compatible else "12.6",
            "device": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            "capability": [12, 0],
            "arch_list": ["sm_90", "sm_120"] if compatible else ["sm_86", "sm_90"],
            "checks": checks,
        }

    def test_expected_exact_failure_and_compatible_success_pass(self) -> None:
        passes = evaluate_runtime_pair(
            self._runtime(compatible=False), self._runtime(compatible=True)
        )
        self.assertTrue(all(passes.values()), passes)

    def test_missing_sm120_compatible_kernel_is_rejected(self) -> None:
        compatible = self._runtime(compatible=True)
        compatible["checks"]["gru"] = {"passed": False, "error": "failure"}
        passes = evaluate_runtime_pair(self._runtime(compatible=False), compatible)
        self.assertFalse(passes["compatible_three_kernel_success"])


if __name__ == "__main__":
    unittest.main()
