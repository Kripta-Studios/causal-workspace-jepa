from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from causal_workspace_jepa.common.config import get_nested, load_config
from causal_workspace_jepa.common.resources import inspect_resources


class ConfigResourceTests(unittest.TestCase):
    def test_load_nested_cpu_profile(self) -> None:
        config = load_config("configs/resource/cpu_vps.yaml")
        self.assertEqual(config["name"], "cpu_vps")
        self.assertFalse(get_nested(config, "network.allow_model_weight_downloads"))
        self.assertEqual(get_nested(config, "storage.min_free_gb"), 4)

    def test_doctor_passes_cpu_profile(self) -> None:
        report = inspect_resources("configs/resource/cpu_vps.yaml")
        self.assertTrue(report.ok, report.messages)
        self.assertGreaterEqual(report.free_gb, report.min_free_gb)

    def test_gpu_profile_blocks_without_gpu(self) -> None:
        report = inspect_resources("configs/resource/gpu_12gb.yaml")
        if not report.ok:
            self.assertTrue(any("BLOCKED_RESOURCE" in message for message in report.messages))
        else:
            self.assertIsNotNone(report.gpu_name)
            self.assertIsNotNone(report.gpu_vram_gb)
            self.assertTrue(report.torch_cuda_available)
            self.assertGreaterEqual(report.gpu_vram_gb or 0.0, 12.0)

    def test_resource_report_serializes_runtime_fields(self) -> None:
        payload = inspect_resources("configs/resource/cpu_vps.yaml").as_dict()
        self.assertIn("ram_gb", payload)
        self.assertIn("gpu_name", payload)
        self.assertIn("torch_cuda_available", payload)

    def test_reject_odd_indentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text("root:\n key: value\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
