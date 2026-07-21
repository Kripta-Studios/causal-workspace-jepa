from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_workspace_jepa.data.activation_store import read_hdf5_shards, write_hdf5_shards


@unittest.skipUnless(importlib.util.find_spec("h5py"), "optional h5py dependency unavailable")
class ActivationStoreTests(unittest.TestCase):
    def test_shards_checksum_and_resume(self) -> None:
        arrays = {
            "context": np.arange(30, dtype=np.float32).reshape(10, 3),
            "target": np.arange(20, dtype=np.float16).reshape(10, 2),
        }
        records = [{"id": index} for index in range(10)]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_hdf5_shards(
                tmpdir,
                arrays,
                records,
                dataset_id="test",
                config_digest="abc",
                max_shard_mb=0.0001,
                budget_mb=1,
            )
            self.assertGreater(len(manifest["shards"]), 1)
            resumed = write_hdf5_shards(
                tmpdir,
                arrays,
                records,
                dataset_id="test",
                config_digest="abc",
                max_shard_mb=0.0001,
                budget_mb=1,
            )
            self.assertEqual(manifest, resumed)
            loaded, loaded_records = read_hdf5_shards(tmpdir)
            np.testing.assert_array_equal(loaded["context"], arrays["context"])
            self.assertEqual(loaded_records, records)

    def test_budget_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                write_hdf5_shards(
                    Path(tmpdir),
                    {"x": np.zeros((10, 100), dtype=np.float32)},
                    [{"id": index} for index in range(10)],
                    dataset_id="too-large",
                    config_digest="abc",
                    budget_mb=0.0001,
                )


if __name__ == "__main__":
    unittest.main()
