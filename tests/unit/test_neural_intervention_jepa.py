from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_workspace_jepa.models.intervention_jepa import (
    NeuralInterventionJEPA,
    TrajectoryInterventionJEPA,
)


class NeuralInterventionJEPATests(unittest.TestCase):
    def test_fit_save_load_without_target_leakage(self) -> None:
        rng = np.random.default_rng(0)
        context = rng.normal(size=(300, 4)).astype(np.float32)
        intervention = rng.normal(size=(300, 3)).astype(np.float32)
        target = (intervention[:, :2] + context[:, :2] * intervention[:, :1]).astype(np.float32)
        model = NeuralInterventionJEPA.fit(
            context[:240],
            intervention[:240],
            target[:240],
            (context[240:270], intervention[240:270], target[240:270]),
            hidden_dim=16,
            meta_dim=8,
            steps=400,
            seed=0,
        )
        prediction = model.predict(context[270:], intervention[270:])
        self.assertLess(float(np.mean((prediction - target[270:]) ** 2)), 0.2)
        self.assertEqual(model.encode_meta(context[:2], intervention[:2]).shape, (2, 8))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ijepa.npz"
            model.save(path)
            loaded = NeuralInterventionJEPA.load(path)
            np.testing.assert_allclose(
                loaded.predict(context[270:], intervention[270:]), prediction, atol=0.0, rtol=0.0
            )

    def test_trajectory_variant_restores_target_shape(self) -> None:
        rng = np.random.default_rng(1)
        context = rng.normal(size=(40, 2, 3)).astype(np.float32)
        intervention = rng.normal(size=(40, 2, 2)).astype(np.float32)
        target = np.repeat(intervention[..., :1], 3, axis=-1)
        model = TrajectoryInterventionJEPA.fit(
            context[:30],
            intervention[:30],
            target[:30],
            (context[30:35], intervention[30:35], target[30:35]),
            hidden_dim=12,
            meta_dim=6,
            steps=120,
            seed=1,
        )
        self.assertEqual(model.predict(context[35:], intervention[35:]).shape, (5, 2, 3))


if __name__ == "__main__":
    unittest.main()
