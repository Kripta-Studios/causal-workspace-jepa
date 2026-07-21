from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from causal_workspace_jepa.experiments.world_model.lewm_action_path_geometry_study import (
    _composite_legendre,
    _decoded_directional_derivatives,
    _load_progress,
    _run_fingerprint,
    _spearman,
    _stratified_permutation_null,
    _stratified_profile_indices,
    _write_progress,
)


class LeWMActionPathGeometryTests(unittest.TestCase):
    def test_streamed_jacobians_are_exact_detached_cpu_values(self) -> None:
        class LinearRollout(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "matrix",
                    torch.tensor([[1.0, 2.0, 0.0, -1.0], [0.5, 0.0, 1.0, 1.5]]),
                )

            def rollout(
                self, initial: torch.Tensor, actions: torch.Tensor
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
                increments = actions @ self.matrix.T
                return initial[:, None] + torch.cumsum(increments, dim=1), {}

        model = LinearRollout()
        initial = torch.zeros(5, 2)
        suffix = torch.empty(5, 0, 4)
        actions = torch.eye(4)[torch.tensor([0, 1, 2, 3, 0])]
        delta = torch.tensor(
            [[-1.0, 1.0, 0.0, 0.0]] * 5, dtype=torch.float32
        )
        actual = _decoded_directional_derivatives(
            model,
            initial,
            suffix,
            actions,
            delta,
            torch.eye(2),
            chunk_size=2,
            outer_batch_size=2,
        )
        expected = delta @ model.matrix.T
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(actual.device.type, "cpu")
        self.assertFalse(actual.requires_grad)

    def test_composite_legendre_integrates_polynomial(self) -> None:
        nodes, weights = _composite_legendre(order=4, panels=5)
        self.assertEqual(len(nodes), 20)
        self.assertTrue(np.all(np.diff(nodes) > 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(np.sum(weights * nodes**6)), 1.0 / 7.0, places=12)

    def test_profile_sampling_is_action_pair_stratified(self) -> None:
        contexts, pairs = _stratified_profile_indices(
            20, contexts_per_pair=3, seed=11
        )
        self.assertEqual(len(contexts), 36)
        self.assertEqual(np.bincount(pairs).tolist(), [3] * 12)
        for pair_id in range(12):
            self.assertEqual(len(set(contexts[pairs == pair_id])), 3)

    def test_stratified_null_breaks_within_pair_association(self) -> None:
        predictor = np.tile(np.arange(4, dtype=np.float64), 12)
        target = predictor.copy()
        strata = np.repeat(np.arange(12), 4)
        null = _stratified_permutation_null(
            predictor, target, strata, permutations=128, seed=13
        )
        self.assertAlmostEqual(_spearman(predictor, target), 1.0)
        self.assertLess(null["p95_spearman"], 0.99)

    def test_progress_round_trip_is_bound_to_config_and_commit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "study.yaml"
            progress = root / "study.progress.json"
            config.write_text("id: example\n", encoding="utf-8")
            fingerprint = _run_fingerprint(config, "abc123")
            seed_results = [{"model_seed": 101, "horizons": {"1": {"ok": True}}}]
            _write_progress(
                progress,
                experiment_id="EXAMPLE-001",
                run_fingerprint=fingerprint,
                git_commit="abc123",
                seed_results=seed_results,
            )
            self.assertEqual(
                _load_progress(
                    progress,
                    experiment_id="EXAMPLE-001",
                    run_fingerprint=fingerprint,
                ),
                seed_results,
            )
            self.assertFalse(progress.with_name(progress.name + ".tmp").exists())
            with self.assertRaisesRegex(RuntimeError, "stale progress fingerprint"):
                _load_progress(
                    progress,
                    experiment_id="EXAMPLE-001",
                    run_fingerprint=_run_fingerprint(config, "different"),
                )

    def test_run_fingerprint_changes_with_config_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            config = Path(directory) / "study.yaml"
            config.write_text("value: 1\n", encoding="utf-8")
            first = _run_fingerprint(config, "abc123")
            config.write_text("value: 2\n", encoding="utf-8")
            self.assertNotEqual(first, _run_fingerprint(config, "abc123"))


if __name__ == "__main__":
    unittest.main()
