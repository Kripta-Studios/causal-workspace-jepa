from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_workspace_jepa.hooks.gradients import finite_difference_jacobian
from causal_workspace_jepa.interpretability.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from causal_workspace_jepa.interpretability.probing import RidgeProbe, random_label_control
from causal_workspace_jepa.models.sparse_dictionary import SparseDictionary


class InterpretabilityPrimitiveTests(unittest.TestCase):
    def test_ridge_probe_beats_mean_on_linear_signal(self) -> None:
        x = np.arange(20, dtype=np.float32)[:, None]
        y = 2.0 * x + 1.0
        probe = RidgeProbe.fit(x[:15], y[:15])
        self.assertLess(probe.score_mse(x[15:], y[15:]), probe.mean_baseline_mse(y[15:]))
        shuffled = random_label_control(y, seed=0)
        self.assertEqual(shuffled.shape, y.shape)

    def test_sparse_dictionary_shapes(self) -> None:
        rng = np.random.default_rng(0)
        activations = rng.normal(size=(10, 6)).astype(np.float32)
        dictionary = SparseDictionary.fit(activations, components=3)
        codes = dictionary.encode(activations)
        self.assertEqual(codes.shape, (10, 3))
        self.assertGreaterEqual(dictionary.feature_density(activations), 0.0)
        shifted = activations + 10.0
        shifted_dictionary = SparseDictionary.fit(shifted, components=6, threshold=0.0)
        self.assertLess(shifted_dictionary.reconstruction_mse(shifted), 1e-10)

    def test_finite_difference_jacobian(self) -> None:
        jacobian = finite_difference_jacobian(lambda value: value**2, np.array([3.0]))
        self.assertAlmostEqual(float(jacobian.reshape(-1)[0]), 6.0, places=3)

    def test_circuit_graph_json_and_graphml(self) -> None:
        graph = CircuitGraph(
            graph_id="g",
            nodes=(CircuitNode("n1", "site", "Availability", 1.0),),
            edges=(CircuitEdge("n1", "n1", 0.5, 0.8),),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "graph.json"
            graphml_path = Path(tmpdir) / "graph.graphml"
            graph.write_json(json_path)
            graph.write_graphml(graphml_path)
            self.assertEqual(CircuitGraph.read_json(json_path).graph_id, "g")
            self.assertIn("<graphml", graphml_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
