from __future__ import annotations

import unittest

import numpy as np

from causal_workspace_jepa.common.types import InterventionSpec, LatentState, WorldModelOutput


class TypeTests(unittest.TestCase):
    def test_intervention_roundtrip_and_hash(self) -> None:
        spec = InterventionSpec(
            site="predictor.block0",
            operation="patch",
            positions=(0, 2),
            feature_ids=(3,),
            magnitude=0.5,
            donor_example_id="donor-1",
            seed=123,
        )
        restored = InterventionSpec.from_dict(spec.to_dict())
        self.assertEqual(restored, spec)
        self.assertEqual(hash(restored), hash(spec))

    def test_world_model_output_shape_contract(self) -> None:
        latent = LatentState(np.zeros((2, 4), dtype=np.float32), names=("z",))
        output = WorldModelOutput(
            predicted_latents=np.zeros((2, 3, 4), dtype=np.float32),
            uncertainty=None,
            decoded_state={"position": np.zeros((2, 2), dtype=np.float32)},
            intermediates={"encoder": latent.tensor},
            action_embeddings=np.zeros((2, 3, 2), dtype=np.float32),
            cost_features=None,
        )
        self.assertEqual(output.predicted_latents.shape, (2, 3, 4))
        self.assertIn("encoder", output.intermediates)


if __name__ == "__main__":
    unittest.main()
