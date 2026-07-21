from __future__ import annotations

import unittest

from causal_workspace_jepa.data.llm_prompts.capture_qwen import resolve_selected_positions


class QwenActivationCaptureTests(unittest.TestCase):
    def test_selected_positions_are_bounded_and_deterministic(self) -> None:
        self.assertEqual(resolve_selected_positions(7, ("last", "filler", "answer")), (6, 3, 6))
        self.assertEqual(resolve_selected_positions(1, ("last", "filler")), (0, 0))

    def test_empty_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_selected_positions(0, ("last",))

    def test_unknown_position_is_rejected_by_resolver(self) -> None:
        with self.assertRaises(KeyError):
            resolve_selected_positions(4, ("unknown",))


if __name__ == "__main__":
    unittest.main()
