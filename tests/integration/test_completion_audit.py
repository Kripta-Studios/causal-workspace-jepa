from __future__ import annotations

import unittest

from causal_workspace_jepa.reporting.completion_audit import evaluate_completion


class CompletionAuditTests(unittest.TestCase):
    def test_committed_scientific_artifacts_cover_explicit_criteria(self) -> None:
        criteria = evaluate_completion()
        failed = [name for name, item in criteria.items() if not item["passed"]]
        self.assertEqual(failed, [], criteria)


if __name__ == "__main__":
    unittest.main()
