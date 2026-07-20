from __future__ import annotations

import json
import subprocess
import sys
import unittest


class ReproducibilityAuditTests(unittest.TestCase):
    def test_audit_script(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/audit_reproducibility.py"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "SMOKE_VALIDATED")
        self.assertEqual(payload["errors"], [])
        self.assertGreaterEqual(payload["metric_artifacts"], 6)
        self.assertGreaterEqual(payload["checksums_verified"], 6)


if __name__ == "__main__":
    unittest.main()
