from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from causal_workspace_jepa.cli import main


class CliTests(unittest.TestCase):
    def test_doctor_cli_outputs_json(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["doctor", "--resource-profile", "configs/resource/cpu_vps.yaml"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["profile_name"], "cpu_vps")
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
