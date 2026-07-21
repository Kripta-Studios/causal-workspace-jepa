#!/usr/bin/env python
"""Run the explicit AGENTS.md completion audit."""

from __future__ import annotations

import json

from causal_workspace_jepa.reporting.completion_audit import run_completion_audit


def main() -> int:
    result = run_completion_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
