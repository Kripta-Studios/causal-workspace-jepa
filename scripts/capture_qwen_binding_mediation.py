#!/usr/bin/env python
"""Capture the frozen Qwen binding treatment trajectories."""

from __future__ import annotations

import argparse
import json

from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_capture import (
    run_qwen_binding_mediation_capture,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = run_qwen_binding_mediation_capture(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
