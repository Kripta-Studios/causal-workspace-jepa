#!/usr/bin/env python
"""Run configured experiments."""

from __future__ import annotations

import argparse
import json
import sys

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.world_model.tiny_jepa_smoke import run_tiny_jepa_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_id = str(config.get("id", ""))
    if experiment_id == "WM-T0-001":
        metrics = run_tiny_jepa_smoke(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    print(f"NOT_STARTED: no experiment runner is registered for {args.config}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
