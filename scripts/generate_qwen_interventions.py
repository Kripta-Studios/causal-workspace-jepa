#!/usr/bin/env python
"""Generate a real, bounded, checksummed Qwen intervention dataset."""

from __future__ import annotations

import argparse
import json

from causal_workspace_jepa.experiments.llm.qwen_intervention_dataset import (
    run_qwen_intervention_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = run_qwen_intervention_dataset(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
