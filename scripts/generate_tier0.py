#!/usr/bin/env python
"""Generate deterministic Tier 0 datasets."""

from __future__ import annotations

import argparse
import json
import sys

from causal_workspace_jepa.data.synthetic.generate import generate_tier0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = generate_tier0(args.config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
