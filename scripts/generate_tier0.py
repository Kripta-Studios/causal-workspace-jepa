#!/usr/bin/env python
"""Generate deterministic Tier 0 datasets.

Implemented in Milestone 1. This placeholder is intentionally explicit so the
CPU control plane can audit missing work without pretending data exists.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.parse_args()
    print("NOT_STARTED: Tier 0 generation is scheduled for Milestone 1", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
