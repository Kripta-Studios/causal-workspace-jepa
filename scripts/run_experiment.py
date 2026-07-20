#!/usr/bin/env python
"""Run configured experiments.

Concrete CPU smoke experiments are implemented in later milestones.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"NOT_STARTED: no experiment runner is registered for {args.config}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
