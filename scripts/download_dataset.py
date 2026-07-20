#!/usr/bin/env python
"""Download external datasets with resource checks.

External dataset downloads are blocked on the CPU VPS.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--confirm-size", action="store_true")
    parser.parse_args()
    print("BLOCKED_RESOURCE: external dataset downloads are disabled for cpu_vps", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
