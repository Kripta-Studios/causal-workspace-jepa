#!/usr/bin/env python
"""Generate real Qwen intervention-outcome datasets.

Blocked on the CPU VPS because it requires Hugging Face hidden activations.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.parse_args()
    print("BLOCKED_RESOURCE: real Qwen intervention data requires gpu_12gb or larger", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
