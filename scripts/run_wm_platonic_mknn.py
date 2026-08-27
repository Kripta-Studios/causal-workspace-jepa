#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from causal_workspace_jepa.experiments.world_model.platonic_mknn import main as run_main

    return run_main()


if __name__ == "__main__":
    raise SystemExit(main())
