#!/usr/bin/env python
"""Clone or validate the pinned official EB-JEPA source checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from causal_workspace_jepa.adapters.eb_jepa_adapter import (
    EB_JEPA_UPSTREAM_REVISION,
    EB_JEPA_UPSTREAM_URL,
    verify_eb_jepa_checkout,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=".cache/upstream/eb_jepa")
    parser.add_argument("--revision", default=EB_JEPA_UPSTREAM_REVISION)
    args = parser.parse_args()
    target = Path(args.target).resolve()
    allowed_root = (Path.cwd() / ".cache" / "upstream").resolve()
    if target != allowed_root and allowed_root not in target.parents:
        raise RuntimeError(f"target must stay inside {allowed_root}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", EB_JEPA_UPSTREAM_URL, str(target)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", str(args.revision)],
            check=True,
        )
    metadata = verify_eb_jepa_checkout(target, expected_revision=str(args.revision))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
