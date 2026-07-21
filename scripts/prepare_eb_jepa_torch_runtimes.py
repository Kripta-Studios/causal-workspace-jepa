#!/usr/bin/env python
"""Prepare isolated exact-pin and SM120-compatible EB-JEPA Torch probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


RUNTIMES = {
    "exact": {
        "target": ".cache/venvs/eb-jepa-py312-torch26",
        "torch": "torch==2.6.0",
        "index": "https://download.pytorch.org/whl/cu126",
    },
    "compatible": {
        "target": ".cache/venvs/eb-jepa-py312-torch210",
        "torch": "torch==2.10.0",
        "index": "https://download.pytorch.org/whl/cu128",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exact", "compatible", "both"), default="both")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to prepare isolated runtimes")
    names = tuple(RUNTIMES) if args.mode == "both" else (args.mode,)
    prepared = {}
    allowed_root = (Path.cwd() / ".cache" / "venvs").resolve()
    for name in names:
        spec = RUNTIMES[name]
        target = Path(spec["target"]).resolve()
        if allowed_root not in target.parents:
            raise RuntimeError(f"runtime target must stay inside {allowed_root}")
        interpreter = target / "Scripts" / "python.exe"
        if not interpreter.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([uv, "venv", str(target), "--python", "3.12"], check=True)
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(interpreter),
                str(spec["torch"]),
                "--index-url",
                str(spec["index"]),
            ],
            check=True,
        )
        prepared[name] = {
            "interpreter": str(interpreter),
            "torch": spec["torch"],
            "index": spec["index"],
        }
    print(json.dumps(prepared, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
