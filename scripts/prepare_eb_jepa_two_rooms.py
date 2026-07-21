#!/usr/bin/env python
"""Install the pinned Two Rooms import closure without replacing compatible Torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


DEFAULT_REVISION = "966e61e9285b3a876f49b9774e9720d9a99a7925"
DIRECT_IMPORT_DISTRIBUTIONS = (
    "einops",
    "fire",
    "gymnasium",
    "imageio",
    "matplotlib",
    "numpy",
    "omegaconf",
    "opencv-python",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "seaborn",
    "tqdm",
    "wandb",
)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interpreter",
        default=".cache/venvs/eb-jepa-py312-torch210/Scripts/python.exe",
    )
    parser.add_argument("--source-root", default=".cache/upstream/eb_jepa")
    parser.add_argument(
        "--lock",
        default="configs/resource/eb_jepa_two_rooms_py312_sm120.lock.txt",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    workspace = Path.cwd().resolve()
    allowed_cache = (workspace / ".cache").resolve()
    interpreter = Path(args.interpreter).resolve()
    source_root = Path(args.source_root).resolve()
    lock_path = Path(args.lock).resolve()
    if allowed_cache not in interpreter.parents or allowed_cache not in source_root.parents:
        raise RuntimeError("interpreter and source checkout must stay inside the workspace cache")
    if not interpreter.is_file() or not source_root.is_dir() or not lock_path.is_file():
        raise FileNotFoundError("missing interpreter, source checkout, or dependency lock")
    revision = _run(["git", "rev-parse", "HEAD"], cwd=source_root)
    if revision != args.revision:
        raise RuntimeError(f"EB-JEPA revision mismatch: {revision} != {args.revision}")
    if _run(["git", "status", "--porcelain"], cwd=source_root):
        raise RuntimeError("EB-JEPA checkout is dirty")

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required")
    before = json.loads(
        _run(
            [
                str(interpreter),
                "-c",
                (
                    "import json,torch; print(json.dumps({'torch':str(torch.__version__),"
                    "'cuda':torch.version.cuda,'arch':torch.cuda.get_arch_list()}))"
                ),
            ]
        )
    )
    if not str(before["torch"]).startswith("2.10.0+cu128") or "sm_120" not in before["arch"]:
        raise RuntimeError(f"interpreter is not the frozen SM120-compatible runtime: {before}")

    subprocess.run(
        [uv, "pip", "install", "--python", str(interpreter), "-r", str(lock_path)],
        check=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            "-e",
            str(source_root),
        ],
        check=True,
    )
    version_code = (
        "import importlib.metadata as m,json; names="
        + repr(DIRECT_IMPORT_DISTRIBUTIONS)
        + "; print(json.dumps({n:m.version(n) for n in names},sort_keys=True))"
    )
    packages = json.loads(_run([str(interpreter), "-c", version_code]))
    after = json.loads(
        _run(
            [
                str(interpreter),
                "-c",
                (
                    "import json,torch; print(json.dumps({'torch':str(torch.__version__),"
                    "'cuda':torch.version.cuda,'arch':torch.cuda.get_arch_list()}))"
                ),
            ]
        )
    )
    if before != after:
        raise RuntimeError(f"dependency installation changed the Torch runtime: {before} -> {after}")
    print(
        json.dumps(
            {
                "interpreter": str(interpreter),
                "source_revision": revision,
                "lock": str(lock_path),
                "runtime": after,
                "direct_import_distributions": packages,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
