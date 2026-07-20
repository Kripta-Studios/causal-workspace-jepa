"""Manifest validation helpers."""

from __future__ import annotations

from pathlib import Path


def require_manifest(path: str | Path) -> Path:
    manifest = Path(path)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return manifest
