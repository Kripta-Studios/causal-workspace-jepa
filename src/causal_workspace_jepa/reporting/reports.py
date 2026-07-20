"""Lightweight report-card helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_json_report(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
