"""Small helpers for Markdown-backed registries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegistryEntry:
    identifier: str
    status: str
    summary: str


def append_registry_entry(path: str | Path, entry: RegistryEntry) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {entry.identifier}\n\n")
        handle.write(f"- status: `{entry.status}`\n")
        handle.write(f"- summary: {entry.summary}\n")
