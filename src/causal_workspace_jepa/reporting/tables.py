"""Metric table helpers."""

from __future__ import annotations

from typing import Iterable, Mapping


def markdown_table(rows: Iterable[Mapping[str, object]], columns: list[str]) -> str:
    rows = list(rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])
