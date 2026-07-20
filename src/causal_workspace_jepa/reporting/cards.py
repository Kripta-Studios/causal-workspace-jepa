"""Reproducibility card helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultCard:
    result_id: str
    evidence_level: str
    status: str
    metrics_path: str
    commit: str
