"""Activation cache containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class ActivationCache:
    activations: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, site: str, value: np.ndarray) -> None:
        self.activations[site] = np.array(value, copy=True)

    def require(self, site: str) -> np.ndarray:
        if site not in self.activations:
            raise KeyError(f"activation site not captured: {site}")
        return self.activations[site]

    def subset(self, sites: list[str] | tuple[str, ...]) -> Mapping[str, np.ndarray]:
        return {site: self.require(site) for site in sites}
