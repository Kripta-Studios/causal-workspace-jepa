"""Tiny sparse dictionary for CPU smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SparseDictionary:
    dictionary: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, activations: np.ndarray, components: int, threshold: float = 0.05) -> "SparseDictionary":
        centered = activations.reshape(-1, activations.shape[-1]).astype(np.float64)
        centered = centered - centered.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        dictionary = vh[:components].astype(np.float32)
        return cls(dictionary=dictionary, threshold=threshold)

    def encode(self, activations: np.ndarray) -> np.ndarray:
        flat = activations.reshape(-1, activations.shape[-1]).astype(np.float32)
        codes = flat @ self.dictionary.T
        return np.sign(codes) * np.maximum(np.abs(codes) - self.threshold, 0.0)

    def decode(self, codes: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        flat = codes @ self.dictionary
        return flat.reshape(original_shape)

    def reconstruction_mse(self, activations: np.ndarray) -> float:
        codes = self.encode(activations)
        reconstruction = self.decode(codes, activations.shape)
        return float(np.mean((reconstruction - activations) ** 2))

    def feature_density(self, activations: np.ndarray) -> float:
        codes = self.encode(activations)
        return float(np.mean(codes != 0.0))
