"""CPU-safe ensemble uncertainty calibration and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ensemble_disagreement(predictions: np.ndarray) -> np.ndarray:
    return np.var(predictions, axis=0)


@dataclass(frozen=True)
class IntervalCalibrator:
    """Single-scale split calibrator for ensemble predictive variance."""

    scale: float
    variance_floor: float
    target_coverage: float
    normal_quantile: float
    homoscedastic_variance: float

    @classmethod
    def fit(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        *,
        target_coverage: float = 0.9,
    ) -> "IntervalCalibrator":
        members = np.asarray(predictions, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        if members.ndim < 3 or members.shape[1:] != target.shape:
            raise ValueError("predictions must have shape [members, *targets.shape]")
        if not 0.5 < target_coverage < 1.0:
            raise ValueError("target_coverage must be in (0.5, 1)")
        mean = members.mean(axis=0)
        variance = members.var(axis=0, ddof=1)
        positive = variance[variance > 0]
        floor = float(np.median(positive) * 1e-3) if positive.size else 1e-8
        floor = max(floor, 1e-10)
        standardized_error = np.abs(target - mean) / np.sqrt(variance + floor)
        normal_quantile = _normal_quantile(target_coverage)
        empirical_quantile = float(np.quantile(standardized_error, target_coverage))
        scale = max((empirical_quantile / normal_quantile) ** 2, 1e-8)
        return cls(
            scale=scale,
            variance_floor=floor,
            target_coverage=target_coverage,
            normal_quantile=normal_quantile,
            homoscedastic_variance=max(float(np.mean((target - mean) ** 2)), 1e-10),
        )

    def variance(self, predictions: np.ndarray) -> np.ndarray:
        members = np.asarray(predictions, dtype=np.float64)
        raw = members.var(axis=0, ddof=1)
        return ((raw + self.variance_floor) * self.scale).astype(np.float32)

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        members = np.asarray(predictions, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        mean = members.mean(axis=0)
        variance = np.asarray(self.variance(members), dtype=np.float64)
        errors = target - mean
        covered = np.abs(errors) <= self.normal_quantile * np.sqrt(variance)
        nll = _gaussian_nll(errors, variance)
        baseline_variance = np.full_like(
            variance,
            self.homoscedastic_variance,
        )
        return {
            "coverage": float(np.mean(covered)),
            "target_coverage": self.target_coverage,
            "mean_variance": float(np.mean(variance)),
            "mean_squared_error": float(np.mean(errors**2)),
            "gaussian_nll": nll,
            "homoscedastic_nll": _gaussian_nll(errors, baseline_variance),
        }


def rank_auc(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    """Return tie-aware probability that a positive score exceeds a negative."""

    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if negative.size == 0 or positive.size == 0:
        raise ValueError("both score groups must be non-empty")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0).astype(np.float64) + 0.5 * (comparisons == 0)))


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _average_ranks(np.asarray(left, dtype=np.float64).reshape(-1))
    right_ranks = _average_ranks(np.asarray(right, dtype=np.float64).reshape(-1))
    if np.std(left_ranks) < 1e-12 or np.std(right_ranks) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _normal_quantile(coverage: float) -> float:
    # Fixed values avoid pulling scipy into the CPU profile.
    values = {0.8: 1.2815515655, 0.9: 1.644853627, 0.95: 1.959963985}
    rounded = round(coverage, 2)
    if rounded not in values or abs(rounded - coverage) > 1e-8:
        raise ValueError("supported target coverage values are 0.80, 0.90, and 0.95")
    return values[rounded]


def _gaussian_nll(errors: np.ndarray, variance: np.ndarray) -> float:
    stable_variance = np.maximum(variance, 1e-12)
    terms = np.log(2.0 * np.pi * stable_variance) + errors**2 / stable_variance
    return float(np.mean(0.5 * terms))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks
