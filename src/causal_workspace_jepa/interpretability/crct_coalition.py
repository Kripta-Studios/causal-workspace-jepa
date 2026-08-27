"""Coalition, redundancy, cancellation, and equivalence metrics for CRCT successors.

These helpers are prospective. They must not be used to retune or relabel
``CRCT-STAGE0-HARD-002``. Literal graph recall and epsilon-functional sufficiency
are reported as distinct objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


def _energy(values: Tensor) -> float:
    return float(torch.mean(values.reshape(values.shape[0], -1).square()).item())


def _nmse(prediction: Tensor, target: Tensor) -> float:
    denom = max(_energy(target), 1e-12)
    return _energy(prediction - target) / denom


def signed_contributions(
    contributions: Mapping[str, Tensor],
) -> dict[str, dict[str, float]]:
    """Record signed energy, not only absolute attribution."""

    payload: dict[str, dict[str, float]] = {}
    for name, tensor in contributions.items():
        flat = tensor.reshape(tensor.shape[0], -1)
        payload[name] = {
            "mean": float(flat.mean().item()),
            "energy": float(torch.mean(flat.square()).item()),
            "abs_mean": float(flat.abs().mean().item()),
            "sign": float(torch.sign(flat.mean()).item()) if flat.numel() else 0.0,
        }
    return payload


def epsilon_functional_error(
    candidate_sum: Tensor,
    full_target: Tensor,
) -> float:
    """Unexplained fraction when reconstructing the full effect from a candidate set."""

    return _nmse(candidate_sum, full_target)


def is_epsilon_sufficient(error: float, *, epsilon: float) -> bool:
    return float(error) <= float(epsilon)


def minimal_sufficient_sets(
    contributions: Mapping[str, Tensor],
    full_target: Tensor,
    *,
    epsilon: float,
    required: Sequence[str] | None = None,
) -> list[tuple[str, ...]]:
    """Enumerate inclusion-minimal additive subsets that reconstruct the target."""

    names = tuple(contributions)
    required_set = set(required or ())
    found: list[tuple[str, ...]] = []
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            if required_set and not required_set.issubset(combo):
                continue
            stacked = sum((contributions[name] for name in combo), torch.zeros_like(full_target))
            if not is_epsilon_sufficient(epsilon_functional_error(stacked, full_target), epsilon=epsilon):
                continue
            if any(set(existing).issubset(combo) for existing in found):
                continue
            found.append(tuple(combo))
    return found


def necessity_drop(
    contributions: Mapping[str, Tensor],
    full_target: Tensor,
    member: str,
    *,
    base: Sequence[str],
) -> dict[str, float]:
    """Degradation after removing one member from an otherwise evaluated set."""

    with_member = sum((contributions[name] for name in base), torch.zeros_like(full_target))
    without = sum(
        (contributions[name] for name in base if name != member),
        torch.zeros_like(full_target),
    )
    return {
        "with_error": epsilon_functional_error(with_member, full_target),
        "without_error": epsilon_functional_error(without, full_target),
        "delta_error": epsilon_functional_error(without, full_target)
        - epsilon_functional_error(with_member, full_target),
    }


def redundancy_groups(
    contributions: Mapping[str, Tensor],
    full_target: Tensor,
    *,
    epsilon: float,
    correlation_min: float,
) -> list[dict[str, Any]]:
    """Pairs that can substitute for each other inside epsilon."""

    names = list(contributions)
    groups: list[dict[str, Any]] = []
    for left, right in combinations(names, 2):
        a = contributions[left].reshape(full_target.shape[0], -1)
        b = contributions[right].reshape(full_target.shape[0], -1)
        a_c = a - a.mean(dim=0, keepdim=True)
        b_c = b - b.mean(dim=0, keepdim=True)
        denom = torch.linalg.norm(a_c) * torch.linalg.norm(b_c)
        corr = float((a_c * b_c).sum().item() / denom.item()) if float(denom) > 0 else 0.0
        if corr < correlation_min:
            continue
        others = [name for name in names if name not in {left, right}]
        rest = sum((contributions[name] for name in others), torch.zeros_like(full_target))
        left_ok = is_epsilon_sufficient(
            epsilon_functional_error(rest + contributions[left], full_target),
            epsilon=epsilon,
        )
        right_ok = is_epsilon_sufficient(
            epsilon_functional_error(rest + contributions[right], full_target),
            epsilon=epsilon,
        )
        if left_ok and right_ok:
            groups.append(
                {
                    "members": [left, right],
                    "correlation": corr,
                    "substitutable": True,
                }
            )
    return groups


def cancellation_groups(
    contributions: Mapping[str, Tensor],
    *,
    sum_energy_ratio_max: float,
    min_member_energy: float,
) -> list[dict[str, Any]]:
    """Pairs with material opposing signed effects that cancel in aggregate."""

    names = list(contributions)
    groups: list[dict[str, Any]] = []
    for left, right in combinations(names, 2):
        a = contributions[left]
        b = contributions[right]
        energy_a = _energy(a)
        energy_b = _energy(b)
        if energy_a < min_member_energy or energy_b < min_member_energy:
            continue
        summed = a + b
        ratio = _energy(summed) / max(0.5 * (energy_a + energy_b), 1e-12)
        mean_product = float((a * b).mean().item())
        if ratio <= sum_energy_ratio_max and mean_product < 0:
            groups.append(
                {
                    "members": [left, right],
                    "sum_energy_ratio": ratio,
                    "mean_product": mean_product,
                    "signed": True,
                }
            )
    return groups


def equivalent_circuits(
    sets: Sequence[Sequence[str]],
    contributions: Mapping[str, Tensor],
    full_target: Tensor,
    *,
    epsilon: float,
) -> list[dict[str, Any]]:
    """Mark circuit pairs whose finite additive behavior is indistinguishable."""

    records: list[dict[str, Any]] = []
    materialized = [tuple(item) for item in sets]
    for left, right in combinations(range(len(materialized)), 2):
        sum_left = sum(
            (contributions[name] for name in materialized[left]),
            torch.zeros_like(full_target),
        )
        sum_right = sum(
            (contributions[name] for name in materialized[right]),
            torch.zeros_like(full_target),
        )
        pair_error = _nmse(sum_left, sum_right)
        left_ok = is_epsilon_sufficient(epsilon_functional_error(sum_left, full_target), epsilon=epsilon)
        right_ok = is_epsilon_sufficient(
            epsilon_functional_error(sum_right, full_target),
            epsilon=epsilon,
        )
        records.append(
            {
                "left": list(materialized[left]),
                "right": list(materialized[right]),
                "pairwise_nmse": pair_error,
                "equivalent": bool(left_ok and right_ok and pair_error <= epsilon),
            }
        )
    return records


def literal_recall(
    selected: Iterable[str],
    planted: Iterable[str],
) -> dict[str, float]:
    chosen = set(selected)
    truth = set(planted)
    tp = chosen & truth
    precision = len(tp) / max(len(chosen), 1)
    recall = len(tp) / max(len(truth), 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "true_positive_count": float(len(tp)),
        "planted_count": float(len(truth)),
        "selected_count": float(len(chosen)),
    }


def intervention_support_label(
    *,
    operation: str,
    site: str,
    magnitude: float,
    combination: Sequence[str],
    trained_operations: Sequence[str],
    trained_sites: Sequence[str],
    magnitude_max: float,
    trained_combination_size_max: int,
) -> dict[str, Any]:
    """Label an intervention in-support or out-of-support of the training envelope."""

    in_support = (
        operation in trained_operations
        and site in trained_sites
        and abs(float(magnitude)) <= float(magnitude_max)
        and len(tuple(combination)) <= int(trained_combination_size_max)
    )
    return {
        "operation": operation,
        "site": site,
        "magnitude": float(magnitude),
        "combination": list(combination),
        "in_support": bool(in_support),
        "support_status": "in_support" if in_support else "out_of_support",
    }


@dataclass(frozen=True)
class CoalitionReport:
    literal: dict[str, float]
    sufficiency_error: float
    sufficient: bool
    minimal_sets: list[tuple[str, ...]]
    necessity: dict[str, dict[str, float]]
    redundancy: list[dict[str, Any]]
    cancellation: list[dict[str, Any]]
    equivalence: list[dict[str, Any]]
    signed: dict[str, dict[str, float]]
    support_labels: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["minimal_sets"] = [list(item) for item in self.minimal_sets]
        return payload


def evaluate_coalition(
    contributions: Mapping[str, Tensor],
    full_target: Tensor,
    *,
    planted: Sequence[str],
    selected: Sequence[str],
    epsilon: float,
    redundancy_correlation_min: float,
    cancellation_sum_energy_ratio_max: float,
    cancellation_min_member_energy: float,
    support_labels: Sequence[Mapping[str, Any]] | None = None,
) -> CoalitionReport:
    selected_sum = sum(
        (contributions[name] for name in selected),
        torch.zeros_like(full_target),
    )
    sufficiency_error = epsilon_functional_error(selected_sum, full_target)
    minimal = minimal_sufficient_sets(contributions, full_target, epsilon=epsilon)
    necessity = {
        name: necessity_drop(contributions, full_target, name, base=selected)
        for name in selected
    }
    return CoalitionReport(
        literal=literal_recall(selected, planted),
        sufficiency_error=sufficiency_error,
        sufficient=is_epsilon_sufficient(sufficiency_error, epsilon=epsilon),
        minimal_sets=minimal,
        necessity=necessity,
        redundancy=redundancy_groups(
            contributions,
            full_target,
            epsilon=epsilon,
            correlation_min=redundancy_correlation_min,
        ),
        cancellation=cancellation_groups(
            contributions,
            sum_energy_ratio_max=cancellation_sum_energy_ratio_max,
            min_member_energy=cancellation_min_member_energy,
        ),
        equivalence=equivalent_circuits(minimal, contributions, full_target, epsilon=epsilon),
        signed=signed_contributions(contributions),
        support_labels=[dict(item) for item in (support_labels or ())],
    )
