"""Preregistered protocol primitives for Qwen binding mediation.

The protocol deliberately separates three objects that are often conflated:

* a token-level treatment that swaps two value embeddings at layer zero;
* a ranked set of downstream module outputs that may mediate that treatment;
* a later Intervention-JEPA trained on the resulting causal trajectories.

This module contains no model fitting and makes no circuit claim.  Its pure
helpers make the episode construction and mediation estimands independently
testable before protected model outcomes are generated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PRIMARY_TEMPLATE = (
    "Use the four mappings. Reply with only the value.\n"
    "- {k0} -> {v0}\n"
    "- {k1} -> {v1}\n"
    "- {k2} -> {v2}\n"
    "- {k3} -> {v3}\n"
    "Query: {query} ->"
)

PARAPHRASE_TEMPLATE = (
    "Read this lookup table and return only the requested entry.\n"
    "- {k0} = {v0}\n"
    "- {k1} = {v1}\n"
    "- {k2} = {v2}\n"
    "- {k3} = {v3}\n"
    "Value paired with {query}:"
)


@dataclass(frozen=True)
class BindingEpisode:
    """One independent four-pair treatment episode."""

    episode_id: str
    split: str
    keys: tuple[str, str, str, str]
    recipient_values: tuple[str, str, str, str]
    donor_values: tuple[str, str, str, str]
    query_index: int
    swapped_indices: tuple[int, int]
    template: str = "primary"

    def __post_init__(self) -> None:
        if len(set(self.keys)) != 4 or len(set(self.recipient_values)) != 4:
            raise ValueError("binding episodes require four distinct keys and values")
        if Counter(self.recipient_values) != Counter(self.donor_values):
            raise ValueError("recipient and donor must contain the same value multiset")
        left, right = self.swapped_indices
        if left == right or not ({left, right} <= set(range(4))):
            raise ValueError("swapped_indices must identify two distinct entries")
        expected = list(self.recipient_values)
        expected[left], expected[right] = expected[right], expected[left]
        if tuple(expected) != self.donor_values:
            raise ValueError("donor_values must be exactly one registered transposition")
        if self.query_index not in self.swapped_indices:
            raise ValueError("the queried binding must be changed by the treatment")
        if self.template not in {"primary", "paraphrase"}:
            raise ValueError(f"unknown binding template: {self.template}")

    @property
    def recipient_answer(self) -> str:
        return self.recipient_values[self.query_index]

    @property
    def donor_answer(self) -> str:
        return self.donor_values[self.query_index]

    def recipient_prompt(self) -> str:
        return render_binding_prompt(
            self.keys,
            self.recipient_values,
            self.query_index,
            template=self.template,
        )

    def donor_prompt(self) -> str:
        return render_binding_prompt(
            self.keys,
            self.donor_values,
            self.query_index,
            template=self.template,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizedTreatment:
    """Audited token-level treatment for one episode."""

    recipient_ids: tuple[int, ...]
    donor_ids: tuple[int, ...]
    changed_positions: tuple[int, int]
    recipient_answer_id: int
    donor_answer_id: int


@dataclass(frozen=True)
class MediationEstimate:
    """Aggregate ratio-of-sums mediation estimands."""

    treatment_effect: float
    sufficiency: float
    necessity: float
    eligible: bool


@dataclass(frozen=True)
class PrefixSelection:
    """Frozen train-only mediator-prefix decision."""

    selected: tuple[str, ...]
    eligible: bool
    reason: str


def render_binding_prompt(
    keys: Sequence[str],
    values: Sequence[str],
    query_index: int,
    *,
    template: str,
) -> str:
    """Render one fixed-length binding prompt."""

    if len(keys) != 4 or len(values) != 4:
        raise ValueError("the binding benchmark is fixed to exactly four pairs")
    if query_index not in range(4):
        raise ValueError("query_index must be in [0, 3]")
    pattern = PRIMARY_TEMPLATE if template == "primary" else PARAPHRASE_TEMPLATE
    if template not in {"primary", "paraphrase"}:
        raise ValueError(f"unknown template: {template}")
    fields = {f"k{index}": key for index, key in enumerate(keys)}
    fields.update({f"v{index}": value for index, value in enumerate(values)})
    fields["query"] = keys[query_index]
    return pattern.format(**fields)


def generate_binding_episodes(
    *,
    split: str,
    keys: Sequence[str],
    values: Sequence[str],
    count: int,
    seed: int,
    template: str = "primary",
) -> list[BindingEpisode]:
    """Generate deterministic, balanced, duplicate-free treatment episodes.

    Query position and swap partner are cycled rather than sampled, preventing a
    lucky imbalance from becoming a hidden task cue.  Keys and values are still
    shuffled within each episode by the registered split seed.
    """

    if len(keys) < 4 or len(values) < 4:
        raise ValueError("each split requires at least four keys and values")
    if len(set(keys)) != len(keys) or len(set(values)) != len(values):
        raise ValueError("split pools must not contain duplicates")
    if count <= 0:
        raise ValueError("episode count must be positive")
    rng = np.random.default_rng(seed)
    episodes: list[BindingEpisode] = []
    signatures: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = count * 100
    while len(episodes) < count and attempts < max_attempts:
        attempts += 1
        index = len(episodes)
        episode_keys = tuple(str(value) for value in rng.choice(keys, size=4, replace=False))
        episode_values = tuple(
            str(value) for value in rng.choice(values, size=4, replace=False)
        )
        query = index % 4
        partner = (query + 1 + (index // 4) % 3) % 4
        swapped = tuple(sorted((query, partner)))
        donor = list(episode_values)
        donor[query], donor[partner] = donor[partner], donor[query]
        signature = (episode_keys, episode_values, query, swapped, template)
        if signature in signatures:
            continue
        signatures.add(signature)
        episodes.append(
            BindingEpisode(
                episode_id=f"{split}-{index:04d}",
                split=split,
                keys=episode_keys,  # type: ignore[arg-type]
                recipient_values=episode_values,  # type: ignore[arg-type]
                donor_values=tuple(donor),  # type: ignore[arg-type]
                query_index=query,
                swapped_indices=swapped,  # type: ignore[arg-type]
                template=template,
            )
        )
    if len(episodes) != count:
        raise RuntimeError("could not generate the requested unique binding episodes")
    return episodes


def audit_tokenized_treatment(
    episode: BindingEpisode,
    *,
    encode_prompt: Callable[[str], Sequence[int]],
    encode_answer: Callable[[str], Sequence[int]],
) -> TokenizedTreatment:
    """Prove that a donor treatment is exactly a two-token transposition."""

    recipient = tuple(int(value) for value in encode_prompt(episode.recipient_prompt()))
    donor = tuple(int(value) for value in encode_prompt(episode.donor_prompt()))
    if len(recipient) != len(donor):
        raise ValueError("recipient and donor token lengths differ")
    changed = tuple(index for index, pair in enumerate(zip(recipient, donor, strict=True)) if pair[0] != pair[1])
    if len(changed) != 2:
        raise ValueError(f"treatment must change exactly two token positions, observed {len(changed)}")
    if Counter(recipient) != Counter(donor):
        raise ValueError("recipient and donor token-ID multisets differ")
    recipient_answer = tuple(int(value) for value in encode_answer(episode.recipient_answer))
    donor_answer = tuple(int(value) for value in encode_answer(episode.donor_answer))
    if len(recipient_answer) != 1 or len(donor_answer) != 1:
        raise ValueError("recipient and donor answers must each be exactly one token")
    if recipient_answer == donor_answer:
        raise ValueError("treatment must change the queried answer token")
    return TokenizedTreatment(
        recipient_ids=recipient,
        donor_ids=donor,
        changed_positions=(changed[0], changed[1]),
        recipient_answer_id=recipient_answer[0],
        donor_answer_id=donor_answer[0],
    )


def mediation_estimate(
    clean_scores: Sequence[float],
    treated_scores: Sequence[float],
    sufficient_scores: Sequence[float],
    restored_scores: Sequence[float],
    *,
    denominator_floor: float = 1e-8,
) -> MediationEstimate:
    """Compute sufficiency and necessity without averaging unstable ratios."""

    clean = _one_dimensional(clean_scores, "clean_scores")
    treated = _one_dimensional(treated_scores, "treated_scores")
    sufficient = _one_dimensional(sufficient_scores, "sufficient_scores")
    restored = _one_dimensional(restored_scores, "restored_scores")
    if not (clean.shape == treated.shape == sufficient.shape == restored.shape):
        raise ValueError("all mediation score arrays must have the same shape")
    effect = float(np.sum(treated - clean))
    eligible = bool(np.isfinite(effect) and abs(effect) > denominator_floor)
    if not eligible:
        return MediationEstimate(effect, float("nan"), float("nan"), False)
    sufficiency = float(np.sum(sufficient - clean) / effect)
    necessity = float(np.sum(treated - restored) / effect)
    return MediationEstimate(effect, sufficiency, necessity, True)


def clustered_bootstrap_mediation(
    clean_scores: Sequence[float],
    treated_scores: Sequence[float],
    sufficient_scores: Sequence[float],
    restored_scores: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> Mapping[str, float | int]:
    """Episode-clustered bootstrap intervals for the two mediation ratios."""

    arrays = [
        _one_dimensional(clean_scores, "clean_scores"),
        _one_dimensional(treated_scores, "treated_scores"),
        _one_dimensional(sufficient_scores, "sufficient_scores"),
        _one_dimensional(restored_scores, "restored_scores"),
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("all mediation score arrays must have the same shape")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    sufficiency: list[float] = []
    necessity: list[float] = []
    rows = arrays[0].shape[0]
    for _ in range(draws):
        selection = rng.integers(0, rows, size=rows)
        estimate = mediation_estimate(*(array[selection] for array in arrays))
        if estimate.eligible:
            sufficiency.append(estimate.sufficiency)
            necessity.append(estimate.necessity)
    if not sufficiency:
        raise ValueError("no eligible bootstrap draw had a nonzero aggregate treatment effect")
    return {
        "draws_requested": draws,
        "draws_eligible": len(sufficiency),
        "sufficiency_p025": float(np.quantile(sufficiency, 0.025)),
        "sufficiency_p975": float(np.quantile(sufficiency, 0.975)),
        "necessity_p025": float(np.quantile(necessity, 0.025)),
        "necessity_p975": float(np.quantile(necessity, 0.975)),
    }


def select_train_prefix(
    ranking: Sequence[str],
    prefix_estimates: Mapping[int, MediationEstimate],
    *,
    max_nodes: int = 4,
    sufficiency_min: float = 0.60,
    necessity_min: float = 0.60,
) -> PrefixSelection:
    """Freeze the smallest eligible mediator prefix using training data only."""

    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate nodes")
    limit = min(max_nodes, len(ranking))
    for size in range(1, limit + 1):
        estimate = prefix_estimates.get(size)
        if (
            estimate is not None
            and estimate.eligible
            and estimate.sufficiency >= sufficiency_min
            and estimate.necessity >= necessity_min
        ):
            return PrefixSelection(tuple(ranking[:size]), True, "train_threshold_reached")
    return PrefixSelection(
        tuple(ranking[:limit]),
        False,
        "INELIGIBLE_MEDIATION: no train prefix reached both thresholds",
    )


def monte_carlo_upper_tail_p(observed: float, null_values: Sequence[float]) -> float:
    """Finite-sample corrected upper-tail Monte Carlo p-value."""

    null = _one_dimensional(null_values, "null_values")
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def assert_disjoint_pools(pools: Mapping[str, Sequence[str]]) -> None:
    """Fail closed if any registered token leaks across split pools."""

    owners: dict[str, str] = {}
    for split, values in pools.items():
        for value in values:
            if value in owners:
                raise ValueError(f"token {value!r} appears in {owners[value]!r} and {split!r}")
            owners[value] = split


def _one_dimensional(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array
