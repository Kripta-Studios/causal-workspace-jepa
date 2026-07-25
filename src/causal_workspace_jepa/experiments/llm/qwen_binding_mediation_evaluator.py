"""Frozen estimators and direct-test plans for Qwen binding mediation.

The functions in this module are deliberately outcome-agnostic.  They define
the train-only ranking estimators, deterministic tie breaking, matched random
sets, and intervention programs before the protected Qwen capture is opened.
Approximate rankings are never promoted to causal evidence without executing
the returned patch/restore programs on the language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.hooks.names import transformer_site


@dataclass(frozen=True, order=True)
class CandidateNode:
    """One registered post-projection residual contribution."""

    layer: int
    family_order: int
    site: str

    @property
    def family(self) -> str:
        return "attn_out" if self.family_order == 0 else "mlp_out"

    @property
    def layer_quartile(self) -> int:
        return min(3, (4 * self.layer) // 28)

    @property
    def stratum(self) -> tuple[str, int]:
        return self.family, self.layer_quartile


@dataclass(frozen=True)
class FrozenRanking:
    """A deterministic ranking and its raw finite node scores."""

    method: str
    ordered_sites: tuple[str, ...]
    scores: Mapping[str, float]


@dataclass(frozen=True)
class EpisodeDerivativeEstimates:
    """Lossless derivatives and per-node finite-edit terms for one episode."""

    local_gradients: np.ndarray
    first_order_effects: np.ndarray
    directional_hvp_terms: np.ndarray
    graddrop_effects: np.ndarray
    clean_candidate: np.ndarray
    clean_score: float


@dataclass(frozen=True)
class DirectMediationOutcome:
    """Four directly executed score states for one episode and mediator set."""

    clean_score: float
    treated_score: float
    sufficient_score: float
    restored_score: float
    donor_score: float
    clean_top_token: int
    treated_top_token: int
    sufficient_top_token: int
    restored_top_token: int
    donor_top_token: int


@dataclass(frozen=True)
class DirectMediationAggregate:
    """Fail-closed aggregate of directly executed mediator outcomes.

    ``q_sufficiency`` and ``n_necessity`` are ratios of sums, never means of
    per-episode ratios.  The aggregate is eligible only when the observed and
    a preregistered fraction of bootstrap treatment effects are positive.
    """

    outcome_count: int
    treatment_effect_sum: float
    treatment_effect_mean: float
    q_sufficiency: float
    n_necessity: float
    q_ci_lower: float
    q_ci_upper: float
    n_ci_lower: float
    n_ci_upper: float
    bootstrap_draws_requested: int
    bootstrap_draws_eligible: int
    bootstrap_eligible_fraction: float
    clean_donor_transfer: float
    donor_reference_transfer: float
    treatment_donor_transfer: float
    sufficiency_donor_transfer: float
    restoration_donor_transfer: float
    sufficiency_transfer_gap: float
    restoration_transfer_reduction: float
    eligible: bool
    failure_reasons: tuple[str, ...]

    @property
    def min_qn(self) -> float:
        """Return the conservative joint mediation endpoint."""

        return float(min(self.q_sufficiency, self.n_necessity))


@dataclass(frozen=True)
class PopulationMediationComparison:
    """Protected-split population ranking comparison using direct outcomes."""

    population: DirectMediationAggregate
    comparator_min_qn: Mapping[str, float]
    comparator_qn_margins: Mapping[str, float]
    paired_ci_lower_by_comparator: Mapping[str, float]
    paired_ci_upper_by_comparator: Mapping[str, float]
    paired_draws_eligible_by_comparator: Mapping[str, int]
    paired_eligible_fraction_by_comparator: Mapping[str, float]
    best_comparator_method: str | None
    best_comparator_min_qn: float
    qn_margin: float
    paired_ci_lower: float
    paired_ci_upper: float
    paired_draws_requested: int
    paired_draws_eligible: int
    paired_eligible_fraction: float
    matched_random_min_qn: tuple[float, ...]
    matched_random_p99: float
    matched_random_margin: float
    monte_carlo_p: float
    eligible: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SpecificityComparison:
    """Direct correct-donor advantage over each causal control."""

    component_margins: Mapping[str, tuple[float, float]]
    conservative_margins: Mapping[str, float]
    minimum_margin: float
    eligible: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class HypothesisDecision:
    """Pure, auditable hypothesis decision with no model or file access."""

    hypothesis: str
    passed: bool
    evidence_level: str
    split_pass: Mapping[str, bool]
    reasons: tuple[str, ...]


def binding_candidate_nodes(num_layers: int = 28) -> tuple[CandidateNode, ...]:
    """Return the canonical layer-major, attention-before-MLP node roster."""

    if num_layers != 28:
        raise ValueError("the preregistered binding study requires exactly 28 layers")
    return tuple(
        CandidateNode(
            layer=layer,
            family_order=family_order,
            site=transformer_site(layer, family),
        )
        for layer in range(num_layers)
        for family_order, family in enumerate(("attn_out", "mlp_out"))
    )


def exact_local_atp_scores(
    deltas: np.ndarray, local_gradients: np.ndarray
) -> np.ndarray:
    """Mean absolute episode-local first-order attribution per node."""

    delta, gradients = _paired_node_tensors(deltas, local_gradients)
    effects = np.einsum("enh,enh->en", delta, gradients, optimize=True)
    return _finite_scores(np.mean(np.abs(effects), axis=0), "exact_local_atp")


def population_atp_scores(
    deltas: np.ndarray, local_gradients: np.ndarray
) -> np.ndarray:
    """Apply each episode delta to the train-population mean gradient."""

    delta, gradients = _paired_node_tensors(deltas, local_gradients)
    mean_gradient = np.mean(gradients, axis=0, dtype=np.float64)
    effects = np.einsum("enh,nh->en", delta, mean_gradient, optimize=True)
    return _finite_scores(np.mean(np.abs(effects), axis=0), "population_atp")


def directional_hvp_scores(
    first_order_effects: np.ndarray,
    directional_hvp_terms: np.ndarray,
    *,
    coefficient: float = 0.5,
) -> np.ndarray:
    """Aggregate ``g·delta + coefficient * delta^T H delta`` by episode."""

    first = _episode_node_matrix(first_order_effects, "first_order_effects")
    second = _episode_node_matrix(directional_hvp_terms, "directional_hvp_terms")
    if first.shape != second.shape:
        raise ValueError("first-order and HVP arrays must have identical shapes")
    if not np.isfinite(coefficient):
        raise ValueError("directional HVP coefficient must be finite")
    return _finite_scores(
        np.mean(np.abs(first + coefficient * second), axis=0),
        "directional_hvp",
    )


def atp_star_graddrop_scores(graddrop_effects: np.ndarray) -> np.ndarray:
    """Faithful GradDrop aggregation for the 56 sequential residual semilayers.

    ``graddrop_effects[e, d, n]`` is ``delta_n dot grad^(d)_n`` after the
    backward path through semilayer ``d`` has been set to zero while retaining
    its clean forward value.  Kramar et al.'s scaling is ``1 / (L - 1)``.
    The Q/K softmax correction is not applicable because the registered nodes
    are post-``o_proj`` attention outputs and MLP outputs.
    """

    effects = np.asarray(graddrop_effects, dtype=np.float64)
    if effects.ndim != 3 or effects.shape[0] == 0 or effects.shape[1] < 2:
        raise ValueError("graddrop_effects must have shape [episodes, drops>=2, nodes]")
    if effects.shape[1] != effects.shape[2]:
        raise ValueError("the frozen GradDrop roster must equal the candidate-node roster")
    if not np.all(np.isfinite(effects)):
        raise ValueError("graddrop_effects contains nonfinite values")
    per_episode = np.sum(np.abs(effects), axis=1) / (effects.shape[1] - 1)
    return _finite_scores(np.mean(per_episode, axis=0), "atp_star")


def delta_norm_scores(deltas: np.ndarray) -> np.ndarray:
    """Mean L2 activation displacement, a deliberately noncausal baseline."""

    delta = _node_tensor(deltas, "deltas")
    return _finite_scores(np.mean(np.linalg.norm(delta, axis=-1), axis=0), "delta_norm")


def leave_value_out_probe_scores(
    deltas: np.ndarray,
    treatment_effects: Sequence[float],
    donor_answer_ids: Sequence[int],
    *,
    projection_dim: int,
    projection_seed: int,
    ridge: float,
) -> np.ndarray:
    """Negative OOF MSE for a fixed projected ridge effect probe.

    Every donor answer value is held out in turn.  Projection and feature
    normalization are fit without behavioral labels; normalization and ridge
    coefficients use only the remaining folds.  The probe predicts the scalar
    full-treatment score effect and is a level-1 availability baseline.
    """

    delta = _node_tensor(deltas, "deltas")
    targets = _one_dimensional(treatment_effects, "treatment_effects")
    labels = np.asarray(donor_answer_ids, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != delta.shape[0]:
        raise ValueError("donor_answer_ids must align with episodes")
    if targets.shape[0] != delta.shape[0]:
        raise ValueError("treatment_effects must align with episodes")
    if projection_dim <= 0 or projection_dim > delta.shape[-1]:
        raise ValueError("projection_dim must be in [1, hidden_size]")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    groups = np.unique(labels)
    if groups.size < 2:
        raise ValueError("leave-value-out probing requires at least two donor values")
    if any(np.count_nonzero(labels != group) <= projection_dim for group in groups):
        raise ValueError("each probe fold needs more training rows than projected dimensions")

    rng = np.random.default_rng(projection_seed)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(projection_dim),
        size=(delta.shape[-1], projection_dim),
    )
    projected = np.einsum("enh,hp->enp", delta, projection, optimize=True)
    scores = np.empty(delta.shape[1], dtype=np.float64)
    for node in range(delta.shape[1]):
        predictions = np.empty(delta.shape[0], dtype=np.float64)
        for group in groups:
            held_out = labels == group
            train = ~held_out
            x_train = projected[train, node]
            x_test = projected[held_out, node]
            mean = np.mean(x_train, axis=0)
            scale = np.std(x_train, axis=0)
            scale = np.where(scale < 1e-8, 1.0, scale)
            train_design = np.column_stack(
                [np.ones(np.count_nonzero(train)), (x_train - mean) / scale]
            )
            test_design = np.column_stack(
                [np.ones(np.count_nonzero(held_out)), (x_test - mean) / scale]
            )
            penalty = np.eye(train_design.shape[1], dtype=np.float64) * ridge
            penalty[0, 0] = 0.0
            weights = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ targets[train],
            )
            predictions[held_out] = test_design @ weights
        scores[node] = -float(np.mean(np.square(predictions - targets)))
    return _finite_scores(scores, "leave_value_out_probe")


def deterministic_random_scores(node_count: int, *, seed: int) -> np.ndarray:
    """Freeze a reproducible random ordering without relying on sort stability."""

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(node_count)
    scores = np.empty(node_count, dtype=np.float64)
    scores[permutation] = np.arange(node_count, 0, -1, dtype=np.float64)
    return scores


def freeze_ranking(
    method: str,
    scores: Sequence[float],
    *,
    nodes: Sequence[CandidateNode] | None = None,
) -> FrozenRanking:
    """Rank descending by score with the preregistered structural tie break."""

    roster = tuple(nodes or binding_candidate_nodes())
    values = _finite_scores(np.asarray(scores, dtype=np.float64), method)
    if values.shape != (len(roster),):
        raise ValueError("ranking score count differs from the candidate roster")
    ordered_indices = sorted(
        range(len(roster)),
        key=lambda index: (-values[index], roster[index].layer, roster[index].family_order),
    )
    return FrozenRanking(
        method=method,
        ordered_sites=tuple(roster[index].site for index in ordered_indices),
        scores={node.site: float(values[index]) for index, node in enumerate(roster)},
    )


def matched_random_sets(
    selected_sites: Sequence[str],
    *,
    count: int,
    seed: int,
    nodes: Sequence[CandidateNode] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Draw i.i.d. controls preserving family and 7-layer quartile counts.

    Repeated sets are retained.  Requiring 128 unique sets is impossible for
    some registered strata (for example, only 34 non-target four-node subsets
    exist within one seven-node stratum), while i.i.d. Monte Carlo draws with
    replacement retain a valid finite-sample corrected randomization test.
    """

    roster = tuple(nodes or binding_candidate_nodes())
    by_site = {node.site: node for node in roster}
    selected = tuple(selected_sites)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_sites must be nonempty and unique")
    unknown = set(selected).difference(by_site)
    if unknown:
        raise ValueError(f"unknown selected sites: {sorted(unknown)}")
    if count <= 0:
        raise ValueError("random-set count must be positive")

    required: dict[tuple[str, int], int] = {}
    for site in selected:
        stratum = by_site[site].stratum
        required[stratum] = required.get(stratum, 0) + 1
    pools = {
        stratum: tuple(node for node in roster if node.stratum == stratum)
        for stratum in required
    }
    if any(len(pools[stratum]) < amount for stratum, amount in required.items()):
        raise ValueError("a matched stratum has fewer nodes than the selected set requires")

    rng = np.random.default_rng(seed)
    original = frozenset(selected)
    controls: list[tuple[str, ...]] = []
    attempts = 0
    max_attempts = max(10_000, count * 1_000)
    while len(controls) < count and attempts < max_attempts:
        attempts += 1
        sampled: list[CandidateNode] = []
        for stratum in sorted(required):
            pool = pools[stratum]
            chosen = rng.choice(len(pool), size=required[stratum], replace=False)
            sampled.extend(pool[int(index)] for index in np.atleast_1d(chosen))
        canonical = tuple(node.site for node in sorted(sampled))
        frozen = frozenset(canonical)
        if frozen == original:
            continue
        controls.append(canonical)
    if len(controls) != count:
        raise RuntimeError("could not generate the requested number of unique matched sets")
    return tuple(controls)


def state_patch_program(
    sites: Sequence[str],
    *,
    donor_prefix: str,
    seed: int,
    position: int = -1,
) -> tuple[InterventionSpec, ...]:
    """Build a replayable multi-site final-query state replacement program."""

    selected = tuple(sites)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("state patch sites must be nonempty and unique")
    return tuple(
        InterventionSpec(
            site=site,
            operation="patch",
            positions=(position,),
            donor_example_id=f"{donor_prefix}:{site}",
            seed=seed,
        )
        for site in selected
    )


def autograd_episode_estimators(
    adapter: object,
    batch: object,
    deltas: np.ndarray,
    *,
    recipient_answer_id: int,
    donor_answer_id: int,
    candidate_sites: Sequence[str] | None = None,
) -> EpisodeDerivativeEstimates:
    """Compute local AtP, exact directional HVP, and semilayer GradDrop.

    Model parameters should be frozen while the embedding output remains
    differentiable.  GradDrop zeroes the backward signal of one complete
    post-projection residual contribution tensor at a time; its clean forward
    value and every skip connection are preserved.
    """

    import torch

    sites = tuple(candidate_sites or (node.site for node in binding_candidate_nodes()))
    delta = np.asarray(deltas, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[0] != len(sites) or not np.all(np.isfinite(delta)):
        raise ValueError("deltas must be finite with shape [candidate_sites, hidden]")
    forward = getattr(adapter, "forward_with_cache", None)
    if forward is None:
        raise TypeError("adapter must expose forward_with_cache")
    run = forward(batch, [*sites, "logits"])
    activations = tuple(run.activations[site] for site in sites)
    if any(not torch.is_tensor(activation) for activation in activations):
        raise TypeError("autograd localization requires tensor activations")
    if any(not activation.requires_grad for activation in activations):
        raise RuntimeError("candidate activations must preserve autograd")
    logits = run.logits
    score = logits[0, -1, donor_answer_id] - logits[0, -1, recipient_answer_id]
    gradients = torch.autograd.grad(
        score,
        activations,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    directions = tuple(
        torch.as_tensor(delta[index], device=activation.device, dtype=activation.dtype)
        for index, activation in enumerate(activations)
    )
    first_order = torch.stack(
        [
            torch.sum(gradient[0, -1].to(dtype=torch.float32) * direction.float())
            for gradient, direction in zip(gradients, directions, strict=True)
        ]
    )
    hvp_terms: list[object] = []
    for activation, gradient, direction in zip(
        activations, gradients, directions, strict=True
    ):
        directional_first = torch.sum(gradient[0, -1] * direction)
        second_gradient = torch.autograd.grad(
            directional_first,
            activation,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hvp_terms.append(
            torch.sum(second_gradient[0, -1].float() * direction.float())
        )

    graddrop = torch.empty(
        (len(sites), len(sites)),
        device=logits.device,
        dtype=torch.float32,
    )
    for dropped, dropped_activation in enumerate(activations):
        handle = dropped_activation.register_hook(lambda gradient: torch.zeros_like(gradient))
        try:
            dropped_gradients = torch.autograd.grad(
                score,
                activations,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
        finally:
            handle.remove()
        graddrop[dropped] = torch.stack(
            [
                torch.zeros((), device=logits.device, dtype=torch.float32)
                if node_index == dropped
                else torch.sum(gradient[0, -1].float() * direction.float())
                for node_index, (gradient, direction) in enumerate(
                    zip(dropped_gradients, directions, strict=True)
                )
            ]
        )
    clean_candidate = torch.stack(
        [activation[0, -1].detach().float().cpu() for activation in activations]
    ).numpy()
    result = EpisodeDerivativeEstimates(
        local_gradients=torch.stack(
            [gradient[0, -1].detach().float().cpu() for gradient in gradients]
        ).numpy(),
        first_order_effects=first_order.detach().cpu().numpy(),
        directional_hvp_terms=torch.stack(hvp_terms).detach().cpu().numpy(),
        graddrop_effects=graddrop.detach().cpu().numpy(),
        clean_candidate=clean_candidate,
        clean_score=float(score.detach().float().cpu()),
    )
    for name in (
        "local_gradients",
        "first_order_effects",
        "directional_hvp_terms",
        "graddrop_effects",
        "clean_candidate",
    ):
        if not np.all(np.isfinite(getattr(result, name))):
            raise FloatingPointError(f"autograd estimator produced nonfinite {name}")
    return result


def execute_direct_mediation_episode(
    adapter: object,
    *,
    recipient_prompt: str,
    donor_prompt: str,
    treatment_site: str,
    treatment_positions: Sequence[int],
    recipient_answer_id: int,
    donor_answer_id: int,
    mediator_sites: Sequence[str],
    seed: int,
    mediator_position: int = -1,
    source_position: int = -1,
    sufficient_states: Mapping[str, np.ndarray] | None = None,
    restore_states: Mapping[str, np.ndarray] | None = None,
) -> DirectMediationOutcome:
    """Execute treatment, mediator sufficiency, and mediator restoration.

    The default sufficient states are the states produced by the complete
    upstream treatment at ``source_position`` in the same episode; default
    restore states are from the clean recipient at that source position. They
    are written to ``mediator_position``. Explicit state maps support frozen
    shuffled/resampled controls while retaining the identical clean/treated
    denominator.
    """

    import torch

    sites = tuple(mediator_sites)
    if not sites or len(set(sites)) != len(sites):
        raise ValueError("mediator_sites must be nonempty and unique")
    if not isinstance(mediator_position, int):
        raise TypeError("mediator_position must be an integer token index")
    if not isinstance(source_position, int):
        raise TypeError("source_position must be an integer token index")
    tokenize = getattr(adapter, "tokenize", None)
    forward = getattr(adapter, "forward_with_cache", None)
    forward_many = getattr(adapter, "forward_with_interventions", None)
    register = getattr(adapter, "register_donor", None)
    unregister = getattr(adapter, "unregister_donor", None)
    if any(value is None for value in (tokenize, forward, forward_many, register, unregister)):
        raise TypeError("adapter lacks the direct mediation interface")
    recipient_batch = tokenize([recipient_prompt])
    donor_batch = tokenize([donor_prompt])
    clean = forward(recipient_batch, [*sites, treatment_site, "logits"])
    donor = forward(donor_batch, [treatment_site, "logits"])
    upstream_id = "direct-mediation:upstream"
    register(upstream_id, treatment_site, donor.activations[treatment_site])
    upstream = InterventionSpec(
        site=treatment_site,
        operation="patch",
        positions=tuple(int(position) for position in treatment_positions),
        donor_example_id=upstream_id,
        seed=seed,
    )
    registered: list[tuple[str, str]] = [(upstream_id, treatment_site)]
    try:
        treated = forward_many(recipient_batch, [upstream], [*sites, "logits"])
        sufficient_source = (
            {site: treated.activations[site][0, source_position] for site in sites}
            if sufficient_states is None
            else sufficient_states
        )
        restore_source = (
            {site: clean.activations[site][0, source_position] for site in sites}
            if restore_states is None
            else restore_states
        )
        if set(sufficient_source) != set(sites) or set(restore_source) != set(sites):
            raise ValueError("state maps must contain exactly the mediator sites")
        sufficient_specs: list[InterventionSpec] = []
        restore_specs: list[InterventionSpec] = []
        for site in sites:
            sufficient_id = f"direct-mediation:sufficient:{site}"
            restore_id = f"direct-mediation:restore:{site}"
            register(sufficient_id, site, sufficient_source[site])
            register(restore_id, site, restore_source[site])
            registered.extend(((sufficient_id, site), (restore_id, site)))
            sufficient_specs.append(
                InterventionSpec(
                    site=site,
                    operation="patch",
                    positions=(mediator_position,),
                    donor_example_id=sufficient_id,
                    seed=seed,
                )
            )
            restore_specs.append(
                InterventionSpec(
                    site=site,
                    operation="patch",
                    positions=(mediator_position,),
                    donor_example_id=restore_id,
                    seed=seed,
                )
            )
        sufficient = forward_many(recipient_batch, sufficient_specs, ["logits"])
        restored = forward_many(recipient_batch, [upstream, *restore_specs], ["logits"])
    finally:
        for donor_id, site in reversed(registered):
            unregister(donor_id, site)

    def score(run: object) -> float:
        logits = getattr(run, "logits")
        value = logits[0, -1, donor_answer_id] - logits[0, -1, recipient_answer_id]
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        result = float(value)
        if not np.isfinite(result):
            raise FloatingPointError("direct mediation produced a nonfinite score")
        return result

    def top_token(run: object) -> int:
        logits = getattr(run, "logits")
        row = logits[0, -1]
        if torch.is_tensor(row):
            return int(row.detach().argmax().cpu())
        return int(np.asarray(row).argmax())

    return DirectMediationOutcome(
        clean_score=score(clean),
        treated_score=score(treated),
        sufficient_score=score(sufficient),
        restored_score=score(restored),
        donor_score=score(donor),
        clean_top_token=top_token(clean),
        treated_top_token=top_token(treated),
        sufficient_top_token=top_token(sufficient),
        restored_top_token=top_token(restored),
        donor_top_token=top_token(donor),
    )


def aggregate_direct_mediation_outcomes(
    outcomes: Sequence[DirectMediationOutcome],
    donor_answer_ids: Sequence[int],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
    minimum_eligible_fraction: float,
    treatment_effect_signed_mean_min: float = 1e-4,
    denominator_floor: float = 1e-8,
) -> DirectMediationAggregate:
    """Aggregate direct outcomes with positive-denominator bootstrap inference.

    Invalid numerical input is an implementation error and raises.  An absent,
    sign-reversed, or bootstrap-unstable treatment effect is a scientific
    ineligibility and returns an aggregate with explicit failure reasons and
    NaN mediation intervals.  This distinction prevents null effects from
    becoming favorable ratios while retaining a serializable negative result.
    """

    arrays = _direct_outcome_arrays(outcomes, donor_answer_ids)
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    if not np.isfinite(minimum_eligible_fraction) or not (
        0.0 <= minimum_eligible_fraction <= 1.0
    ):
        raise ValueError("minimum_eligible_fraction must be finite and in [0, 1]")
    if not np.isfinite(treatment_effect_signed_mean_min) or (
        treatment_effect_signed_mean_min < 0.0
    ):
        raise ValueError("treatment_effect_signed_mean_min must be finite and nonnegative")
    if not np.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive")

    clean = arrays["clean_score"]
    treated = arrays["treated_score"]
    sufficient = arrays["sufficient_score"]
    restored = arrays["restored_score"]
    effect_sum = float(np.sum(treated - clean, dtype=np.float64))
    effect_mean = float(np.mean(treated - clean, dtype=np.float64))
    reasons: list[str] = []
    if effect_sum <= denominator_floor:
        reasons.append("INELIGIBLE_TREATMENT_EFFECT: aggregate signed effect is not positive")
    if effect_mean < treatment_effect_signed_mean_min:
        reasons.append(
            "INELIGIBLE_TREATMENT_EFFECT: signed mean effect is below the registered floor"
        )

    q_sufficiency = float("nan")
    n_necessity = float("nan")
    if effect_sum > denominator_floor:
        q_sufficiency = float(np.sum(sufficient - clean, dtype=np.float64) / effect_sum)
        n_necessity = float(np.sum(treated - restored, dtype=np.float64) / effect_sum)

    bootstrap = _positive_ratio_bootstrap(
        clean,
        treated,
        sufficient,
        restored,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
        denominator_floor=denominator_floor,
    )
    eligible_fraction = bootstrap["eligible_count"] / bootstrap_draws
    if eligible_fraction < minimum_eligible_fraction:
        reasons.append(
            "INELIGIBLE_BOOTSTRAP: positive-denominator fraction is below the registered floor"
        )
    if bootstrap["eligible_count"] == 0:
        reasons.append("INELIGIBLE_BOOTSTRAP: no positive-denominator draw")

    donor = arrays["donor_answer_id"]
    clean_transfer = float(np.mean(arrays["clean_top_token"] == donor))
    donor_reference_transfer = float(np.mean(arrays["donor_top_token"] == donor))
    treatment_transfer = float(np.mean(arrays["treated_top_token"] == donor))
    sufficiency_transfer = float(np.mean(arrays["sufficient_top_token"] == donor))
    restoration_transfer = float(np.mean(arrays["restored_top_token"] == donor))
    return DirectMediationAggregate(
        outcome_count=int(clean.size),
        treatment_effect_sum=effect_sum,
        treatment_effect_mean=effect_mean,
        q_sufficiency=q_sufficiency,
        n_necessity=n_necessity,
        q_ci_lower=bootstrap["q_lower"],
        q_ci_upper=bootstrap["q_upper"],
        n_ci_lower=bootstrap["n_lower"],
        n_ci_upper=bootstrap["n_upper"],
        bootstrap_draws_requested=bootstrap_draws,
        bootstrap_draws_eligible=int(bootstrap["eligible_count"]),
        bootstrap_eligible_fraction=float(eligible_fraction),
        clean_donor_transfer=clean_transfer,
        donor_reference_transfer=donor_reference_transfer,
        treatment_donor_transfer=treatment_transfer,
        sufficiency_donor_transfer=sufficiency_transfer,
        restoration_donor_transfer=restoration_transfer,
        sufficiency_transfer_gap=abs(treatment_transfer - sufficiency_transfer),
        restoration_transfer_reduction=treatment_transfer - restoration_transfer,
        eligible=not reasons,
        failure_reasons=tuple(reasons),
    )


def compare_population_mediation(
    population_outcomes: Sequence[DirectMediationOutcome],
    comparator_outcomes: Mapping[str, Sequence[DirectMediationOutcome]],
    matched_random_outcomes: Sequence[Sequence[DirectMediationOutcome]],
    donor_answer_ids: Sequence[int],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
    minimum_eligible_fraction: float,
    treatment_effect_signed_mean_min: float = 1e-4,
    denominator_floor: float = 1e-8,
) -> PopulationMediationComparison:
    """Compare population localization to executed comparators and matched sets.

    Every comparator receives its own paired interval over identical episodes;
    the hypothesis decision is their conjunction and never selects a method on
    a protected split.  The observed best comparator is retained only as a
    descriptive label. Matched random sets use a finite-sample corrected upper-
    tail Monte Carlo test; ranking overlap never enters the endpoint.
    """

    if not comparator_outcomes:
        raise ValueError("comparator_outcomes must be nonempty")
    if not matched_random_outcomes:
        raise ValueError("matched_random_outcomes must be nonempty")
    population_arrays = _direct_outcome_arrays(population_outcomes, donor_answer_ids)
    comparator_arrays = {
        str(method): _direct_outcome_arrays(outcomes, donor_answer_ids)
        for method, outcomes in comparator_outcomes.items()
    }
    random_arrays = tuple(
        _direct_outcome_arrays(outcomes, donor_answer_ids)
        for outcomes in matched_random_outcomes
    )
    for method, arrays in comparator_arrays.items():
        _assert_paired_treatment(population_arrays, arrays, f"comparator {method}")
    for index, arrays in enumerate(random_arrays):
        _assert_paired_treatment(population_arrays, arrays, f"matched random set {index}")

    population = aggregate_direct_mediation_outcomes(
        population_outcomes,
        donor_answer_ids,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
        minimum_eligible_fraction=minimum_eligible_fraction,
        treatment_effect_signed_mean_min=treatment_effect_signed_mean_min,
        denominator_floor=denominator_floor,
    )
    comparator_min_qn = {
        method: _point_min_qn(arrays, denominator_floor=denominator_floor)
        for method, arrays in comparator_arrays.items()
    }
    ordered_comparators = sorted(
        (
            method
            for method, value in comparator_min_qn.items()
            if np.isfinite(value)
        ),
        key=lambda method: (-comparator_min_qn[method], method),
    )
    best_method = ordered_comparators[0] if ordered_comparators else None
    best_value = (
        comparator_min_qn[best_method] if best_method is not None else float("nan")
    )
    comparator_margins = {
        method: float(population.min_qn - value)
        for method, value in comparator_min_qn.items()
    }
    qn_margin = (
        float(min(comparator_margins.values()))
        if comparator_margins
        else float("nan")
    )
    paired_by_comparator = {
        method: _paired_min_qn_bootstrap(
            population_arrays,
            arrays,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            denominator_floor=denominator_floor,
        )
        for method, arrays in comparator_arrays.items()
    }
    paired_lower = {
        method: float(result["lower"])
        for method, result in paired_by_comparator.items()
    }
    paired_upper = {
        method: float(result["upper"])
        for method, result in paired_by_comparator.items()
    }
    paired_eligible = {
        method: int(result["eligible_count"])
        for method, result in paired_by_comparator.items()
    }
    paired_fractions = {
        method: count / bootstrap_draws for method, count in paired_eligible.items()
    }
    paired_fraction = min(paired_fractions.values(), default=0.0)
    random_values = tuple(
        _point_min_qn(arrays, denominator_floor=denominator_floor)
        for arrays in random_arrays
    )
    random_array = np.asarray(random_values, dtype=np.float64)
    finite_random = bool(np.all(np.isfinite(random_array)))
    random_p99 = float(np.quantile(random_array, 0.99)) if finite_random else float("nan")
    monte_carlo_p = (
        float(
            (1 + np.count_nonzero(random_array >= population.min_qn))
            / (random_array.size + 1)
        )
        if finite_random and np.isfinite(population.min_qn)
        else float("nan")
    )

    reasons = list(population.failure_reasons)
    if not all(np.isfinite(value) for value in comparator_min_qn.values()):
        reasons.append("INELIGIBLE_COMPARATORS: a direct comparator endpoint is nonfinite")
    if best_method is None:
        reasons.append("INELIGIBLE_COMPARATORS: no finite direct comparator endpoint")
    if not finite_random:
        reasons.append("INELIGIBLE_MATCHED_RANDOM: nonfinite direct random-set endpoint")
    if any(count == 0 for count in paired_eligible.values()) or not paired_eligible:
        reasons.append(
            "INELIGIBLE_PAIRED_BOOTSTRAP: a comparator has no positive-denominator draw"
        )
    if any(
        fraction < minimum_eligible_fraction for fraction in paired_fractions.values()
    ) or not paired_fractions:
        reasons.append(
            "INELIGIBLE_PAIRED_BOOTSTRAP: a comparator eligible fraction is below the registered floor"
        )
    return PopulationMediationComparison(
        population=population,
        comparator_min_qn=comparator_min_qn,
        comparator_qn_margins=comparator_margins,
        paired_ci_lower_by_comparator=paired_lower,
        paired_ci_upper_by_comparator=paired_upper,
        paired_draws_eligible_by_comparator=paired_eligible,
        paired_eligible_fraction_by_comparator=paired_fractions,
        best_comparator_method=best_method,
        best_comparator_min_qn=float(best_value),
        qn_margin=float(qn_margin),
        paired_ci_lower=min(paired_lower.values(), default=float("nan")),
        paired_ci_upper=min(paired_upper.values(), default=float("nan")),
        paired_draws_requested=bootstrap_draws,
        paired_draws_eligible=min(paired_eligible.values(), default=0),
        paired_eligible_fraction=float(paired_fraction),
        matched_random_min_qn=random_values,
        matched_random_p99=random_p99,
        matched_random_margin=float(population.min_qn - random_p99),
        monte_carlo_p=monte_carlo_p,
        eligible=not reasons,
        failure_reasons=tuple(reasons),
    )


def compare_specificity_controls(
    primary_outcomes: Sequence[DirectMediationOutcome],
    control_outcomes: Mapping[str, Sequence[DirectMediationOutcome]],
    donor_answer_ids: Sequence[int],
    *,
    denominator_floor: float = 1e-8,
) -> SpecificityComparison:
    """Compare direct Q and N numerators to identically executed controls."""

    if not control_outcomes:
        raise ValueError("control_outcomes must be nonempty")
    primary = _direct_outcome_arrays(primary_outcomes, donor_answer_ids)
    primary_q, primary_n = _point_qn(primary, denominator_floor=denominator_floor)
    reasons: list[str] = []
    if not np.isfinite(primary_q) or not np.isfinite(primary_n):
        reasons.append("INELIGIBLE_SPECIFICITY: primary treatment effect is not positive")
    component: dict[str, tuple[float, float]] = {}
    conservative: dict[str, float] = {}
    for name, outcomes in control_outcomes.items():
        control = _direct_outcome_arrays(outcomes, donor_answer_ids)
        _assert_paired_treatment(primary, control, f"specificity control {name}")
        control_q, control_n = _point_qn(control, denominator_floor=denominator_floor)
        if not np.isfinite(control_q) or not np.isfinite(control_n):
            reasons.append(
                f"INELIGIBLE_SPECIFICITY: control {name} has a nonpositive or nonfinite denominator"
            )
        q_margin = float(primary_q - control_q)
        n_margin = float(primary_n - control_n)
        component[str(name)] = (q_margin, n_margin)
        conservative[str(name)] = min(q_margin, n_margin)
    minimum = (
        float(min(conservative.values()))
        if conservative and all(np.isfinite(value) for value in conservative.values())
        else float("nan")
    )
    return SpecificityComparison(
        component_margins=component,
        conservative_margins=conservative,
        minimum_margin=minimum,
        eligible=not reasons,
        failure_reasons=tuple(reasons),
    )


def decide_h_llm_15(
    comparisons: Mapping[str, PopulationMediationComparison],
    *,
    task_eligibility: Mapping[str, bool],
    population_prefix_eligible: bool,
    protected_splits: Sequence[str] = ("test", "paraphrase"),
    required_eligibility_splits: Sequence[str] = (
        "train",
        "validation",
        "test",
        "paraphrase",
    ),
    required_comparators: Sequence[str] = (
        "exact_local_atp",
        "directional_hvp",
        "atp_star",
        "leave_value_out_probe",
        "delta_norm",
    ),
    qn_margin_min: float = 0.10,
    paired_ci_lower_min: float = 0.0,
    matched_random_margin_min: float = 0.0,
    monte_carlo_alpha: float = 0.01,
    required_bootstrap_draws: int = 10_000,
    minimum_bootstrap_eligible_fraction: float = 0.99,
    required_matched_random_count: int = 128,
) -> HypothesisDecision:
    """Apply the preregistered H-LLM-15 rule without external state."""

    split_pass: dict[str, bool] = {}
    reasons: list[str] = []
    missing_eligibility = set(required_eligibility_splits).difference(task_eligibility)
    all_task_eligible = not missing_eligibility and all(
        bool(task_eligibility[split]) for split in required_eligibility_splits
    )
    if not all_task_eligible:
        reasons.append(
            "global: not all registered task eligibility gates passed"
            + (f" ({sorted(missing_eligibility)})" if missing_eligibility else "")
        )
    for split in protected_splits:
        comparison = comparisons.get(split)
        if comparison is None:
            split_pass[split] = False
            reasons.append(f"{split}: missing population comparison")
            continue
        missing = set(required_comparators).difference(comparison.comparator_min_qn)
        gates = {
            "all task splits eligible": all_task_eligible,
            "train-selected population prefix eligible": population_prefix_eligible,
            "comparison eligible": comparison.eligible,
            "all comparators present": not missing,
            "all comparators finite": all(
                np.isfinite(comparison.comparator_min_qn[method])
                for method in required_comparators
                if method in comparison.comparator_min_qn
            )
            and not missing,
            "registered bootstrap count": (
                comparison.population.bootstrap_draws_requested
                == required_bootstrap_draws
                and comparison.paired_draws_requested == required_bootstrap_draws
            ),
            "positive bootstrap fraction": (
                comparison.population.bootstrap_eligible_fraction
                >= minimum_bootstrap_eligible_fraction
                and comparison.paired_eligible_fraction
                >= minimum_bootstrap_eligible_fraction
            ),
            "registered matched-random count": (
                len(comparison.matched_random_min_qn)
                == required_matched_random_count
            ),
            "all min(Q,N) margins": all(
                comparison.comparator_qn_margins.get(method, float("-inf"))
                >= qn_margin_min
                for method in required_comparators
            ),
            "all paired bootstrap lower bounds": all(
                comparison.paired_ci_lower_by_comparator.get(
                    method, float("-inf")
                )
                > paired_ci_lower_min
                for method in required_comparators
            ),
            "matched-random p99": (
                comparison.matched_random_margin > matched_random_margin_min
            ),
            "Monte Carlo p": comparison.monte_carlo_p <= monte_carlo_alpha,
        }
        split_pass[split] = all(gates.values())
        for gate, passed in gates.items():
            if not passed:
                suffix = f" ({sorted(missing)})" if gate == "all comparators present" else ""
                reasons.append(f"{split}: failed {gate}{suffix}")
    passed = bool(split_pass) and all(split_pass.values())
    return HypothesisDecision(
        hypothesis="H-LLM-15",
        passed=passed,
        evidence_level="Specificity" if passed else "None",
        split_pass=split_pass,
        reasons=tuple(reasons),
    )


def decide_h_llm_16(
    aggregates: Mapping[str, DirectMediationAggregate],
    specificity: Mapping[str, SpecificityComparison],
    *,
    task_eligibility: Mapping[str, bool],
    population_prefix_eligible: bool,
    protected_splits: Sequence[str] = ("test", "paraphrase"),
    required_eligibility_splits: Sequence[str] = (
        "train",
        "validation",
        "test",
        "paraphrase",
    ),
    required_controls: Sequence[str] = (
        "donor_shuffle",
        "norm_matched_resample",
        "irrelevant_position",
        "unqueried_value_swap",
    ),
    mediation_min: float = 0.50,
    mediation_ci_lower_min: float = 0.35,
    treatment_transfer_gap_max: float = 0.15,
    restoration_transfer_reduction_min: float = 0.30,
    specificity_margin_fraction_min: float = 0.20,
    required_bootstrap_draws: int = 10_000,
    minimum_bootstrap_eligible_fraction: float = 0.99,
) -> HypothesisDecision:
    """Apply the preregistered H-LLM-16 rule without model or file access."""

    split_pass: dict[str, bool] = {}
    reasons: list[str] = []
    missing_eligibility = set(required_eligibility_splits).difference(task_eligibility)
    all_task_eligible = not missing_eligibility and all(
        bool(task_eligibility[split]) for split in required_eligibility_splits
    )
    if not all_task_eligible:
        reasons.append(
            "global: not all registered task eligibility gates passed"
            + (f" ({sorted(missing_eligibility)})" if missing_eligibility else "")
        )
    for split in protected_splits:
        aggregate = aggregates.get(split)
        control = specificity.get(split)
        if aggregate is None or control is None:
            split_pass[split] = False
            missing = "aggregate" if aggregate is None else "specificity comparison"
            reasons.append(f"{split}: missing {missing}")
            continue
        missing_controls = set(required_controls).difference(control.conservative_margins)
        gates = {
            "all task splits eligible": all_task_eligible,
            "train-selected population prefix eligible": population_prefix_eligible,
            "mediation aggregate eligible": aggregate.eligible,
            "specificity comparison eligible": control.eligible,
            "all controls present": not missing_controls,
            "registered bootstrap count": (
                aggregate.bootstrap_draws_requested == required_bootstrap_draws
            ),
            "positive bootstrap fraction": (
                aggregate.bootstrap_eligible_fraction
                >= minimum_bootstrap_eligible_fraction
            ),
            "Q floor": aggregate.q_sufficiency >= mediation_min,
            "N floor": aggregate.n_necessity >= mediation_min,
            "Q bootstrap lower bound": aggregate.q_ci_lower >= mediation_ci_lower_min,
            "N bootstrap lower bound": aggregate.n_ci_lower >= mediation_ci_lower_min,
            "sufficiency transfer recovery": (
                aggregate.sufficiency_transfer_gap <= treatment_transfer_gap_max
            ),
            "restoration transfer reduction": (
                aggregate.restoration_transfer_reduction
                >= restoration_transfer_reduction_min
            ),
            "specificity margin": (
                control.minimum_margin >= specificity_margin_fraction_min
            ),
        }
        split_pass[split] = all(gates.values())
        for gate, gate_passed in gates.items():
            if not gate_passed:
                suffix = (
                    f" ({sorted(missing_controls)})"
                    if gate == "all controls present"
                    else ""
                )
                reasons.append(f"{split}: failed {gate}{suffix}")
    passed = bool(split_pass) and all(split_pass.values())
    return HypothesisDecision(
        hypothesis="H-LLM-16",
        passed=passed,
        evidence_level="Specificity" if passed else "None",
        split_pass=split_pass,
        reasons=tuple(reasons),
    )


def norm_matched_resampled_states(
    clean_states: np.ndarray,
    target_deltas: np.ndarray,
    train_deltas: np.ndarray,
    *,
    seed: int,
    bins: int,
) -> np.ndarray:
    """Resample same-node train deltas by norm bin and match target L2 exactly."""

    output_dtype = np.asarray(clean_states).dtype
    clean = _node_tensor(clean_states, "clean_states")
    target = _node_tensor(target_deltas, "target_deltas")
    train = _node_tensor(train_deltas, "train_deltas")
    if clean.shape != target.shape:
        raise ValueError("clean_states and target_deltas must have identical shapes")
    if train.shape[1:] != target.shape[1:]:
        raise ValueError("train_deltas node and hidden dimensions must match targets")
    if bins <= 0:
        raise ValueError("bins must be positive")
    rng = np.random.default_rng(seed)
    result = np.empty_like(clean, dtype=np.float64)
    train_norms = np.linalg.norm(train, axis=-1)
    target_norms = np.linalg.norm(target, axis=-1)
    for node in range(target.shape[1]):
        edges = np.quantile(train_norms[:, node], np.linspace(0.0, 1.0, bins + 1))
        for episode in range(target.shape[0]):
            bin_index = int(
                np.clip(np.searchsorted(edges[1:-1], target_norms[episode, node]), 0, bins - 1)
            )
            candidates = np.flatnonzero(
                (train_norms[:, node] >= edges[bin_index])
                & (train_norms[:, node] <= edges[bin_index + 1])
            )
            if candidates.size == 0:
                candidates = np.arange(train.shape[0])
            sampled = train[int(rng.choice(candidates)), node].astype(np.float64, copy=True)
            sampled_norm = float(np.linalg.norm(sampled))
            desired_norm = float(target_norms[episode, node])
            if desired_norm == 0.0:
                sampled.fill(0.0)
            elif sampled_norm <= 1e-12:
                raise ValueError("cannot rescale a zero sampled delta to a nonzero target norm")
            else:
                sampled *= desired_norm / sampled_norm
            result[episode, node] = clean[episode, node] + sampled
    return result.astype(output_dtype, copy=False)


def _direct_outcome_arrays(
    outcomes: Sequence[DirectMediationOutcome],
    donor_answer_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    rows = tuple(outcomes)
    if not rows:
        raise ValueError("direct mediation outcomes must be nonempty")
    donor_ids = np.asarray(donor_answer_ids)
    if donor_ids.ndim != 1 or donor_ids.shape != (len(rows),):
        raise ValueError("donor_answer_ids must align with direct outcomes")
    if not np.issubdtype(donor_ids.dtype, np.integer):
        if not np.all(np.isfinite(donor_ids)) or not np.all(donor_ids == np.floor(donor_ids)):
            raise ValueError("donor_answer_ids must contain finite integer IDs")
    arrays: dict[str, np.ndarray] = {
        "donor_answer_id": donor_ids.astype(np.int64, copy=False),
    }
    score_fields = (
        "clean_score",
        "treated_score",
        "sufficient_score",
        "restored_score",
        "donor_score",
    )
    token_fields = (
        "clean_top_token",
        "treated_top_token",
        "sufficient_top_token",
        "restored_top_token",
        "donor_top_token",
    )
    for field in score_fields:
        values = np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"direct outcome field {field} contains nonfinite values")
        arrays[field] = values
    for field in token_fields:
        raw = np.asarray([getattr(row, field) for row in rows])
        if not np.issubdtype(raw.dtype, np.integer):
            if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
                raise ValueError(f"direct outcome field {field} contains invalid token IDs")
        arrays[field] = raw.astype(np.int64, copy=False)
    return arrays


def _positive_ratio_bootstrap(
    clean: np.ndarray,
    treated: np.ndarray,
    sufficient: np.ndarray,
    restored: np.ndarray,
    *,
    draws: int,
    seed: int,
    denominator_floor: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    q_values: list[float] = []
    n_values: list[float] = []
    rows = clean.size
    for _ in range(draws):
        selected = rng.integers(0, rows, size=rows)
        effect = float(np.sum(treated[selected] - clean[selected], dtype=np.float64))
        if effect <= denominator_floor:
            continue
        q_values.append(
            float(np.sum(sufficient[selected] - clean[selected], dtype=np.float64) / effect)
        )
        n_values.append(
            float(np.sum(treated[selected] - restored[selected], dtype=np.float64) / effect)
        )
    if not q_values:
        return {
            "eligible_count": 0,
            "q_lower": float("nan"),
            "q_upper": float("nan"),
            "n_lower": float("nan"),
            "n_upper": float("nan"),
        }
    return {
        "eligible_count": len(q_values),
        "q_lower": float(np.quantile(q_values, 0.025)),
        "q_upper": float(np.quantile(q_values, 0.975)),
        "n_lower": float(np.quantile(n_values, 0.025)),
        "n_upper": float(np.quantile(n_values, 0.975)),
    }


def _point_qn(
    arrays: Mapping[str, np.ndarray], *, denominator_floor: float
) -> tuple[float, float]:
    clean = arrays["clean_score"]
    treated = arrays["treated_score"]
    effect = float(np.sum(treated - clean, dtype=np.float64))
    if effect <= denominator_floor:
        return float("nan"), float("nan")
    q_value = float(
        np.sum(arrays["sufficient_score"] - clean, dtype=np.float64) / effect
    )
    n_value = float(
        np.sum(treated - arrays["restored_score"], dtype=np.float64) / effect
    )
    return q_value, n_value


def _point_min_qn(
    arrays: Mapping[str, np.ndarray], *, denominator_floor: float
) -> float:
    q_value, n_value = _point_qn(arrays, denominator_floor=denominator_floor)
    return float(min(q_value, n_value))


def _assert_paired_treatment(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    label: str,
) -> None:
    if reference["clean_score"].shape != candidate["clean_score"].shape:
        raise ValueError(f"{label} has a different episode count")
    for field in (
        "clean_score",
        "treated_score",
        "clean_top_token",
        "treated_top_token",
        "donor_answer_id",
    ):
        if not np.array_equal(reference[field], candidate[field]):
            raise ValueError(f"{label} does not share identical paired {field}")


def _paired_min_qn_bootstrap(
    population: Mapping[str, np.ndarray],
    comparator: Mapping[str, np.ndarray],
    *,
    draws: int,
    seed: int,
    denominator_floor: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    rows = population["clean_score"].size
    clean = population["clean_score"]
    treated = population["treated_score"]
    for _ in range(draws):
        selected = rng.integers(0, rows, size=rows)
        effect = float(np.sum(treated[selected] - clean[selected], dtype=np.float64))
        if effect <= denominator_floor:
            continue

        def selected_min_qn(arrays: Mapping[str, np.ndarray]) -> float:
            q_value = float(
                np.sum(
                    arrays["sufficient_score"][selected] - clean[selected],
                    dtype=np.float64,
                )
                / effect
            )
            n_value = float(
                np.sum(
                    treated[selected] - arrays["restored_score"][selected],
                    dtype=np.float64,
                )
                / effect
            )
            return min(q_value, n_value)

        differences.append(selected_min_qn(population) - selected_min_qn(comparator))
    if not differences:
        return {"eligible_count": 0, "lower": float("nan"), "upper": float("nan")}
    return {
        "eligible_count": len(differences),
        "lower": float(np.quantile(differences, 0.025)),
        "upper": float(np.quantile(differences, 0.975)),
    }


def _paired_node_tensors(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left = _node_tensor(first, "first")
    right = _node_tensor(second, "second")
    if left.shape != right.shape:
        raise ValueError("paired node tensors must have identical shapes")
    return left, right


def _node_tensor(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or any(dimension == 0 for dimension in array.shape):
        raise ValueError(f"{name} must have shape [episodes, nodes, hidden]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _episode_node_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or any(dimension == 0 for dimension in array.shape):
        raise ValueError(f"{name} must have shape [episodes, nodes]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _one_dimensional(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _finite_scores(values: np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} produced invalid node scores")
    return scores
