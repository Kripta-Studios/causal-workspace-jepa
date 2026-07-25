from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_evaluator import (
    DirectMediationOutcome,
    aggregate_direct_mediation_outcomes,
    atp_star_graddrop_scores,
    autograd_episode_estimators,
    binding_candidate_nodes,
    compare_population_mediation,
    compare_specificity_controls,
    decide_h_llm_15,
    decide_h_llm_16,
    delta_norm_scores,
    deterministic_random_scores,
    directional_hvp_scores,
    exact_local_atp_scores,
    execute_direct_mediation_episode,
    freeze_ranking,
    leave_value_out_probe_scores,
    matched_random_sets,
    norm_matched_resampled_states,
    population_atp_scores,
    state_patch_program,
)


def _synthetic_direct_outcomes(
    *,
    q_sufficiency: float,
    n_necessity: float,
    count: int = 40,
    sufficient_transfer: float = 1.0,
    restoration_transfer: float = 0.0,
) -> tuple[DirectMediationOutcome, ...]:
    """Create heterogeneous positive effects with exactly controlled Q and N."""

    rows: list[DirectMediationOutcome] = []
    sufficient_count = round(sufficient_transfer * count)
    restoration_count = round(restoration_transfer * count)
    for index in range(count):
        clean = -0.25 + 0.01 * (index % 5)
        effect = 0.75 + 0.05 * (index % 7)
        treated = clean + effect
        rows.append(
            DirectMediationOutcome(
                clean_score=clean,
                treated_score=treated,
                sufficient_score=clean + q_sufficiency * effect,
                restored_score=treated - n_necessity * effect,
                donor_score=treated + 0.1,
                clean_top_token=3,
                treated_top_token=7,
                sufficient_top_token=7 if index < sufficient_count else 3,
                restored_top_token=7 if index < restoration_count else 3,
                donor_top_token=7,
            )
        )
    return tuple(rows)


class QwenBindingMediationEvaluatorTests(unittest.TestCase):
    def test_candidate_roster_and_tie_break_are_frozen(self) -> None:
        nodes = binding_candidate_nodes()
        self.assertEqual(len(nodes), 56)
        self.assertEqual(nodes[0].site, "blocks.0.attn_out")
        self.assertEqual(nodes[1].site, "blocks.0.mlp_out")
        self.assertEqual(nodes[-1].site, "blocks.27.mlp_out")
        ranking = freeze_ranking("tied", np.ones(56), nodes=nodes)
        self.assertEqual(ranking.ordered_sites, tuple(node.site for node in nodes))

    def test_local_and_population_atp_are_distinct_estimators(self) -> None:
        deltas = np.asarray([[[1.0]], [[1.0]]])
        gradients = np.asarray([[[1.0]], [[-1.0]]])
        np.testing.assert_allclose(exact_local_atp_scores(deltas, gradients), [1.0])
        np.testing.assert_allclose(population_atp_scores(deltas, gradients), [0.0])

    def test_directional_hvp_uses_registered_half_coefficient(self) -> None:
        first = np.asarray([[1.0, -1.0], [3.0, -3.0]])
        second = np.asarray([[2.0, 2.0], [2.0, 2.0]])
        np.testing.assert_allclose(directional_hvp_scores(first, second), [3.0, 1.0])

    def test_atp_star_uses_sum_over_drops_divided_by_l_minus_one(self) -> None:
        effects = np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        # L=2, so the denominator is one; then average episodes.
        np.testing.assert_allclose(atp_star_graddrop_scores(effects), [8.0, 10.0])
        with self.assertRaisesRegex(ValueError, "roster"):
            atp_star_graddrop_scores(np.ones((2, 3, 2)))

    def test_delta_norm_is_noncausal_magnitude_baseline(self) -> None:
        deltas = np.asarray([[[3.0, 4.0], [0.0, 2.0]], [[0.0, 0.0], [0.0, 4.0]]])
        np.testing.assert_allclose(delta_norm_scores(deltas), [2.5, 3.0])

    def test_leave_value_out_probe_generalizes_an_effect_signal(self) -> None:
        rng = np.random.default_rng(17)
        labels = np.repeat(np.arange(4), 12)
        targets = np.linspace(-1.0, 1.0, labels.size)
        deltas = rng.normal(size=(labels.size, 2, 8)) * 0.01
        deltas[:, 0, 0] = targets
        scores = leave_value_out_probe_scores(
            deltas,
            targets,
            labels,
            projection_dim=3,
            projection_seed=19,
            ridge=1e-3,
        )
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_random_ranking_is_seeded_and_permutation_like(self) -> None:
        first = deterministic_random_scores(56, seed=467)
        second = deterministic_random_scores(56, seed=467)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 56)

    def test_matched_random_sets_preserve_every_stratum(self) -> None:
        selected = (
            "blocks.1.attn_out",
            "blocks.6.attn_out",
            "blocks.15.mlp_out",
        )
        controls = matched_random_sets(selected, count=128, seed=471)
        self.assertEqual(len(controls), 128)
        self.assertNotIn(frozenset(selected), {frozenset(control) for control in controls})
        nodes = {node.site: node for node in binding_candidate_nodes()}
        selected_strata = sorted(nodes[site].stratum for site in selected)
        for control in controls:
            self.assertEqual(sorted(nodes[site].stratum for site in control), selected_strata)

    def test_matched_draws_allow_repetition_when_unique_space_is_too_small(self) -> None:
        selected = tuple(f"blocks.{layer}.attn_out" for layer in range(4))
        controls = matched_random_sets(selected, count=128, seed=471)
        self.assertEqual(len(controls), 128)
        self.assertLess(len(set(controls)), 128)

    def test_state_patch_program_is_replayable_and_final_query_only(self) -> None:
        program = state_patch_program(
            ["blocks.3.attn_out", "blocks.9.mlp_out"],
            donor_prefix="test-0001:sufficient",
            seed=419,
        )
        self.assertEqual(len(program), 2)
        self.assertEqual(program[0].positions, (-1,))
        self.assertEqual(program[0].operation, "patch")
        self.assertEqual(
            program[0].donor_example_id,
            "test-0001:sufficient:blocks.3.attn_out",
        )
        self.assertEqual(program[0].to_dict(), program[0].to_dict())

    def test_norm_resampling_matches_each_target_norm_exactly(self) -> None:
        rng = np.random.default_rng(23)
        clean = rng.normal(size=(5, 3, 4)).astype(np.float32)
        target = rng.normal(size=(5, 3, 4)).astype(np.float32)
        train = rng.normal(size=(20, 3, 4)).astype(np.float32)
        states_a = norm_matched_resampled_states(
            clean,
            target,
            train,
            seed=491,
            bins=4,
        )
        states_b = norm_matched_resampled_states(
            clean,
            target,
            train,
            seed=491,
            bins=4,
        )
        np.testing.assert_array_equal(states_a, states_b)
        np.testing.assert_allclose(
            np.linalg.norm(states_a - clean, axis=-1),
            np.linalg.norm(target, axis=-1),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_autograd_estimators_return_exact_hvp_and_graddrop_contract(self) -> None:
        import torch

        class AnalyticAdapter:
            def forward_with_cache(self, _batch: object, sites: list[str]) -> object:
                source = torch.tensor([[[-0.4, 0.3]]], requires_grad=True)
                downstream = source + torch.tanh(source)
                score = downstream.square().sum()
                logits = torch.stack(
                    [torch.zeros_like(score), torch.zeros_like(score), score]
                ).reshape(1, 1, 3)
                activations = {"site0": source, "site1": downstream}
                return SimpleNamespace(
                    logits=logits,
                    activations={site: activations[site] for site in sites if site != "logits"},
                )

        delta = np.asarray([[0.2, -0.1], [0.1, 0.3]], dtype=np.float32)
        estimates = autograd_episode_estimators(
            AnalyticAdapter(),
            object(),
            delta,
            recipient_answer_id=1,
            donor_answer_id=2,
            candidate_sites=("site0", "site1"),
        )
        self.assertEqual(estimates.local_gradients.shape, (2, 2))
        self.assertEqual(estimates.directional_hvp_terms.shape, (2,))
        self.assertEqual(estimates.graddrop_effects.shape, (2, 2))
        np.testing.assert_allclose(np.diag(estimates.graddrop_effects), 0.0, atol=0.0)
        self.assertGreater(abs(float(estimates.directional_hvp_terms[0])), 0.0)
        self.assertGreater(abs(float(estimates.directional_hvp_terms[1])), 0.0)

    def test_graddrop_exposes_residual_path_cancellation(self) -> None:
        import torch

        class CancellingResidualAdapter:
            def forward_with_cache(self, _batch: object, sites: list[str]) -> object:
                first = torch.tensor([[[1.0]]], requires_grad=True)
                downstream_contribution = -0.9 * first
                residual = first + downstream_contribution
                score = residual.sum() + 0.01 * residual.square().sum()
                logits = torch.stack(
                    [torch.zeros_like(score), torch.zeros_like(score), score]
                ).reshape(1, 1, 3)
                activations = {"site0": first, "site1": downstream_contribution}
                return SimpleNamespace(
                    logits=logits,
                    activations={site: activations[site] for site in sites if site != "logits"},
                )

        estimates = autograd_episode_estimators(
            CancellingResidualAdapter(),
            object(),
            np.ones((2, 1), dtype=np.float32),
            recipient_answer_id=1,
            donor_answer_id=2,
            candidate_sites=("site0", "site1"),
        )
        local_source = abs(float(estimates.first_order_effects[0]))
        source_after_downstream_drop = abs(float(estimates.graddrop_effects[1, 0]))
        self.assertGreater(source_after_downstream_drop, 5.0 * local_source)
        self.assertAlmostEqual(float(estimates.graddrop_effects[0, 0]), 0.0)

    def test_estimators_fail_closed_on_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            exact_local_atp_scores(np.asarray([[[np.nan]]]), np.ones((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            directional_hvp_scores(np.asarray([[np.inf]]), np.ones((1, 1)))

    def test_irrelevant_patch_uses_final_query_source_at_position_zero(self) -> None:
        class PositionAdapter:
            def __init__(self) -> None:
                self.donors: dict[tuple[str, str], np.ndarray] = {}
                self.mediator_patches: list[tuple[int, float]] = []

            def tokenize(self, prompts: list[str]) -> str:
                return prompts[0]

            def forward_with_cache(self, batch: str, sites: list[str]) -> object:
                if batch == "donor":
                    activations = {"upstream": np.asarray([[[5.0], [6.0]]])}
                    logits = np.asarray([[[0.0, 0.0, 2.0]]])
                else:
                    activations = {
                        "upstream": np.asarray([[[1.0], [2.0]]]),
                        "mediator": np.asarray([[[10.0], [20.0]]]),
                    }
                    logits = np.asarray([[[0.0, 1.0, 0.0]]])
                return SimpleNamespace(
                    activations={site: activations[site] for site in sites if site != "logits"},
                    logits=logits,
                )

            def forward_with_interventions(
                self, _batch: str, specs: list[object], sites: list[str]
            ) -> object:
                if len(specs) == 1 and specs[0].site == "upstream":
                    activations = {"mediator": np.asarray([[[30.0], [40.0]]])}
                    logits = np.asarray([[[0.0, 0.0, 2.0]]])
                else:
                    mediator = next(spec for spec in specs if spec.site == "mediator")
                    value = float(
                        self.donors[(mediator.donor_example_id, "mediator")].item()
                    )
                    self.mediator_patches.append((mediator.positions[0], value))
                    activations = {}
                    logits = np.asarray([[[0.0, 0.5, 1.0]]])
                return SimpleNamespace(
                    activations={site: activations[site] for site in sites if site != "logits"},
                    logits=logits,
                )

            def register_donor(self, donor_id: str, site: str, value: object) -> None:
                self.donors[(donor_id, site)] = np.asarray(value)

            def unregister_donor(self, donor_id: str, site: str) -> None:
                del self.donors[(donor_id, site)]

        adapter = PositionAdapter()
        execute_direct_mediation_episode(
            adapter,
            recipient_prompt="recipient",
            donor_prompt="donor",
            treatment_site="upstream",
            treatment_positions=(1,),
            recipient_answer_id=1,
            donor_answer_id=2,
            mediator_sites=("mediator",),
            mediator_position=0,
            source_position=-1,
            seed=419,
        )
        self.assertEqual(adapter.mediator_patches, [(0, 40.0), (0, 20.0)])

    def test_planted_mediator_passes_direct_specificity_and_both_decisions(self) -> None:
        donor_ids = np.full(40, 7)
        population_outcomes = _synthetic_direct_outcomes(
            q_sufficiency=0.82,
            n_necessity=0.78,
        )
        comparator_outcomes = {
            method: _synthetic_direct_outcomes(
                q_sufficiency=0.44 - 0.01 * index,
                n_necessity=0.46 - 0.01 * index,
            )
            for index, method in enumerate(
                (
                    "exact_local_atp",
                    "directional_hvp",
                    "atp_star",
                    "leave_value_out_probe",
                    "delta_norm",
                )
            )
        }
        random_outcomes = tuple(
            _synthetic_direct_outcomes(
                q_sufficiency=0.25 + 0.001 * index,
                n_necessity=0.28 + 0.001 * index,
            )
            for index in range(128)
        )
        controls = {
            name: _synthetic_direct_outcomes(q_sufficiency=0.20, n_necessity=0.18)
            for name in (
                "donor_shuffle",
                "norm_matched_resample",
                "irrelevant_position",
                "unqueried_value_swap",
            )
        }
        aggregate = aggregate_direct_mediation_outcomes(
            population_outcomes,
            donor_ids,
            bootstrap_draws=512,
            bootstrap_seed=503,
            minimum_eligible_fraction=0.99,
        )
        comparison = compare_population_mediation(
            population_outcomes,
            comparator_outcomes,
            random_outcomes,
            donor_ids,
            bootstrap_draws=512,
            bootstrap_seed=509,
            minimum_eligible_fraction=0.99,
        )
        specificity = compare_specificity_controls(
            population_outcomes,
            controls,
            donor_ids,
        )

        self.assertTrue(aggregate.eligible)
        self.assertAlmostEqual(aggregate.q_sufficiency, 0.82)
        self.assertAlmostEqual(aggregate.n_necessity, 0.78)
        self.assertEqual(aggregate.sufficiency_transfer_gap, 0.0)
        self.assertEqual(aggregate.restoration_transfer_reduction, 1.0)
        self.assertGreater(comparison.paired_ci_lower, 0.0)
        self.assertEqual(
            set(comparison.paired_ci_lower_by_comparator), set(comparator_outcomes)
        )
        self.assertTrue(
            all(value > 0.0 for value in comparison.paired_ci_lower_by_comparator.values())
        )
        self.assertGreater(comparison.matched_random_margin, 0.0)
        self.assertAlmostEqual(comparison.monte_carlo_p, 1.0 / 129.0)
        self.assertGreaterEqual(specificity.minimum_margin, 0.20)

        comparisons = {"test": comparison, "paraphrase": comparison}
        aggregates = {"test": aggregate, "paraphrase": aggregate}
        specificities = {"test": specificity, "paraphrase": specificity}
        eligibility = {
            "train": True,
            "validation": True,
            "test": True,
            "paraphrase": True,
        }
        self.assertTrue(
            decide_h_llm_15(
                comparisons,
                task_eligibility=eligibility,
                population_prefix_eligible=True,
                required_bootstrap_draws=512,
            ).passed
        )
        self.assertTrue(
            decide_h_llm_16(
                aggregates,
                specificities,
                task_eligibility=eligibility,
                population_prefix_eligible=True,
                required_bootstrap_draws=512,
            ).passed
        )
        failed = decide_h_llm_15(
                comparisons,
                task_eligibility=eligibility,
                population_prefix_eligible=False,
                required_bootstrap_draws=512,
            )
        self.assertFalse(failed.passed)
        self.assertEqual(failed.evidence_level, "None")

    def test_total_effect_without_mediation_fails_closed_as_a_null(self) -> None:
        donor_ids = np.full(40, 7)
        null_outcomes = _synthetic_direct_outcomes(
            q_sufficiency=0.0,
            n_necessity=0.0,
            sufficient_transfer=0.0,
            restoration_transfer=1.0,
        )
        aggregate = aggregate_direct_mediation_outcomes(
            null_outcomes,
            donor_ids,
            bootstrap_draws=256,
            bootstrap_seed=521,
            minimum_eligible_fraction=0.99,
        )
        specificity = compare_specificity_controls(
            null_outcomes,
            {
                name: null_outcomes
                for name in (
                    "donor_shuffle",
                    "norm_matched_resample",
                    "irrelevant_position",
                    "unqueried_value_swap",
                )
            },
            donor_ids,
        )
        decision = decide_h_llm_16(
            {"test": aggregate, "paraphrase": aggregate},
            {"test": specificity, "paraphrase": specificity},
            task_eligibility={
                "train": True,
                "validation": True,
                "test": True,
                "paraphrase": True,
            },
            population_prefix_eligible=True,
        )
        self.assertTrue(aggregate.eligible)
        self.assertEqual(aggregate.q_sufficiency, 0.0)
        self.assertEqual(aggregate.n_necessity, 0.0)
        self.assertFalse(decision.passed)
        self.assertEqual(decision.evidence_level, "None")
        self.assertIn("test: failed Q floor", decision.reasons)

    def test_high_decodability_magnitude_distractor_is_rejected_by_execution(self) -> None:
        rng = np.random.default_rng(523)
        labels = np.repeat(np.arange(4), 12)
        effects = np.linspace(-1.0, 1.0, labels.size)
        deltas = rng.normal(scale=1e-3, size=(labels.size, 2, 8))
        deltas[:, 1, 0] = 20.0 * effects
        deltas[:, 1, 1] = 10.0
        magnitude = delta_norm_scores(deltas)
        availability = leave_value_out_probe_scores(
            deltas,
            effects,
            labels,
            projection_dim=3,
            projection_seed=527,
            ridge=1e-3,
        )
        self.assertEqual(int(np.argmax(magnitude)), 1)
        self.assertEqual(int(np.argmax(availability)), 1)

        donor_ids = np.full(labels.size, 7)
        distractor_outcomes = _synthetic_direct_outcomes(
            q_sufficiency=0.0,
            n_necessity=0.0,
            count=labels.size,
            sufficient_transfer=0.0,
            restoration_transfer=1.0,
        )
        aggregate = aggregate_direct_mediation_outcomes(
            distractor_outcomes,
            donor_ids,
            bootstrap_draws=256,
            bootstrap_seed=529,
            minimum_eligible_fraction=0.99,
        )
        self.assertEqual(aggregate.min_qn, 0.0)
        self.assertEqual(aggregate.sufficiency_donor_transfer, 0.0)
        self.assertEqual(aggregate.restoration_transfer_reduction, 0.0)

    def test_positive_bootstrap_fraction_and_missing_inputs_fail_closed(self) -> None:
        rows = tuple(
            DirectMediationOutcome(
                clean_score=0.0,
                treated_score=1.0 if index == 0 else -0.2,
                sufficient_score=0.0,
                restored_score=1.0 if index == 0 else -0.2,
                donor_score=1.0,
                clean_top_token=3,
                treated_top_token=7,
                sufficient_top_token=3,
                restored_top_token=7,
                donor_top_token=7,
            )
            for index in range(5)
        )
        aggregate = aggregate_direct_mediation_outcomes(
            rows,
            np.full(5, 7),
            bootstrap_draws=512,
            bootstrap_seed=541,
            minimum_eligible_fraction=0.99,
            treatment_effect_signed_mean_min=0.0,
        )
        self.assertFalse(aggregate.eligible)
        self.assertLess(aggregate.bootstrap_eligible_fraction, 0.99)

        decision = decide_h_llm_15(
            {},
            task_eligibility={
                "train": True,
                "validation": True,
                "test": True,
                "paraphrase": True,
            },
            population_prefix_eligible=True,
        )
        self.assertFalse(decision.passed)
        self.assertIn("test: missing population comparison", decision.reasons)
        self.assertEqual(decision.evidence_level, "None")

    def test_negative_treatment_denominator_returns_ineligible_result(self) -> None:
        rows = tuple(
            DirectMediationOutcome(
                clean_score=0.0,
                treated_score=-1.0,
                sufficient_score=-0.7,
                restored_score=-0.3,
                donor_score=-1.0,
                clean_top_token=3,
                treated_top_token=7,
                sufficient_top_token=7,
                restored_top_token=3,
                donor_top_token=7,
            )
            for _ in range(8)
        )
        aggregate = aggregate_direct_mediation_outcomes(
            rows,
            np.full(8, 7),
            bootstrap_draws=64,
            bootstrap_seed=547,
            minimum_eligible_fraction=0.99,
        )
        self.assertFalse(aggregate.eligible)
        self.assertEqual(aggregate.bootstrap_draws_eligible, 0)
        self.assertTrue(np.isnan(aggregate.q_sufficiency))
        self.assertTrue(np.isnan(aggregate.n_ci_lower))


if __name__ == "__main__":
    unittest.main()
