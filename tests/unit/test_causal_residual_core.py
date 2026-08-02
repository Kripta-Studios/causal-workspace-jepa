from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Callable

import torch

from causal_workspace_jepa.interpretability.differential_baselines import (
    AffineRidgeTransport,
    BaselineInputView,
    BaselineMetadata,
    BaselineSelectionRecord,
    ExactJVPBaseline,
    MeanEffectBaseline,
    NoChangeBaseline,
    ObservedPrefixRelinearizedBaseline,
    QuadraticHVPBaseline,
    select_train_validation_baseline,
)
from causal_workspace_jepa.models.causal_residual import (
    AffineStateTransform,
    BaselineClass,
    BaselineEligibilityError,
    CausalEffectBatch,
    CausalResidualJEPA,
    CausalResidualTarget,
    ConditionalLowRankRouter,
    NoRoutingLocalResidualPredictor,
    ParameterMatchedMLPResidualPredictor,
    ProtectedSplitAccessError,
    ResidualLossBundle,
    SplitRole,
    StandardCrossAttentionResidualPredictor,
    commutator_delta,
    commutator_interaction_loss,
    direct_replay_endpoint_loss,
    identity_residual_loss,
    inverse_restoration_loss,
    matched_control_specificity_loss,
    matched_norm_covariance_action_control,
    norm_matched_random_direction_control,
    normalized_residual_reconstruction_loss,
    reconstruct_treated,
    sequential_composition_loss,
    sequential_delta,
    uncertainty_calibration_nll,
)
from causal_workspace_jepa.experiments.cross_domain.causal_residual_synthetic import (
    TrainOnlyPolynomialResidualHead,
)


def _batch(
    source: torch.Tensor,
    action: torch.Tensor,
    transition: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    split: SplitRole,
    prefix: str,
) -> CausalEffectBatch:
    return CausalEffectBatch(
        source=source,
        treated=transition(source, action),
        intervention=action,
        split=split,
        example_ids=tuple(f"{prefix}-{index}" for index in range(source.shape[0])),
    )


class CausalResidualCoreTests(unittest.TestCase):
    def test_exact_effect_residual_and_reconstruction_identity(self) -> None:
        source = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        treated = torch.tensor([[2.5, -1.0], [-0.5, 2.5]])
        baseline = torch.tensor([[1.0, 0.25], [-0.25, -0.25]])
        batch = CausalEffectBatch(
            source=source,
            treated=treated,
            intervention=torch.ones((2, 1)),
            split=SplitRole.TRAIN,
        )
        target = CausalResidualTarget.from_effect(batch, baseline)
        torch.testing.assert_close(target.delta, treated - source)
        torch.testing.assert_close(target.residual, treated - source - baseline)
        torch.testing.assert_close(target.reconstruct(target.residual), treated)
        torch.testing.assert_close(reconstruct_treated(source, baseline, target.residual), treated)

    def test_affine_identity_inverse_sequential_and_noncommuting_controls(self) -> None:
        state = torch.tensor([[1.0, -2.0], [0.25, 0.75]], dtype=torch.float64)
        first = AffineStateTransform(
            torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
            torch.tensor([0.5, -0.25], dtype=torch.float64),
        )
        second = AffineStateTransform(
            torch.tensor([[1.0, 0.0], [-0.5, 1.0]], dtype=torch.float64),
            torch.tensor([-0.1, 0.4], dtype=torch.float64),
        )
        restored = first.inverse().apply(first.apply(state))
        torch.testing.assert_close(restored, state, atol=1e-12, rtol=1e-12)
        expected_sequence = second.apply(first.apply(state)) - state
        torch.testing.assert_close(sequential_delta(state, first, second), expected_sequence)
        interaction = commutator_delta(state, first, second)
        expected_interaction = second.apply(first.apply(state)) - first.apply(second.apply(state))
        torch.testing.assert_close(interaction, expected_interaction)
        self.assertGreater(float(interaction.abs().max()), 1e-5)
        self.assertEqual(float(inverse_restoration_loss(state, first, first.inverse())), 0.0)
        self.assertEqual(
            float(sequential_composition_loss(state, first, second, expected_sequence)), 0.0
        )
        self.assertEqual(float(commutator_interaction_loss(state, first, second, interaction)), 0.0)

    def test_composable_losses_and_matched_action_control(self) -> None:
        torch.manual_seed(4)
        target = torch.randn(12, 3)
        source = torch.randn(12, 3)
        baseline = torch.randn(12, 3)
        treated = reconstruct_treated(source, baseline, target)
        self.assertEqual(float(normalized_residual_reconstruction_loss(target, target)), 0.0)
        self.assertEqual(float(direct_replay_endpoint_loss(source, baseline, target, treated)), 0.0)
        self.assertEqual(float(identity_residual_loss(torch.zeros_like(target))), 0.0)
        self.assertEqual(
            float(matched_control_specificity_loss(torch.tensor(0.1), torch.tensor(0.4))), 0.0
        )
        self.assertGreater(
            float(matched_control_specificity_loss(torch.tensor(0.5), torch.tensor(0.4))), 0.0
        )
        uncertainty = uncertainty_calibration_nll(target, torch.zeros_like(target), target)
        self.assertEqual(float(uncertainty), 0.0)
        bundle = ResidualLossBundle(
            {"reconstruction": torch.tensor(1.5), "identity": torch.tensor(0.25)}
        )
        self.assertAlmostEqual(float(bundle.total({"reconstruction": 2.0, "identity": 4.0})), 4.0)

        actions = torch.randn(12, 2)
        control = matched_norm_covariance_action_control(actions, seed=77)
        self.assertLess(control.max_norm_difference, 1e-7)
        self.assertLess(control.max_covariance_difference, 1e-7)
        self.assertEqual(control.actions.shape, actions.shape)
        random_direction = norm_matched_random_direction_control(actions, seed=78)
        self.assertLess(random_direction.max_norm_difference, 1e-6)
        self.assertEqual(random_direction.actions.shape, actions.shape)

    def test_jvp_and_hvp_replay_equivalence(self) -> None:
        torch.manual_seed(9)
        source = torch.randn(20, 3, dtype=torch.float64)
        action = torch.randn(20, 2, dtype=torch.float64)
        linear = torch.tensor([[0.7, -0.2], [0.1, 0.6], [-0.4, 0.3]], dtype=torch.float64)
        quadratic = torch.tensor([[0.2, -0.1], [-0.15, 0.3], [0.05, 0.12]], dtype=torch.float64)

        def linear_transition(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            return state + control @ linear.T

        def quadratic_transition(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            return state + control @ linear.T + control.square() @ quadratic.T

        torch.testing.assert_close(
            ExactJVPBaseline(linear_transition).predict(source, action),
            linear_transition(source, action) - source,
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            QuadraticHVPBaseline(quadratic_transition).predict(source, action),
            quadratic_transition(source, action) - source,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_train_validation_selection_is_hashed_deterministic_and_fail_closed(self) -> None:
        torch.manual_seed(12)
        linear = torch.tensor([[0.6, 0.2], [-0.1, 0.8]], dtype=torch.float64)

        def transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return state + action @ linear.T

        source = torch.randn(30, 2, dtype=torch.float64)
        action = torch.randn(30, 2, dtype=torch.float64)
        train = _batch(source[:12], action[:12], transition, SplitRole.TRAIN, "train")
        validation = _batch(
            source[12:22], action[12:22], transition, SplitRole.VALIDATION, "validation"
        )
        test = _batch(source[22:], action[22:], transition, SplitRole.TEST, "test")

        def candidates() -> tuple[object, ...]:
            return (
                NoChangeBaseline(),
                MeanEffectBaseline(),
                ExactJVPBaseline(transition),
                AffineRidgeTransport(
                    classification=BaselineClass.FAIR_COMPARATOR,
                    name="complete_delta_fair_control",
                ),
                ObservedPrefixRelinearizedBaseline(transition),
            )

        first = select_train_validation_baseline(candidates(), train, validation)  # type: ignore[arg-type]
        second = select_train_validation_baseline(candidates(), train, validation)  # type: ignore[arg-type]
        self.assertEqual(first.baseline_star.metadata.name, "exact_jvp")
        self.assertEqual(first.record.sha256, second.record.sha256)
        self.assertEqual(first.record.canonical_json(), second.record.canonical_json())
        self.assertEqual(
            first.record.payload["candidate_classifications"][
                "observed_prefix_relinearized_oracle"
            ],
            BaselineClass.ORACLE_CEILING.value,
        )
        self.assertIn(
            "observed_prefix_relinearized_oracle",
            first.record.payload["excluded_from_residual_target"],
        )
        with self.assertRaises(ProtectedSplitAccessError):
            BaselineInputView.from_train_batch(validation)
        with self.assertRaises(TypeError):
            MeanEffectBaseline().fit(validation)  # type: ignore[arg-type]
        with self.assertRaises(ProtectedSplitAccessError):
            select_train_validation_baseline(candidates(), train, test)  # type: ignore[arg-type]
        with self.assertRaises(ProtectedSplitAccessError):
            first.baseline_star.predict_batch(test)
        prediction = first.baseline_star.predict_batch(test, protected_execution_authorized=True)
        torch.testing.assert_close(prediction, test.delta)
        with self.assertRaises(ValueError):
            BaselineSelectionRecord(first.record.canonical_json(), "not-the-record-hash")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection_record.json"
            first.record.save(path)
            restored_record = BaselineSelectionRecord.load(path)
            self.assertEqual(restored_record.sha256, first.record.sha256)
            self.assertEqual(restored_record.canonical_json(), first.record.canonical_json())

    def test_normalized_ratio_of_sums_is_explicit_and_fail_closed(self) -> None:
        source = torch.zeros((12, 2), dtype=torch.float64)
        action = torch.tensor(
            [[1.0, -1.0], [2.0, 1.0], [-1.0, 2.0], [0.5, -0.5]] * 3,
            dtype=torch.float64,
        )

        def transition(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            return state + control

        train = _batch(source[:6], action[:6], transition, SplitRole.TRAIN, "ratio-train")
        validation = _batch(
            source[6:], action[6:], transition, SplitRole.VALIDATION, "ratio-validation"
        )
        selection = select_train_validation_baseline(
            (NoChangeBaseline(), ExactJVPBaseline(transition)),
            train,
            validation,
            aggregation_rule="normalized_ratio_of_sums",
        )
        scores = {score.name: score.validation_score for score in selection.scores}
        self.assertAlmostEqual(scores["no_change"], 1.0, places=12)
        self.assertLess(scores["exact_jvp"], 1e-20)
        semantics = selection.record.payload["metric_denominator"]
        self.assertEqual(
            semantics["denominator_semantics"],
            "sum_squared_observed_delta_over_validation_rows",
        )
        self.assertGreater(float(semantics["denominator_value"]), 0.0)
        with self.assertRaises(BaselineEligibilityError):
            select_train_validation_baseline(
                (NoChangeBaseline(),), train, validation, endpoint_metric="arbitrary_metric"
            )
        with self.assertRaises(BaselineEligibilityError):
            select_train_validation_baseline(
                (NoChangeBaseline(),), train, validation, aggregation_rule="arbitrary_aggregate"
            )
        zero_validation = CausalEffectBatch(
            source=validation.source,
            treated=validation.source,
            intervention=validation.intervention,
            split=SplitRole.VALIDATION,
        )
        with self.assertRaises(BaselineEligibilityError):
            select_train_validation_baseline(
                (NoChangeBaseline(),),
                train,
                zero_validation,
                aggregation_rule="normalized_ratio_of_sums",
            )

    def test_observed_prefix_oracle_batch_interface_and_selector_exclusion(self) -> None:
        source = torch.tensor([[0.5, -1.0], [1.0, 0.25]], dtype=torch.float64)
        action = torch.tensor([[0.2, -0.4], [-0.3, 0.1]], dtype=torch.float64)

        def transition(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            return state + control + 0.2 * control.square()

        prefix = source + torch.tensor([[0.1, 0.2], [-0.2, 0.3]], dtype=torch.float64)
        batch = CausalEffectBatch(
            source=source,
            treated=transition(source, action),
            intervention=action,
            split=SplitRole.TEST,
            observed_prefix=prefix,
        )
        oracle = ObservedPrefixRelinearizedBaseline(transition)
        explicit = oracle.predict(source, action, observed_prefix=prefix)
        batched = oracle.predict_batch(batch, protected_execution_authorized=True)
        torch.testing.assert_close(explicit, batched, atol=1e-12, rtol=1e-12)
        missing_prefix = CausalEffectBatch(
            source=source,
            treated=transition(source, action),
            intervention=action,
            split=SplitRole.TEST,
        )
        with self.assertRaises(ProtectedSplitAccessError):
            oracle.predict_batch(missing_prefix, protected_execution_authorized=True)

        train = CausalEffectBatch(
            source=source,
            treated=transition(source, action),
            intervention=action,
            split=SplitRole.TRAIN,
            observed_prefix=prefix,
        )
        validation = CausalEffectBatch(
            source=source,
            treated=transition(source, action),
            intervention=action,
            split=SplitRole.VALIDATION,
            observed_prefix=prefix,
        )
        selected = select_train_validation_baseline(
            (ExactJVPBaseline(transition), oracle), train, validation
        )
        self.assertEqual(selected.baseline_star.metadata.name, "exact_jvp")
        self.assertIn(
            oracle.metadata.name,
            selected.record.payload["excluded_from_residual_target"],
        )
        self.assertEqual(oracle.metadata.classification, BaselineClass.ORACLE_CEILING)

    def test_selector_isolates_arbitrary_deployable_candidates_from_outcomes(self) -> None:
        class AdversarialCandidate:
            metadata = BaselineMetadata(
                "adversarial_input_audit", BaselineClass.DEPLOYABLE_RESIDUALIZER
            )

            def __init__(self) -> None:
                self.fit_forbidden_attributes: tuple[bool, ...] = ()
                self.predict_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None

            def fit(self, train_inputs: BaselineInputView) -> "AdversarialCandidate":
                self.fit_forbidden_attributes = tuple(
                    hasattr(train_inputs, name)
                    for name in ("treated", "delta", "observed_prefix", "identities", "split")
                )
                train_inputs.source.zero_()  # Must not mutate the selector's held train batch.
                return self

            def predict(self, source: torch.Tensor, intervention: torch.Tensor) -> torch.Tensor:
                self.predict_shapes = (tuple(source.shape), tuple(intervention.shape))
                source.zero_()  # Must not mutate the held validation target through aliasing.
                return torch.zeros_like(source)

            def artifact_payload(self) -> dict[str, object]:
                return {
                    "name": self.metadata.name,
                    "classification": self.metadata.classification.value,
                }

        source = torch.arange(24, dtype=torch.float64).reshape(12, 2)
        action = torch.ones((12, 2), dtype=torch.float64)

        def transition(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
            return state + control

        train = _batch(source[:6], action[:6], transition, SplitRole.TRAIN, "isolation-train")
        validation = _batch(
            source[6:], action[6:], transition, SplitRole.VALIDATION, "isolation-validation"
        )
        original_train_source = train.source.clone()
        original_validation_source = validation.source.clone()
        adversarial = AdversarialCandidate()
        selection = select_train_validation_baseline(
            (adversarial, ExactJVPBaseline(transition)),
            train,
            validation,  # type: ignore[arg-type]
        )
        self.assertEqual(adversarial.fit_forbidden_attributes, (False, False, False, False, False))
        self.assertEqual(
            adversarial.predict_shapes, (tuple(validation.source.shape), tuple(action[6:].shape))
        )
        torch.testing.assert_close(train.source, original_train_source)
        torch.testing.assert_close(validation.source, original_validation_source)
        self.assertEqual(selection.baseline_star.metadata.name, "exact_jvp")

    def test_residual_head_save_load_resume_and_provenance(self) -> None:
        torch.manual_seed(44)
        source = torch.randn(18, 3, dtype=torch.float64)
        action = torch.randn(18, 2, dtype=torch.float64)
        residual = torch.stack(
            [
                action[:, 0].square() * action[:, 1],
                action[:, 0] * action[:, 1].square(),
                action[:, 0].pow(3),
            ],
            dim=-1,
        )
        train = CausalEffectBatch(
            source=source,
            treated=source + residual,
            intervention=action,
            split=SplitRole.TRAIN,
        )
        head = TrainOnlyPolynomialResidualHead.fit(train, residual)
        initial_prediction = head(source, action)
        initial_provenance = head.provenance()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "residual_head.pt"
            head.save(checkpoint)
            loaded = TrainOnlyPolynomialResidualHead.load(checkpoint)
            torch.testing.assert_close(
                loaded(source, action), initial_prediction, atol=0.0, rtol=0.0
            )
            self.assertEqual(loaded.provenance(), initial_provenance)
            loaded.resume_fit(train, residual)
            torch.testing.assert_close(
                loaded(source, action), initial_prediction, atol=1e-12, rtol=1e-12
            )
            resumed_provenance = loaded.provenance()
            self.assertEqual(resumed_provenance["fit_count"], 2)
            self.assertEqual(resumed_provenance["state_sha256"], initial_provenance["state_sha256"])
            self.assertEqual(
                resumed_provenance["target_sha256"], initial_provenance["target_sha256"]
            )

    def test_router_candidates_have_full_width_values_and_independent_ablations(self) -> None:
        torch.manual_seed(25)
        context = torch.randn(5, 4, 6)
        intervention = torch.randn(5, 2)
        coordinates = torch.randn(4, 3)
        local = NoRoutingLocalResidualPredictor(6, 2, hidden_dim=11)
        router = ConditionalLowRankRouter(6, 2, 3, routing_rank=3, hidden_dim=10)
        matched = ParameterMatchedMLPResidualPredictor.matched_to(router, 6, 2)
        cross_attention = StandardCrossAttentionResidualPredictor(6, 2, hidden_dim=10, num_heads=2)
        self.assertEqual(local(context, intervention).shape, (5, 6))
        self.assertEqual(matched(context, intervention).shape, (5, 6))
        router_parameters = sum(parameter.numel() for parameter in router.parameters())
        matched_parameters = sum(parameter.numel() for parameter in matched.parameters())
        self.assertLessEqual(abs(router_parameters - matched_parameters), 6)

        routed = router(context, intervention, coordinates, return_routing=True)
        routing_ablated = router(
            context, intervention, coordinates, routing_ablation=True, return_routing=True
        )
        value_ablated = router(
            context, intervention, coordinates, value_ablation=True, return_routing=True
        )
        self.assertEqual(routed.prediction.shape, (5, 6))
        self.assertEqual(routed.routing_weights.shape, (5, 1, 4))
        self.assertEqual(routed.transported_values.shape, (5, 6))
        self.assertGreater(float(routed.transported_values.abs().sum().detach()), 0.0)
        self.assertEqual(float(value_ablated.transported_values.abs().sum().detach()), 0.0)
        expected_uniform = torch.full((5, 1, 4), 0.25)
        torch.testing.assert_close(routing_ablated.routing_weights, expected_uniform)
        cross = cross_attention(context, intervention, return_routing=True)
        cross_value_ablated = cross_attention(
            context, intervention, value_ablation=True, return_routing=True
        )
        self.assertEqual(cross.prediction.shape, (5, 6))
        self.assertEqual(cross.routing_weights.shape, (5, 1, 4))
        self.assertEqual(float(cross_value_ablated.transported_values.abs().sum().detach()), 0.0)
        model = CausalResidualJEPA(local)
        baseline = torch.zeros((5, 6))
        self.assertEqual(model.replay(context[:, 0], baseline, context, intervention).shape, (5, 6))


if __name__ == "__main__":
    unittest.main()
