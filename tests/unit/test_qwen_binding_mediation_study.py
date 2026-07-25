from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_evaluator import (
    DirectMediationOutcome,
    freeze_ranking,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    MediationEstimate,
    binding_episodes_from_config,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_study import (
    CaptureBundle,
    GitState,
    QwenProtectedOutcomeExecutor,
    aggregate_protected_outcomes,
    assert_budget,
    assert_protected_git_state,
    audited_unqueried_swap,
    build_frozen_train_plan,
    capture_config_digest,
    estimate_study_work,
    load_derivative_unit,
    load_progress,
    load_direct_outcome_unit,
    save_derivative_unit,
    sha256_file,
    sha256_json,
    validate_frozen_plan,
    write_frozen_plan,
    write_progress,
)


CONFIG = Path("configs/experiments/qwen_binding_mediation_v2.yaml")


class QwenBindingMediationStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)
        self.capture = CaptureBundle(
            arrays={},
            records=(),
            metrics={},
            manifest={},
            identity={
                "config_file_sha256": sha256_file(CONFIG),
                "config_digest": capture_config_digest(self.config),
                "capture_metrics_sha256": "1" * 64,
                "capture_manifest_sha256": "2" * 64,
                "capture_digest": "3" * 64,
                "capture_content_sha256": "4" * 64,
            },
        )

    def _plan(self) -> dict[str, object]:
        methods = tuple(str(value) for value in self.config["ranking"]["methods"])
        scores = np.arange(56, 0, -1, dtype=np.float64)
        rankings = {method: freeze_ranking(method, scores) for method in methods}
        return build_frozen_train_plan(
            CONFIG,
            self.capture,
            rankings,
            {1: MediationEstimate(10.0, 0.8, 0.8, True)},
            source_git_commit="a" * 40,
            calibration_sha256="b" * 64,
            answer_row_permutation_diagnostic={
                "endpoint": "maximum_exact_local_atp_node_score",
                "scope": "train_only_episode_local_alignment_not_population_mean_gradient",
                "observed": 1.0,
                "null_values": [0.0] * 256,
                "upper_tail_monte_carlo_p": 1.0 / 257.0,
            },
        )

    def test_work_estimate_counts_every_registered_direct_set(self) -> None:
        estimate = estimate_study_work(
            self.config,
            measured_derivative_seconds_per_episode=2.0,
            measured_direct_seconds_per_forward=0.1,
        )
        self.assertEqual(estimate.derivative_backward_calls_per_episode, 113)
        self.assertEqual(estimate.train_backward_calls, 256 * 113)
        self.assertEqual(estimate.protected_direct_sets_per_episode, 138)
        self.assertEqual(estimate.protected_direct_forward_calls, 192 * (138 * 5 + 2))
        self.assertGreater(estimate.projected_total_gpu_hours or 0.0, 0.0)

    def test_frozen_plan_is_train_only_self_hashed_and_deterministic(self) -> None:
        first = self._plan()
        second = self._plan()
        self.assertEqual(first, second)
        self.assertEqual(first["population_prefix"]["k"], 1)  # type: ignore[index]
        self.assertFalse(first["train_contract"]["protected_splits_opened"])  # type: ignore[index]
        self.assertEqual(len(first["matched_random_sets"]), 128)  # type: ignore[arg-type]
        self.assertFalse(
            first["direct_control_contracts"]["norm_matched_resample"]["covariance_matched"]  # type: ignore[index]
        )
        validate_frozen_plan(first, config_path=CONFIG, capture=self.capture)

    def test_plan_tampering_and_stale_capture_fail_closed(self) -> None:
        plan = self._plan()
        plan["population_prefix"]["k"] = 2  # type: ignore[index]
        with self.assertRaisesRegex(RuntimeError, "self-hash"):
            validate_frozen_plan(plan)
        valid = self._plan()
        stale = replace(
            self.capture,
            identity={**self.capture.identity, "capture_content_sha256": "9" * 64},
        )
        with self.assertRaisesRegex(RuntimeError, "different capture"):
            validate_frozen_plan(valid, capture=stale)
        rehashed = self._plan()
        rehashed["matched_random_sets"] = rehashed["matched_random_sets"][:-1]  # type: ignore[index]
        rehashed["matched_random_set_count"] = 127
        unsigned = dict(rehashed)
        unsigned.pop("plan_sha256")
        rehashed["plan_sha256"] = sha256_json(unsigned)
        with self.assertRaisesRegex(RuntimeError, "differs from configuration"):
            validate_frozen_plan(rehashed, config_path=CONFIG)

    def test_committed_plan_gate_rejects_dirty_untracked_and_changed_bytes(self) -> None:
        plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_frozen_plan(path, plan)
            committed = path.read_bytes()
            state = GitState("c" * 40, False, committed)
            self.assertEqual(assert_protected_git_state(path, state), state)
            with self.assertRaisesRegex(RuntimeError, "clean"):
                assert_protected_git_state(path, replace(state, dirty=True))
            with self.assertRaisesRegex(RuntimeError, "git-tracked"):
                assert_protected_git_state(path, replace(state, tracked_plan_bytes=None))
            with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
                assert_protected_git_state(path, replace(state, source_commit_is_ancestor=False))
            with self.assertRaisesRegex(RuntimeError, "code changed"):
                assert_protected_git_state(
                    path,
                    replace(
                        state,
                        changed_contract_paths=(
                            "src/causal_workspace_jepa/experiments/llm/"
                            "qwen_binding_mediation_study.py",
                        ),
                    ),
                )
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differ"):
                assert_protected_git_state(path, state)

    def test_progress_is_atomic_and_bound_to_exact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            fingerprint = {"phase": "protected_eval", "plan_sha256": "a" * 64}
            write_progress(path, fingerprint, {"done": ["test:0"]})
            self.assertEqual(load_progress(path, fingerprint), {"done": ["test:0"]})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            with self.assertRaisesRegex(RuntimeError, "stale"):
                load_progress(path, {**fingerprint, "plan_sha256": "b" * 64})

    def test_derivative_units_are_lossless_checksum_bound_and_resumable(self) -> None:
        rng = np.random.default_rng(7)
        payload = {
            "local_gradients": rng.normal(size=(56, 8)).astype(np.float32),
            "first_order_effects": rng.normal(size=56).astype(np.float32),
            "directional_hvp_terms": rng.normal(size=56).astype(np.float32),
            "graddrop_effects": rng.normal(size=(56, 56)).astype(np.float32),
            "clean_candidate": rng.normal(size=(56, 8)).astype(np.float32),
            "clean_score": np.asarray(0.25, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train-0000.npz"
            fingerprint = {"phase": "train_plan", "capture": "a" * 64}
            digest = save_derivative_unit(path, fingerprint, "train-0000", payload)
            loaded = load_derivative_unit(path, fingerprint, "train-0000", digest)
            for name in payload:
                np.testing.assert_array_equal(loaded[name], payload[name])
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                load_derivative_unit(path, fingerprint, "train-0000", "0" * 64)

    def test_budget_requires_measured_projection_and_explicit_bound(self) -> None:
        missing = estimate_study_work(self.config)
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            assert_budget(missing, max_gpu_hours=10.0)
        measured = estimate_study_work(
            self.config,
            measured_derivative_seconds_per_episode=10.0,
            measured_direct_seconds_per_forward=1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            assert_budget(measured, max_gpu_hours=0.001)

    def test_fake_protected_executor_materializes_every_unit_and_resumes_by_checksum(self) -> None:
        class FakeExecutor(QwenProtectedOutcomeExecutor):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.calls = 0

            def _execute_condition(self, **_kwargs: object) -> DirectMediationOutcome:
                self.calls += 1
                return QwenBindingMediationStudyTests._make_outcome(0.8, 0.8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            text = CONFIG.read_text(encoding="utf-8")
            text = text.replace("count: 16", "count: 2", 1)
            text = text.replace("count: 256", "count: 2", 1)
            text = text.replace("count: 96", "count: 2")
            text = text.replace("random_sets: 128", "random_sets: 2")
            config_path.write_text(text, encoding="utf-8")
            config = load_config(config_path)
            episodes = binding_episodes_from_config(config)
            rows = len(episodes)
            arrays = {
                "clean_candidate": np.zeros((rows, 56, 2), dtype=np.float32),
                "treated_candidate": np.ones((rows, 56, 2), dtype=np.float32),
                "donor_answer_id": np.ones(rows, dtype=np.int64),
            }
            records = tuple(
                {
                    "example_id": episode.episode_id,
                    "changed_positions": [0, 1],
                    "recipient_answer_id": 0,
                    "donor_answer_id": 1,
                }
                for episode in episodes
            )
            capture = CaptureBundle(
                arrays=arrays,
                records=records,
                metrics={},
                manifest={},
                identity={
                    "config_file_sha256": sha256_file(config_path),
                    "config_digest": capture_config_digest(config),
                    "capture_metrics_sha256": "1" * 64,
                    "capture_manifest_sha256": "2" * 64,
                    "capture_digest": "3" * 64,
                    "capture_content_sha256": "4" * 64,
                },
            )
            rankings = {
                method: freeze_ranking(method, np.arange(56, 0, -1, dtype=np.float64))
                for method in config["ranking"]["methods"]
            }
            plan = build_frozen_train_plan(
                config_path,
                capture,
                rankings,
                {1: MediationEstimate(2.0, 0.8, 0.8, True)},
                source_git_commit="a" * 40,
                calibration_sha256="b" * 64,
                answer_row_permutation_diagnostic={
                    "endpoint": "maximum_exact_local_atp_node_score",
                    "scope": "train_only_episode_local_alignment_not_population_mean_gradient",
                    "observed": 1.0,
                    "null_values": [0.0] * 256,
                    "upper_tail_monte_carlo_p": 1.0 / 257.0,
                },
            )
            units = root / "units"
            executor = FakeExecutor(config_path, capture=capture, unit_root=units)
            checkpoints: list[dict[str, object]] = []
            result = executor.execute_split(
                split="test",
                plan=plan,
                progress={},
                progress_callback=lambda value: checkpoints.append(dict(value)),
            )
            self.assertEqual(result["unit_count"], 24)
            self.assertEqual(executor.calls, 24)
            self.assertEqual(len(result["outcomes"]["matched_random"]), 2)
            self.assertEqual(set(result["outcomes"]["controls"]), {
                "donor_shuffle",
                "norm_matched_resample",
                "irrelevant_position",
                "unqueried_value_swap",
            })
            resumed = FakeExecutor(config_path, capture=capture, unit_root=units)
            resumed.execute_split(
                split="test",
                plan=plan,
                progress=checkpoints[-1],
                progress_callback=lambda _value: None,
            )
            self.assertEqual(resumed.calls, 0)
            unit_path = next(units.rglob("*.json"))
            payload = json.loads(unit_path.read_text(encoding="utf-8"))
            payload["outcome"]["clean_score"] = 99.0
            unit_path.write_text(json.dumps(payload), encoding="utf-8")
            identity = payload["identity"]
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                load_direct_outcome_unit(unit_path, identity)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                resumed.execute_split(
                    split="test",
                    plan=plan,
                    progress={"conditions_complete": ["population_atp"]},
                    progress_callback=lambda _value: None,
                )

    def test_unqueried_control_audits_disjoint_two_token_transposition(self) -> None:
        class TokenizerAdapter:
            def __init__(self) -> None:
                self.vocabulary: dict[str, int] = {}

            def tokenize(self, prompts: list[str]) -> object:
                tokens = prompts[0].split()
                ids = [self.vocabulary.setdefault(token, len(self.vocabulary) + 1) for token in tokens]
                return SimpleNamespace(input_ids=np.asarray([ids], dtype=np.int64))

        episode = next(
            episode
            for episode in binding_episodes_from_config(self.config)
            if episode.split == "calibration"
        )
        prompt, changed = audited_unqueried_swap(TokenizerAdapter(), episode)
        self.assertEqual(len(changed), 2)
        self.assertNotEqual(prompt, episode.recipient_prompt())
        self.assertIn(episode.keys[episode.query_index], prompt)

    def test_protected_aggregation_uses_direct_controls_and_decides_both_hypotheses(self) -> None:
        plan = self._plan()
        config = json.loads(json.dumps(self.config))
        config["splits"]["test"]["count"] = 4
        config["splits"]["paraphrase"]["count"] = 4
        primary = [self._make_outcome(0.8, 0.8) for _ in range(4)]
        comparator = [self._make_outcome(0.2, 0.2) for _ in range(4)]
        control = [self._make_outcome(0.1, 0.1) for _ in range(4)]
        split_payload = {
            "population_atp": primary,
            "comparators": {name: comparator for name in (
                "exact_local_atp",
                "directional_hvp",
                "atp_star",
                "leave_value_out_probe",
                "delta_norm",
            )},
            "matched_random": [comparator for _ in range(128)],
            "controls": {name: control for name in (
                "donor_shuffle",
                "norm_matched_resample",
                "irrelevant_position",
                "unqueried_value_swap",
            )},
            "donor_answer_ids": [1, 1, 1, 1],
        }
        payload = {
            "task_eligibility": {
                "train": True,
                "validation": True,
                "test": True,
                "paraphrase": True,
            },
            "test": split_payload,
            "paraphrase": split_payload,
        }
        result = aggregate_protected_outcomes(config, plan, payload)
        self.assertTrue(result["decisions"]["H-LLM-15"]["passed"])
        self.assertTrue(result["decisions"]["H-LLM-16"]["passed"])
        one_unbeaten = dict(split_payload)
        one_unbeaten["comparators"] = dict(split_payload["comparators"])
        one_unbeaten["comparators"]["directional_hvp"] = primary
        conjunction_payload = {**payload, "test": one_unbeaten}
        conjunction = aggregate_protected_outcomes(config, plan, conjunction_payload)
        self.assertFalse(conjunction["decisions"]["H-LLM-15"]["passed"])
        self.assertEqual(conjunction["decisions"]["H-LLM-15"]["evidence_level"], "None")
        incomplete = json.loads(json.dumps(payload, default=lambda value: value.__dict__))
        del incomplete["test"]["controls"]["irrelevant_position"]
        with self.assertRaisesRegex(RuntimeError, "controls are incomplete"):
            aggregate_protected_outcomes(config, plan, incomplete)
        with self.assertRaisesRegex(RuntimeError, "population outcome count"):
            aggregate_protected_outcomes(self.config, plan, payload)

    @staticmethod
    def _make_outcome(q: float, n: float) -> DirectMediationOutcome:
        return DirectMediationOutcome(
            clean_score=0.0,
            treated_score=1.0,
            sufficient_score=q,
            restored_score=1.0 - n,
            donor_score=1.0,
            clean_top_token=0,
            treated_top_token=1,
            sufficient_top_token=1 if q >= 0.5 else 0,
            restored_top_token=0 if n >= 0.5 else 1,
            donor_top_token=1,
        )


if __name__ == "__main__":
    unittest.main()
