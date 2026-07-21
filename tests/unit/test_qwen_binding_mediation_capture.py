from __future__ import annotations

import importlib.util
import json
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_capture import (
    aggregate_capture_metrics,
    assert_capture_contract,
    capture_content_digest,
    capture_eligibility_gates,
    estimate_capture_bytes,
    load_episode_progress,
    save_episode_progress,
    stack_episode_captures,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    BindingEpisode,
    TokenizedTreatment,
    binding_episodes_from_config,
)
from causal_workspace_jepa.hooks.names import transformer_site


class QwenBindingMediationCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/experiments/qwen_binding_mediation_v2.yaml")

    def _contract_fixture(
        self, *, count: int = 1
    ) -> tuple[
        dict[str, object],
        list[tuple[dict[str, np.ndarray], dict[str, object]]],
        dict[str, TokenizedTreatment],
    ]:
        config = copy.deepcopy(self.config)
        for split in config["splits"].values():
            split["count"] = count
        episodes = binding_episodes_from_config(config)
        captures = []
        treatments = {}
        for episode in episodes:
            treatment = TokenizedTreatment(
                recipient_ids=(0, 1, 2, 3, 4),
                donor_ids=(0, 3, 2, 1, 4),
                changed_positions=(1, 3),
                recipient_answer_id=1,
                donor_answer_id=2,
            )
            treatments[episode.episode_id] = treatment
            arrays = {
                "clean_candidate": np.zeros((2, 4), dtype=np.float32),
                "treated_candidate": np.zeros((2, 4), dtype=np.float32),
                "clean_final": np.zeros(4, dtype=np.float32),
                "treated_final": np.zeros(4, dtype=np.float32),
                "clean_value_logits": np.zeros(3, dtype=np.float32),
                "donor_value_logits": np.zeros(3, dtype=np.float32),
                "treated_value_logits": np.zeros(3, dtype=np.float32),
                "recipient_answer_id": np.asarray(1, dtype=np.int64),
                "donor_answer_id": np.asarray(2, dtype=np.int64),
                "clean_top_token": np.asarray(1, dtype=np.int64),
                "donor_top_token": np.asarray(2, dtype=np.int64),
                "treated_top_token": np.asarray(2, dtype=np.int64),
                "clean_score": np.asarray(0.0, dtype=np.float32),
                "treated_score": np.asarray(1.0, dtype=np.float32),
                "treatment_effect": np.asarray(1.0, dtype=np.float32),
                "treatment_source_replay_error": np.asarray(0.0, dtype=np.float32),
                "treatment_logit_replay_error": np.asarray(0.0, dtype=np.float32),
            }
            changed_positions = [1, 3]
            record = {
                "example_id": episode.episode_id,
                "episode": episode.to_dict(),
                "changed_positions": changed_positions,
                "recipient_answer_id": treatment.recipient_answer_id,
                "donor_answer_id": treatment.donor_answer_id,
                "intervention": {
                    "site": config["treatment"]["site"],
                    "operation": "patch",
                    "positions": changed_positions,
                    "feature_ids": None,
                    "magnitude": 1.0,
                    "donor_example_id": f"{episode.episode_id}:treatment",
                    "seed": config["seed"],
                },
            }
            captures.append((arrays, record))
        return config, captures, treatments

    def test_registered_episode_order_matches_token_audit(self) -> None:
        episodes = binding_episodes_from_config(self.config)
        self.assertEqual(len(episodes), 560)
        self.assertEqual(episodes[0].episode_id, "calibration-0000")
        self.assertEqual(episodes[-1].episode_id, "paraphrase-0095")
        test = [episode for episode in episodes if episode.split == "test"]
        paraphrase = [episode for episode in episodes if episode.split == "paraphrase"]
        self.assertEqual(
            [(e.keys, e.recipient_values, e.donor_values, e.query_index) for e in test],
            [(e.keys, e.recipient_values, e.donor_values, e.query_index) for e in paraphrase],
        )
        self.assertTrue(all(left.template != right.template for left, right in zip(test, paraphrase)))

    def test_capture_estimate_stays_under_registered_budget(self) -> None:
        estimate = estimate_capture_bytes(
            examples=560,
            candidates=56,
            hidden_size=1024,
            selected_logits=40,
        )
        self.assertLess(estimate, self.config["activation_budget_mb"] * 1024**2)
        self.assertGreater(estimate, 250 * 1024**2)
        self.assertGreater(estimate, 512 * 1024**2)

    def test_progress_round_trip_fails_closed_on_stale_identity(self) -> None:
        episode = binding_episodes_from_config(self.config)[0]
        arrays = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        record = {"episode": episode.to_dict(), "example_id": episode.episode_id}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.npz"
            save_episode_progress(path, "digest", episode, arrays, record)
            restored, restored_record = load_episode_progress(path, "digest", episode)
            np.testing.assert_array_equal(restored["x"], arrays["x"])
            self.assertEqual(restored["x"].dtype, np.float32)
            self.assertEqual(
                json.dumps(restored_record, sort_keys=True),
                json.dumps(record, sort_keys=True),
            )
            with self.assertRaises(RuntimeError):
                load_episode_progress(path, "different", episode)
            bad_record = {**record, "example_id": "wrong"}
            with self.assertRaisesRegex(RuntimeError, "example ID"):
                save_episode_progress(path, "digest", episode, arrays, bad_record)

    def test_stack_rejects_schema_mismatch(self) -> None:
        stacked = stack_episode_captures(
            [{"x": np.asarray(1)}, {"x": np.asarray(2)}]
        )
        np.testing.assert_array_equal(stacked["x"], [1, 2])
        with self.assertRaises(ValueError):
            stack_episode_captures([{"x": np.asarray(1)}, {"y": np.asarray(2)}])

    def test_progress_load_rejects_tampered_frozen_intervention(self) -> None:
        config, captures, treatments = self._contract_fixture()
        arrays, record = captures[0]
        episode = binding_episodes_from_config(config)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.npz"
            save_episode_progress(path, "digest", episode, arrays, record)
            load_episode_progress(
                path,
                "digest",
                episode,
                expected_treatment_site=config["treatment"]["site"],
                expected_seed=config["seed"],
                expected_treatment=treatments[episode.episode_id],
            )
            tampered = copy.deepcopy(record)
            tampered["intervention"]["site"] = "blocks.1.resid_pre"
            save_episode_progress(path, "digest", episode, arrays, tampered)
            with self.assertRaisesRegex(RuntimeError, "frozen treatment"):
                load_episode_progress(
                    path,
                    "digest",
                    episode,
                    expected_treatment_site=config["treatment"]["site"],
                    expected_seed=config["seed"],
                    expected_treatment=treatments[episode.episode_id],
                )

    def test_capture_metrics_and_gates_use_every_protected_split(self) -> None:
        config = copy.deepcopy(self.config)
        config["splits"]["calibration"]["count"] = 0
        for split in ("train", "validation", "test", "paraphrase"):
            config["splits"][split]["count"] = 10
        captures = []
        for split in ("train", "validation", "test", "paraphrase"):
            for index in range(10):
                correct = index < 9
                arrays = {
                    "clean_top_token": np.asarray(1 if correct else 9),
                    "recipient_answer_id": np.asarray(1),
                    "donor_top_token": np.asarray(2),
                    "donor_answer_id": np.asarray(2),
                    "treated_top_token": np.asarray(2 if index < 8 else 9),
                    "treatment_effect": np.asarray(1.0),
                    "treatment_source_replay_error": np.asarray(0.0),
                    "treatment_logit_replay_error": np.asarray(0.0),
                }
                episode = {
                    "split": split,
                    "query_index": 0,
                    "keys": ["key", "b", "c", "d"],
                    "recipient_values": ["recipient", "x", "y", "z"],
                    "donor_values": ["donor", "x", "y", "z"],
                }
                record = {"example_id": f"{split}-{index:04d}", "episode": episode}
                captures.append((arrays, record))
        metrics = aggregate_capture_metrics(captures, config)
        gates = capture_eligibility_gates(metrics, config)
        self.assertTrue(gates["treatment_replay"])
        self.assertTrue(gates["all_protected_splits_competent"])
        captures[-3][0]["treated_top_token"] = np.asarray(9)
        metrics = aggregate_capture_metrics(captures, config)
        gates = capture_eligibility_gates(metrics, config)
        self.assertFalse(gates["paraphrase_competent"])
        self.assertFalse(gates["all_protected_splits_competent"])

    def test_replay_nan_fails_closed_at_every_position(self) -> None:
        base = []
        for index in range(3):
            arrays = {
                "clean_top_token": np.asarray(1),
                "recipient_answer_id": np.asarray(1),
                "donor_top_token": np.asarray(2),
                "donor_answer_id": np.asarray(2),
                "treated_top_token": np.asarray(2),
                "treatment_effect": np.asarray(1.0),
                "treatment_source_replay_error": np.asarray(0.0),
                "treatment_logit_replay_error": np.asarray(0.0),
            }
            record = {
                "example_id": f"test-{index}",
                "episode": {
                    "split": "test",
                    "query_index": 0,
                    "keys": ["a", "b", "c", "d"],
                    "recipient_values": ["w", "x", "y", "z"],
                    "donor_values": ["x", "w", "y", "z"],
                },
            }
            base.append((arrays, record))
        for position in range(3):
            captures = copy.deepcopy(base)
            captures[position][0]["treatment_logit_replay_error"] = np.asarray(np.nan)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                aggregate_capture_metrics(captures, self.config)

    def test_content_digest_detects_array_or_record_change(self) -> None:
        arrays = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
        records = [{"example_id": "a"}, {"example_id": "b"}]
        original = capture_content_digest(arrays, records)
        changed = {"x": arrays["x"].copy()}
        changed["x"][1, 2] += 1
        self.assertNotEqual(original, capture_content_digest(changed, records))
        self.assertNotEqual(
            original,
            capture_content_digest(arrays, [{"example_id": "a"}, {"example_id": "c"}]),
        )

    def test_contract_rejects_float16_causal_states(self) -> None:
        config, captures, treatments = self._contract_fixture()
        captures[0][0]["clean_candidate"] = captures[0][0]["clean_candidate"].astype(
            np.float16
        )
        with self.assertRaisesRegex(RuntimeError, "float32"):
            assert_capture_contract(
                captures,
                config,
                expected_treatments=treatments,
                candidates=2,
                hidden_size=4,
                selected_logits=3,
            )

    def test_contract_rejects_noncanonical_order_and_episode_record(self) -> None:
        config, captures, treatments = self._contract_fixture(count=2)
        with self.assertRaisesRegex(RuntimeError, "canonical episode roster"):
            assert_capture_contract(
                list(reversed(captures)),
                config,
                expected_treatments=treatments,
                candidates=2,
                hidden_size=4,
                selected_logits=3,
            )
        captures[0][1]["episode"] = captures[1][1]["episode"]
        with self.assertRaisesRegex(RuntimeError, "canonical episode"):
            assert_capture_contract(
                captures,
                config,
                expected_treatments=treatments,
                candidates=2,
                hidden_size=4,
                selected_logits=3,
            )

    def test_contract_checks_every_array_shape_and_dtype(self) -> None:
        config, captures, treatments = self._contract_fixture()
        baseline = captures[0][0]
        for name, value in baseline.items():
            malformed = copy.deepcopy(captures)
            malformed[0][0][name] = np.asarray(value).reshape(-1)
            if np.asarray(value).shape == ():
                malformed[0][0][name] = np.asarray([value])
            else:
                malformed[0][0][name] = np.asarray(value).reshape(-1)[:-1]
            with self.assertRaisesRegex(RuntimeError, "shape"):
                assert_capture_contract(
                    malformed,
                    config,
                    expected_treatments=treatments,
                    candidates=2,
                    hidden_size=4,
                    selected_logits=3,
                )
            wrong_dtype = copy.deepcopy(captures)
            replacement = np.float64 if np.issubdtype(value.dtype, np.floating) else np.int32
            wrong_dtype[0][0][name] = value.astype(replacement)
            with self.assertRaisesRegex(RuntimeError, "float32|int64"):
                assert_capture_contract(
                    wrong_dtype,
                    config,
                    expected_treatments=treatments,
                    candidates=2,
                    hidden_size=4,
                    selected_logits=3,
                )

    def test_contract_rejects_coherently_tampered_treatment_metadata(self) -> None:
        config, captures, treatments = self._contract_fixture()
        arrays, record = captures[0]
        record["changed_positions"] = [0, 4]
        record["intervention"]["positions"] = [0, 4]
        record["recipient_answer_id"] = 999
        arrays["recipient_answer_id"] = np.asarray(999, dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "frozen treatment"):
            assert_capture_contract(
                captures,
                config,
                expected_treatments=treatments,
                candidates=2,
                hidden_size=4,
                selected_logits=3,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
        "optional torch/transformers dependencies are unavailable",
    )
    def test_tiny_qwen_capture_replays_exact_donor(self) -> None:
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from causal_workspace_jepa.adapters.qwen_hf_adapter import (
            QwenAdapterConfig,
            QwenHFAdapter,
        )
        from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_capture import (
            capture_binding_episode,
        )

        episode = BindingEpisode(
            episode_id="tiny-0000",
            split="calibration",
            keys=("a", "b", "c", "d"),
            recipient_values=("w", "x", "y", "z"),
            donor_values=("x", "w", "y", "z"),
            query_index=0,
            swapped_indices=(0, 1),
        )
        tokenizer = _ExactTreatmentTokenizer(episode)
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=16,
            )
        )
        adapter = QwenHFAdapter(
            QwenAdapterConfig(
                model_name="tiny-random-qwen3",
                device="cpu",
                dtype="float32",
                max_length=8,
            ),
            model=model,
            tokenizer=tokenizer,
        )
        candidate_sites = tuple(
            transformer_site(layer, kind)
            for layer in range(2)
            for kind in ("attn_out", "mlp_out")
        )
        treatment = TokenizedTreatment(
            recipient_ids=tuple(tokenizer.recipient_ids),
            donor_ids=tuple(tokenizer.donor_ids),
            changed_positions=(1, 3),
            recipient_answer_id=10,
            donor_answer_id=11,
        )
        arrays, record = capture_binding_episode(
            adapter,
            episode,
            treatment,
            candidate_sites=candidate_sites,
            treatment_site=transformer_site(0, "resid_pre"),
            final_site=transformer_site(1, "resid_post"),
            value_ids=[10, 11, 12, 13],
            seed=1,
        )
        self.assertEqual(arrays["clean_candidate"].shape, (4, 32))
        self.assertEqual(arrays["clean_candidate"].dtype, np.float32)
        self.assertEqual(arrays["treated_candidate"].dtype, np.float32)
        self.assertEqual(float(arrays["treatment_source_replay_error"]), 0.0)
        self.assertEqual(float(arrays["treatment_logit_replay_error"]), 0.0)
        self.assertEqual(record["changed_positions"], [1, 3])


class _ExactTreatmentTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, episode: BindingEpisode) -> None:
        self.recipient_prompt = episode.recipient_prompt()
        self.donor_prompt = episode.donor_prompt()
        self.recipient_ids = [2, 10, 3, 11, 4]
        self.donor_ids = [2, 11, 3, 10, 4]

    def __call__(self, prompts: list[str], **_kwargs: object) -> dict[str, object]:
        import torch

        rows = [
            self.recipient_ids if prompt == self.recipient_prompt else self.donor_ids
            for prompt in prompts
        ]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), len(rows[0])), dtype=torch.long),
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            if text == self.recipient_prompt:
                return self.recipient_ids
            if text == self.donor_prompt:
                return self.donor_ids
        answers = {" w": [10], " x": [11], " y": [12], " z": [13]}
        return answers[text]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"tok_{value}" for value in ids]


if __name__ == "__main__":
    unittest.main()
