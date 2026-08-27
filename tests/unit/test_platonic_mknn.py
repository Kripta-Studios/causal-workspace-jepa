from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal_workspace_jepa.data.splits import deterministic_named_split_ids
from causal_workspace_jepa.experiments.world_model.platonic_mknn import (
    FORBIDDEN_SPLIT_NAMES,
    adjudicate,
    assert_protocol,
    frozen_linear_map,
    load_json_config,
    reject_forbidden_seed,
    run_seed,
    sha256_array,
    write_artifacts,
)
from causal_workspace_jepa.interpretability.mutual_knn import chance_reference, mutual_knn
from causal_workspace_jepa.models.tiny_jepa import TinyActionConditionedJEPA

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/wm_platonic_mknn_v1.json"
PROTOCOL = ROOT / "docs/WM_PLATONIC_MKNN_001_PROTOCOL.md"
MODULE = ROOT / "src/causal_workspace_jepa/experiments/world_model/platonic_mknn.py"


def _tiny_config() -> dict:
    config = load_json_config(CONFIG)
    config["environment"]["trajectories"] = 24
    config["environment"]["steps"] = 8
    config["splits"]["train"] = 12
    config["splits"]["development"] = 6
    config["splits"]["confirmation"] = 6
    config["mknn"]["n_eval"] = 16
    config["mknn"]["k"] = 3
    config["mknn"]["chance"] = chance_reference(n_eval=16, k=3)
    return config


def test_protocol_and_config_are_frozen_before_outcomes() -> None:
    config = load_json_config(CONFIG)
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert_protocol(config)
    assert config["experiment_id"] == "WM-PLATONIC-MKNN-001"
    assert config["confirmation_seeds"] == [131, 137, 139]
    assert "`131, 137, 139`" in protocol
    assert config["mknn"]["k"] == 5
    assert config["mknn"]["n_eval"] == 128
    assert config["gates"]["predictor_mknn_must_beat_chance_multiplier"] == 2.0
    assert config["observation_maps"]["trainable"] is False
    assert config["claim_boundary"]["not_workspace"] is True
    assert "test" not in config["splits"]
    assert list(config["splits"]["forbidden_split_names"]) == ["test", "paraphrase"]
    assert "WM-PLATONIC-STITCH" not in MODULE.read_text(encoding="utf-8")


def test_forbidden_seeds_and_protected_split_names() -> None:
    config = load_json_config(CONFIG)
    for seed in (1009, 2027, 4093, 701, 11, 811):
        with pytest.raises(ValueError):
            reject_forbidden_seed(seed, config["forbidden_seeds"])
    ids = deterministic_named_split_ids(80, 227, train=48, development=16, confirmation=16)
    assert set(ids) == {"train", "development", "confirmation"}
    assert not any(name in FORBIDDEN_SPLIT_NAMES for name in ids)
    again = deterministic_named_split_ids(80, 227, train=48, development=16, confirmation=16)
    for key in ids:
        np.testing.assert_array_equal(ids[key], again[key])


def test_mutual_knn_on_known_synthetic_geometry() -> None:
    rng = np.random.default_rng(0)
    left = rng.normal(size=(32, 4))
    unrelated = rng.normal(size=(32, 4))
    assert mutual_knn(left, left.copy(), k=3) == 1.0
    assert mutual_knn(left, unrelated, k=3) < 0.4
    assert chance_reference(n_eval=128, k=5) == 5 / 127


def test_observation_maps_are_frozen_and_trajectories_are_matched() -> None:
    first = frozen_linear_map(4, 16, 801)
    second = frozen_linear_map(4, 16, 801)
    np.testing.assert_allclose(first, second)
    row = run_seed(_tiny_config(), seed=3)
    assert row["map_a_sha256"] == sha256_array(frozen_linear_map(4, 16, 801))
    assert row["protected_splits_executed"] == []
    assert row["development_metrics_computed"] is False
    assert row["shuffled_latent_mse_b"] > row["conditioned_latent_mse_b"]
    assert row["predictor_mknn_ab"] != row["predictor_mknn_a_shuffle"]


def test_frozen_encoder_is_not_resampled() -> None:
    encoder = np.eye(16, dtype=np.float32)
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(8, 6, 16)).astype(np.float32)
    actions = rng.normal(size=(8, 5, 2)).astype(np.float32)
    model = TinyActionConditionedJEPA.fit(
        obs,
        actions,
        latent_dim=16,
        seed=3,
        frozen_encoder=encoder,
    )
    np.testing.assert_array_equal(model.encoder, encoder)


def test_adjudicate_does_not_learn_thresholds_from_outcomes() -> None:
    config = load_json_config(CONFIG)
    gates_before = json.dumps(config["gates"], sort_keys=True)
    fake = [
        {
            "seed": 131,
            "predictor_mknn_ab": 0.01,
            "predictor_mknn_a_shuffle": 0.02,
            "predictor_mknn_a_random": 0.02,
            "encoder_mknn_ab": 0.01,
        }
    ]
    metrics = adjudicate(fake, config)
    assert json.dumps(config["gates"], sort_keys=True) == gates_before
    assert metrics["status"] == "NEGATIVE_RESULT"
    assert metrics["seed_rows"][0]["chance_floor"] == 2.0 * config["mknn"]["chance"]
    assert metrics["evidence_level"] == "Availability"
    assert metrics["stitching_executed"] is False


def test_write_artifacts_collects_provenance_before_metrics_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from causal_workspace_jepa.common.provenance import Provenance

    existed: dict[str, bool] = {}

    def fake_collect(command: str, resource_profile: str, seed: int | None = None) -> Provenance:
        existed["metrics"] = (tmp_path / "wm.json").exists()
        return Provenance("t", "abc1234", False, "3", "win", command, seed, resource_profile)

    monkeypatch.setattr(
        "causal_workspace_jepa.experiments.world_model.platonic_mknn.collect_provenance",
        fake_collect,
    )
    config = load_json_config(CONFIG)
    config["output_metrics"] = str(tmp_path / "wm.json")
    write_artifacts({"status": "X"}, config, command="cmd")
    assert existed["metrics"] is False
    assert (tmp_path / "wm.json").exists()
    sidecar = json.loads((tmp_path / "wm.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["git_dirty"] is False
