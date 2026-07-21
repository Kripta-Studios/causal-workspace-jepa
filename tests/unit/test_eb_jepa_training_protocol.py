import pytest

from causal_workspace_jepa.experiments.world_model.eb_jepa_training_protocol import (
    EXPECTED_SOURCE_CONFIG,
    required_checkpoint_names,
    validate_source_training_config,
)


def _nested_config() -> dict:
    root: dict = {}
    for path, value in EXPECTED_SOURCE_CONFIG.items():
        cursor = root
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return root


def test_training_protocol_accepts_exact_source_config() -> None:
    assert validate_source_training_config(_nested_config()) == EXPECTED_SOURCE_CONFIG


def test_training_protocol_rejects_batch_drift() -> None:
    config = _nested_config()
    config["data"]["batch_size"] = 128
    with pytest.raises(ValueError, match="data.batch_size"):
        validate_source_training_config(config)


def test_training_protocol_freezes_all_and_last_checkpoints() -> None:
    names = required_checkpoint_names(12)
    assert len(names) == 13
    assert names[:2] == ["e-0.pth.tar", "e-1.pth.tar"]
    assert names[-4:] == ["e-9.pth.tar", "e-10.pth.tar", "e-11.pth.tar", "latest.pth.tar"]
