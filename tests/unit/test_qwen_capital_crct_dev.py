from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("h5py")
import h5py  # noqa: E402
from causal_workspace_jepa.experiments.llm.qwen_capital_crct_dev import (  # noqa: E402
    analyze_arrays,
    analyze_shard,
    nmse,
)


def test_nmse_zero_for_exact_prediction() -> None:
    target = np.array([[1.0, -2.0]])
    assert nmse(target, target.copy()) == 0.0


def test_capital_dev_keeps_direct_and_residual_families_separate() -> None:
    rng = np.random.default_rng(11)
    n = 42
    d = 12
    endpoint = 1030
    x = rng.normal(size=(n, d))
    weight = rng.normal(size=(d, endpoint)) * 0.05
    target = x @ weight + 0.01 * rng.normal(size=(n, endpoint))
    jvp = target * 0.7
    quadratic = target * 0.9
    split = np.array([0] * 30 + [1] * 6 + [2] * 6, dtype=np.int64)

    payload = analyze_arrays(
        source_delta=x,
        target_effect=target,
        exact_jvp=jvp,
        quadratic_taylor=quadratic,
        split_id=split,
        lambdas=[0.01, 0.1, 1.0],
    )
    assert payload["status"] == "DEVELOPMENT_ANALYSIS_COMPLETE"
    assert payload["split_counts"] == {"train": 30, "validation": 6, "test": 6}
    families = payload["equal_family_ridge"]
    assert "direct_delta" in families
    assert "quadratic_residual" in families
    assert payload["scientific_boundary"]["fresh_confirmation_claim_permitted"] is False


def test_capital_dev_reads_repository_hdf5_arrays_group(tmp_path) -> None:
    rng = np.random.default_rng(17)
    rows = 12
    source_dim = 8
    endpoint_dim = 1030
    source = rng.normal(size=(rows, source_dim)).astype(np.float32)
    weight = rng.normal(size=(source_dim, endpoint_dim)).astype(np.float32) * 0.01
    target = (source @ weight).astype(np.float32)
    jvp = (target * 0.75).astype(np.float32)
    quadratic = (target * 0.95).astype(np.float32)
    split = np.array([0] * 6 + [1] * 3 + [2] * 3, dtype=np.int64)

    shard = tmp_path / "shard-00000-of-00001.h5"
    with h5py.File(shard, "w") as handle:
        group = handle.create_group("arrays")
        group.create_dataset("source_delta", data=source)
        group.create_dataset("target_effect", data=target)
        group.create_dataset("exact_jvp", data=jvp)
        group.create_dataset("quadratic_taylor", data=quadratic)
        group.create_dataset("split_id", data=split)
        handle.create_dataset(
            "records_json",
            data=np.asarray(["{}"] * rows, dtype=h5py.string_dtype("utf-8")),
        )

    expected_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()
    payload = analyze_shard(shard, expected_sha256=expected_sha256)

    assert payload["status"] == "DEVELOPMENT_ANALYSIS_COMPLETE"
    assert payload["row_count"] == rows
    assert payload["source_shard"]["hash_matches"] is True
    assert payload["split_counts"] == {"train": 6, "validation": 3, "test": 3}
