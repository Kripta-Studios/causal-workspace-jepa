import numpy as np

from causal_workspace_jepa.experiments.llm.qwen_capital_crct_dev import analyze_arrays, nmse


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
