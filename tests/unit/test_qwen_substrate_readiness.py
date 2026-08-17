from causal_workspace_jepa.experiments.llm.qwen_substrate_readiness import build_readiness


def test_substrate_readiness_never_promotes_missing_sparse_artifacts() -> None:
    payload = build_readiness()
    assert payload["target_model"] == "Qwen/Qwen3-0.6B"
    assert payload["substrates"]["native_residual_and_module_states"]["status"] == (
        "EXECUTABLE_PHASE0"
    )
    for name in (
        "post_rope_gqa_qk_interactions",
        "hvp_screen_flag_fix",
        "qwen_scope_sae",
        "cross_layer_transcoder",
        "sparse_weight_decomposition",
    ):
        assert payload["substrates"][name]["status"].startswith("DEFERRED_")
    boundary = payload["scientific_boundary"]
    assert boundary["model_forward_executed"] is False
    assert boundary["network_access_performed"] is False
    assert boundary["protected_data_accessed"] is False
    assert boundary["missing_substrate_substituted_with_ad_hoc_features"] is False
    assert boundary["substrate_comparison_claim_permitted"] is False
