"""Outcome-blind readiness registry for Qwen circuit-localization substrates.

This module performs no model forward, no network access, and no protected-data access. It exists
so CRCT-QWEN-BRIDGE-001 does not silently substitute ad-hoc sparse features for missing/frozen
artifacts. An executable substrate needs an exact model target, a frozen artifact/implementation,
and a direct native-model validation path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def build_readiness() -> dict[str, Any]:
    return {
        "schema_version": "qwen_substrate_readiness_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "READINESS_AUDIT_COMPLETE",
        "target_model": "Qwen/Qwen3-0.6B",
        "substrates": {
            "native_residual_and_module_states": {
                "status": "EXECUTABLE_PHASE0",
                "role": "native reference and exact replay substrate",
                "claim_boundary": "causal effect only after direct native-model replay/patching",
            },
            "native_attention_head_output_slices": {
                "status": "READINESS_ONLY_PHASE0",
                "role": "exact pre-o_proj linear head decomposition",
                "claim_boundary": (
                    "does not explain attention routing or post-RoPE/GQA query-key scores"
                ),
            },
            "post_rope_gqa_qk_interactions": {
                "status": "DEFERRED_EXACT_RECONSTRUCTION_REQUIRED",
                "role": "future attention-routing localization",
                "required_before_use": [
                    "exact post-RoPE query/key reconstruction",
                    "GQA head mapping reconstruction",
                    "attention-score reconstruction",
                    "softmax-pattern reconstruction",
                    "native intervention validation",
                ],
            },
            "hvp_screen_flag_fix": {
                "status": "DEFERRED_PRIMARY_METHOD_REPRODUCTION",
                "role": "future reliability-aware attribution-patching comparator",
                "note": (
                    "Phase0 measures exact T1/T2 finite fidelity but does not relabel that as the "
                    "published Screen-Flag-Fix reliability score."
                ),
            },
            "qwen_scope_sae": {
                "status": "DEFERRED_NO_FROZEN_EXACT_0_6B_ARTIFACT",
                "role": "future sparse-feature localization comparator",
                "next_registered_target": "Qwen3-1.7B-Base",
                "claim_boundary": "feature labels/steering require direct necessity/sufficiency",
            },
            "cross_layer_transcoder": {
                "status": "DEFERRED_NO_FROZEN_QWEN_0_6B_REPLACEMENT",
                "role": "future sparse replacement/circuit graph comparator",
                "claim_boundary": "replacement graph requires validation in the native model",
            },
            "sparse_weight_decomposition": {
                "status": "DEFERRED_NO_FROZEN_REPOSITORY_IMPLEMENTATION",
                "role": "future weight-native addressable circuit-unit comparator",
                "claim_boundary": "must reproduce fidelity plus necessity/sufficiency first",
            },
        },
        "scientific_boundary": {
            "model_forward_executed": False,
            "network_access_performed": False,
            "protected_data_accessed": False,
            "missing_substrate_substituted_with_ad_hoc_features": False,
            "substrate_comparison_claim_permitted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_readiness()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
