#!/usr/bin/env python
"""Run configured experiments."""

from __future__ import annotations

import argparse
import json
import sys

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.experiments.llm.gpt2_medium_intervention_smoke import (
    run_gpt2_medium_intervention_smoke,
)
from causal_workspace_jepa.experiments.llm.gpt2_medium_mechanistic_study import (
    run_gpt2_medium_mechanistic_study,
)
from causal_workspace_jepa.experiments.llm.gpt2_medium_semantic_composition_study import (
    run_gpt2_medium_semantic_composition_study,
)
from causal_workspace_jepa.experiments.llm.mock_qwen_intervention_jepa_smoke import (
    run_mock_qwen_intervention_jepa_smoke,
)
from causal_workspace_jepa.experiments.llm.qwen_instrumentation_smoke import (
    run_qwen_instrumentation_smoke,
)
from causal_workspace_jepa.experiments.llm.qwen_context_geometry_study import (
    run_qwen_context_geometry_study,
)
from causal_workspace_jepa.experiments.llm.qwen_jvp_audit import run_qwen_jvp_audit
from causal_workspace_jepa.experiments.llm.qwen_element_layer_geometry_study import (
    run_qwen_element_layer_geometry_study,
)
from causal_workspace_jepa.experiments.llm.qwen_population_jacobian_confirmation import (
    run_qwen_population_jacobian_confirmation,
)
from causal_workspace_jepa.experiments.llm.qwen_target_encoder_ijepa_study import (
    run_qwen_target_encoder_ijepa_study,
)
from causal_workspace_jepa.experiments.llm.qwen_intervention_jepa_study import (
    run_qwen_intervention_jepa_study,
)
from causal_workspace_jepa.experiments.world_model.manifold_workspace_study import (
    run_manifold_workspace_study,
)
from causal_workspace_jepa.experiments.world_model.lewm_population_geometry_study import (
    run_lewm_population_geometry_study,
)
from causal_workspace_jepa.experiments.world_model.lewm_action_path_geometry_study import (
    run_lewm_action_path_geometry_study,
)
from causal_workspace_jepa.experiments.world_model.multitask_workspace_study import (
    run_multitask_workspace_study,
)
from causal_workspace_jepa.experiments.world_model.tier0_mechanistic_study import (
    run_tier0_mechanistic_study,
)
from causal_workspace_jepa.experiments.world_model.tiny_jepa_smoke import run_tiny_jepa_smoke
from causal_workspace_jepa.experiments.world_model.workspace_discovery_study import (
    run_workspace_discovery_study,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_id = str(config.get("id", ""))
    if experiment_id == "WM-T0-001":
        metrics = run_tiny_jepa_smoke(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-T0-002":
        metrics = run_tier0_mechanistic_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-T0-003":
        metrics = run_workspace_discovery_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-T0-004":
        metrics = run_manifold_workspace_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-T0-005":
        metrics = run_multitask_workspace_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-LEWM-001":
        from causal_workspace_jepa.experiments.world_model.lewm_reproduction_study import (
            run_lewm_reproduction_study,
        )

        metrics = run_lewm_reproduction_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-EBJEPA-CONTRACT-001":
        from causal_workspace_jepa.experiments.world_model.eb_jepa_contract_smoke import (
            run_eb_jepa_contract_smoke,
        )

        metrics = run_eb_jepa_contract_smoke(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-EBJEPA-RUNTIME-001":
        from causal_workspace_jepa.experiments.world_model.eb_jepa_runtime_compatibility import (
            run_eb_jepa_runtime_compatibility,
        )

        metrics = run_eb_jepa_runtime_compatibility(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "WM-POPULATION-JACOBIAN-001":
        metrics = run_lewm_population_geometry_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id in {
        "WM-ACTION-PATH-CALIBRATION-001",
        "WM-ACTION-PATH-CALIBRATION-002",
        "WM-ACTION-PATH-GEOMETRY-001",
    }:
        metrics = run_lewm_action_path_geometry_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-MOCK-001":
        metrics = run_mock_qwen_intervention_jepa_smoke(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-GPT2-001":
        metrics = run_gpt2_medium_intervention_smoke(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-GPT2-002":
        metrics = run_gpt2_medium_mechanistic_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-GPT2-003":
        metrics = run_gpt2_medium_semantic_composition_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-QWEN-001":
        metrics = run_qwen_instrumentation_smoke(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-IJEPA-001":
        metrics = run_qwen_intervention_jepa_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id in {"LLM-QWEN-JVP-AUDIT-001", "LLM-QWEN-JVP-AUDIT-002"}:
        metrics = run_qwen_jvp_audit(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-TARGET-IJEPA-001":
        metrics = run_qwen_target_encoder_ijepa_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-CONTEXT-GEOMETRY-001":
        metrics = run_qwen_context_geometry_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id == "LLM-POPULATION-JACOBIAN-001":
        metrics = run_qwen_population_jacobian_confirmation(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    if experiment_id in {
        "LLM-ELEMENT-LAYER-GEOMETRY-001",
        "LLM-STATE-LAYER-GEOMETRY-001",
        "LLM-STATE-ONESHOT-LAYER-GEOMETRY-001",
        "LLM-COUNTRY-CODE-LAYER-GEOMETRY-001",
    }:
        metrics = run_qwen_element_layer_geometry_study(args.config)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 0
    print(f"NOT_STARTED: no experiment runner is registered for {args.config}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
