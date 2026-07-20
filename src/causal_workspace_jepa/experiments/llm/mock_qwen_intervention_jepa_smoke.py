"""Mock-Qwen intervention JEPA smoke experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.adapters.mock_transformer import (
    MockInstrumentedCausalLM,
    MockTransformerConfig,
    build_mock_prompts,
)
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.hooks.names import transformer_site
from causal_workspace_jepa.models.intervention_jepa import (
    LayerTransitionInterventionJEPA,
    LinearContextBaseline,
    effect_correlation,
    no_change_mse,
)


def run_mock_qwen_intervention_jepa_smoke(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    require_free_disk(resource_profile)
    seed = int(config.get("seed", 0))
    model = MockInstrumentedCausalLM(
        MockTransformerConfig(
            vocab_size=int(config.get("vocab_size", 32)),
            hidden_size=int(config.get("hidden_size", 16)),
            layers=int(config.get("layers", 3)),
            seed=seed,
        )
    )
    prompts = build_mock_prompts(int(config.get("prompts", 64)))
    batch = model.tokenize(prompts)
    source_site = transformer_site(0, "resid_post")
    intervention_site = transformer_site(1, "mlp_out")
    target_site = transformer_site(2, "resid_post")
    sites = [source_site, intervention_site, target_site, "logits"]
    clean = model.forward_with_cache(batch, sites)
    last_positions = np.maximum(batch.attention_mask.sum(axis=1).astype(int) - 1, 0)
    intervention_specs = _intervention_grid(intervention_site, last_positions)
    contexts: list[np.ndarray] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prompt_ids: list[int] = []
    for prompt_id, position in enumerate(last_positions):
        context_vec = clean.activations[source_site][prompt_id, position]
        for spec in intervention_specs[prompt_id]:
            intervened = model.forward_with_intervention(batch, spec, sites)
            target_delta = (
                intervened.activations[target_site][prompt_id, position]
                - clean.activations[target_site][prompt_id, position]
            )
            logit_delta = intervened.logits[prompt_id, position] - clean.logits[prompt_id, position]
            contexts.append(context_vec)
            features.append(_intervention_features(spec, model.config.hidden_size))
            targets.append(np.concatenate([target_delta, logit_delta], axis=0))
            prompt_ids.append(prompt_id)
    context_array = np.asarray(contexts, dtype=np.float32)
    feature_array = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    prompt_ids_array = np.asarray(prompt_ids)
    heldout_fraction = float(config.get("heldout_fraction", 0.25))
    heldout_start = int(len(prompts) * (1.0 - heldout_fraction))
    train_mask = prompt_ids_array < heldout_start
    test_mask = ~train_mask
    ijepa = LayerTransitionInterventionJEPA.fit(
        context_array[train_mask],
        feature_array[train_mask],
        target_array[train_mask],
    )
    linear = LinearContextBaseline.fit(context_array[train_mask], target_array[train_mask])
    predicted = ijepa.predict(context_array[test_mask], feature_array[test_mask])
    observed = target_array[test_mask].reshape(test_mask.sum(), -1)
    ijepa_mse = float(np.mean((predicted - observed) ** 2))
    metrics = {
        "experiment_id": config.get("id", "LLM-MOCK-001"),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Availability",
        "model": "mock_transformer_not_qwen",
        "seed": seed,
        "prompts": len(prompts),
        "train_examples": int(train_mask.sum()),
        "test_examples": int(test_mask.sum()),
        "source_site": source_site,
        "intervention_site": intervention_site,
        "target_site": target_site,
        "ijepa_mse": ijepa_mse,
        "no_change_mse": no_change_mse(target_array[test_mask]),
        "mean_effect_mse": ijepa.mean_baseline_mse(target_array[test_mask]),
        "linear_context_mse": linear.mse(context_array[test_mask], target_array[test_mask]),
        "effect_correlation": effect_correlation(predicted, observed),
    }
    metrics["passes"] = {
        "beats_no_change": metrics["ijepa_mse"] < metrics["no_change_mse"],
        "beats_mean_effect": metrics["ijepa_mse"] < metrics["mean_effect_mse"],
        "beats_linear_context": metrics["ijepa_mse"] < metrics["linear_context_mse"],
        "positive_effect_correlation": metrics["effect_correlation"] > 0.5,
    }
    metrics["all_passed"] = bool(all(metrics["passes"].values()))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output_path = Path(str(config.get("output_metrics", "artifacts/metrics/mock_qwen_intervention_jepa_smoke.json")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        "artifacts/metrics/mock_qwen_intervention_jepa_smoke.provenance.json",
        provenance,
        extra={"metrics": str(output_path), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"mock intervention JEPA smoke failed: {metrics['passes']}")
    return metrics


def _intervention_grid(site: str, positions: np.ndarray) -> list[list[InterventionSpec]]:
    grid: list[list[InterventionSpec]] = []
    for prompt_id, position in enumerate(positions):
        prompt_specs = []
        for feature_id in (0, 1, 2, 3):
            for magnitude in (-0.75, 0.75):
                prompt_specs.append(
                    InterventionSpec(
                        site=site,
                        operation="steer",
                        positions=(int(position),),
                        feature_ids=(feature_id,),
                        magnitude=magnitude,
                        donor_example_id=None,
                        seed=prompt_id * 17 + feature_id,
                    )
                )
        grid.append(prompt_specs)
    return grid


def _intervention_features(spec: InterventionSpec, hidden_size: int) -> np.ndarray:
    feature_id = 0 if spec.feature_ids is None else spec.feature_ids[0]
    one_hot = np.zeros(hidden_size, dtype=np.float32)
    one_hot[feature_id] = float(spec.magnitude)
    return np.concatenate([one_hot, np.array([spec.magnitude], dtype=np.float32)])
