"""GPT-2 Medium hidden-state intervention smoke experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.models.intervention_jepa import (
    LayerTransitionInterventionJEPA,
    LinearContextBaseline,
    effect_correlation,
    no_change_mse,
)


def run_gpt2_medium_intervention_smoke(config_path: str | Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/cpu_vps.yaml"))
    require_free_disk(resource_profile)
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_name = str(config.get("model", "gpt2-medium"))
    cache_dir = str(config.get("cache_dir", ".cache/huggingface"))
    max_length = int(config.get("max_sequence_length", 32))
    intervention_layer = int(config.get("intervention_layer", 12))
    magnitude = float(config.get("intervention_magnitude", 1.0))
    prompts = _gpt2_smoke_prompts(int(config.get("prompts", 8)))

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    last_positions = batch["attention_mask"].sum(dim=1) - 1
    with torch.no_grad():
        clean = model(**batch, output_hidden_states=True)
    source_hidden = clean.hidden_states[intervention_layer][
        torch.arange(len(prompts)),
        last_positions,
    ].detach()
    target_clean = clean.hidden_states[-1][torch.arange(len(prompts)), last_positions].detach()
    logits_clean = clean.logits[torch.arange(len(prompts)), last_positions].detach()

    contexts: list[np.ndarray] = []
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    prompt_ids: list[int] = []
    interventions = _intervention_specs(
        hidden_size=source_hidden.shape[-1],
        magnitude=magnitude,
        feature_count=int(config.get("intervention_features", 2)),
    )
    for prompt_id in range(len(prompts)):
        for feature_id, signed_magnitude in interventions:
            single_batch = {key: value[prompt_id : prompt_id + 1] for key, value in batch.items()}
            with torch.no_grad():
                intervened = _forward_with_resid_steer(
                    model=model,
                    batch=single_batch,
                    layer=intervention_layer,
                    position=int(last_positions[prompt_id].item()),
                    batch_index=0,
                    feature_id=feature_id,
                    magnitude=signed_magnitude,
                )
            target_delta = (
                intervened.hidden_states[-1][0, last_positions[prompt_id]].detach() - target_clean[prompt_id]
            )
            logit_delta = intervened.logits[0, last_positions[prompt_id]].detach() - logits_clean[prompt_id]
            contexts.append(source_hidden[prompt_id].cpu().numpy().astype(np.float32))
            features.append(_intervention_features(feature_id, signed_magnitude, source_hidden.shape[-1]))
            targets.append(torch.cat([target_delta, logit_delta], dim=0).cpu().numpy().astype(np.float32))
            prompt_ids.append(prompt_id)

    context_array = np.asarray(contexts, dtype=np.float32)
    feature_array = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    prompt_ids_array = np.asarray(prompt_ids)
    heldout_start = int(len(prompts) * 0.75)
    train_mask = prompt_ids_array < heldout_start
    test_mask = ~train_mask
    ijepa = LayerTransitionInterventionJEPA.fit(
        context_array[train_mask],
        feature_array[train_mask],
        target_array[train_mask],
        ridge=1e-3,
    )
    linear = LinearContextBaseline.fit(context_array[train_mask], target_array[train_mask], ridge=1e-3)
    predicted = ijepa.predict(context_array[test_mask], feature_array[test_mask])
    observed = target_array[test_mask].reshape(test_mask.sum(), -1)
    ijepa_mse = float(np.mean((predicted - observed) ** 2))
    selected_layers = config.get("selected_layers", [0, 6, 12, 18, 23])
    metrics: dict[str, Any] = {
        "experiment_id": config.get("id", "LLM-GPT2-001"),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Causal mediation",
        "model": model_name,
        "seed": seed,
        "prompts": len(prompts),
        "max_sequence_length": max_length,
        "selected_layers": selected_layers,
        "intervention_layer": intervention_layer,
        "intervention_site": f"transformer.h.{intervention_layer}.resid_post",
        "target": "final_residual_and_logits_at_last_token",
        "train_examples": int(train_mask.sum()),
        "test_examples": int(test_mask.sum()),
        "ijepa_mse": ijepa_mse,
        "no_change_mse": no_change_mse(target_array[test_mask]),
        "mean_effect_mse": ijepa.mean_baseline_mse(target_array[test_mask]),
        "linear_context_mse": linear.mse(context_array[test_mask], target_array[test_mask]),
        "effect_correlation": effect_correlation(predicted, observed),
        "direct_intervention_mean_abs_logit_delta": float(
            np.mean(np.abs(target_array[:, source_hidden.shape[-1] :]))
        ),
        "claim": (
            "GPT-2 Medium hidden-state interventions produce measurable downstream hidden-state "
            "and logit changes; this is not a workspace/J-space discovery."
        ),
    }
    metrics["passes"] = {
        "direct_intervention_changes_logits": bool(metrics["direct_intervention_mean_abs_logit_delta"] > 0.0),
        "beats_no_change": bool(metrics["ijepa_mse"] < metrics["no_change_mse"]),
        "finite_effect_correlation": bool(np.isfinite(metrics["effect_correlation"])),
    }
    metrics["all_passed"] = bool(all(metrics["passes"].values()))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output_path = Path(str(config.get("output_metrics", "artifacts/metrics/gpt2_medium_intervention_smoke.json")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        "artifacts/metrics/gpt2_medium_intervention_smoke.provenance.json",
        provenance,
        extra={"metrics": str(output_path), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"GPT-2 Medium smoke failed checks: {metrics['passes']}")
    return metrics


def _forward_with_resid_steer(
    *,
    model: Any,
    batch: dict[str, Any],
    layer: int,
    position: int,
    batch_index: int,
    feature_id: int,
    magnitude: float,
) -> Any:
    block = model.transformer.h[layer]

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[batch_index, position, feature_id] += magnitude
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[batch_index, position, feature_id] += magnitude
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        return model(**batch, output_hidden_states=True)
    finally:
        handle.remove()


def _intervention_specs(hidden_size: int, magnitude: float, feature_count: int) -> list[tuple[int, float]]:
    candidates = [0, hidden_size // 4, hidden_size // 2, (3 * hidden_size) // 4]
    feature_ids = candidates[: max(1, min(feature_count, len(candidates)))]
    return [(feature_id, sign * magnitude) for feature_id in feature_ids for sign in (-1.0, 1.0)]


def _intervention_features(feature_id: int, magnitude: float, hidden_size: int) -> np.ndarray:
    one_hot = np.zeros(hidden_size, dtype=np.float32)
    one_hot[feature_id] = magnitude
    return np.concatenate([one_hot, np.array([magnitude], dtype=np.float32)])


def _gpt2_smoke_prompts(count: int) -> list[str]:
    base = [
        "Paris is the capital of France and the Eiffel Tower is in",
        "The chemical symbol for water is H two O and ice is",
        "In a small maze, the agent must move toward the bright",
        "A careful scientist changes one variable and observes the",
        "The recipe says to add flour, sugar, and then",
        "When the switch is flipped, the circuit sends current through the",
        "A chess player plans two moves ahead before moving the",
        "The story begins in a quiet library where the detective finds a",
    ]
    return [base[index % len(base)] for index in range(count)]
