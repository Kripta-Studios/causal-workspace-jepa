"""Real Qwen3-0.6B selected-site instrumentation smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_workspace_jepa.adapters.qwen_hf_adapter import QwenAdapterConfig, QwenHFAdapter
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.hooks.names import transformer_site


def run_qwen_instrumentation_smoke(config_path: str | Path) -> dict[str, Any]:
    import torch

    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resource = require_free_disk(resource_profile)
    seed = int(config.get("seed", 41))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_name = str(config.get("model", "Qwen/Qwen3-0.6B"))
    revision = str(config["revision"])
    dtype = str(config.get("dtype", "bfloat16"))
    max_length = int(config.get("max_sequence_length", 32))
    prompts = _prompts(int(config.get("prompts", 4)))
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=model_name,
            revision=revision,
            device=str(config.get("device", "cuda")),
            dtype=dtype,
            max_length=max_length,
            local_files_only=bool(config.get("local_files_only", False)),
            token=False,
        )
    )
    layer = int(config.get("intervention_layer", len(adapter.layers) // 2))
    site = transformer_site(layer, "resid_post")
    target_site = transformer_site(len(adapter.layers) - 1, "resid_post")
    feature_ids = tuple(range(int(config.get("feature_count", 8))))

    clean_runs = []
    for prompt in prompts:
        batch = adapter.tokenize([prompt])
        clean_runs.append(adapter.forward_with_cache(batch, [site, target_site, "logits"]))
    last_source = np.stack([run.activations[site][0, -1] for run in clean_runs])
    adapter.register_mean(site, last_source.mean(axis=0))
    adapter.register_donor("donor_1", site, clean_runs[1].activations[site])

    recipient_batch = clean_runs[0].token_batch
    clean = clean_runs[0]
    specs = {
        "zero": InterventionSpec(site=site, operation="zero", positions=(-1,), feature_ids=feature_ids),
        "mean": InterventionSpec(site=site, operation="mean", positions=(-1,), feature_ids=feature_ids),
        "resample": InterventionSpec(
            site=site,
            operation="resample",
            positions=(-1,),
            feature_ids=feature_ids,
            donor_example_id="donor_1",
        ),
        "patch": InterventionSpec(
            site=site,
            operation="patch",
            positions=(-1,),
            feature_ids=feature_ids,
            donor_example_id="donor_1",
        ),
        "steer": InterventionSpec(
            site=site,
            operation="steer",
            positions=(-1,),
            feature_ids=feature_ids,
            magnitude=float(config.get("steer_magnitude", 2.0)),
        ),
    }
    effects: dict[str, dict[str, float]] = {}
    for name, spec in specs.items():
        run = adapter.forward_with_intervention(recipient_batch, spec, [target_site, "logits"])
        effects[name] = {
            "mean_abs_logit_delta": float(np.mean(np.abs(run.logits - clean.logits))),
            "target_hidden_l2": float(
                np.linalg.norm(run.activations[target_site] - clean.activations[target_site])
            ),
        }

    repeat = adapter.forward_with_cache(recipient_batch, [site, target_site, "logits"])
    deterministic_max_abs = float(np.max(np.abs(repeat.logits - clean.logits)))
    autograd_adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=model_name,
            revision=revision,
            device=str(config.get("device", "cuda")),
            dtype=dtype,
            max_length=max_length,
            local_files_only=True,
            preserve_autograd=True,
            token=False,
        ),
        model=adapter.model,
        tokenizer=adapter.tokenizer,
    )
    grad_run = autograd_adapter.forward_with_cache(autograd_adapter.tokenize([prompts[0]]), [site])
    last_position = int(grad_run.token_batch.attention_mask[0].sum().item() - 1)
    scalar = grad_run.logits[0, last_position].float().max()
    gradient = torch.autograd.grad(scalar, grad_run.activations[site])[0]
    gradient_norm = float(gradient.float().norm().detach().cpu())

    direct_nonzero = all(value["mean_abs_logit_delta"] > 0 for value in effects.values())
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "LLM-QWEN-001")),
        "status": "SMOKE_VALIDATED",
        "evidence_level": "Causal mediation",
        "model": model_name,
        "model_revision": adapter._metadata()["resolved_revision"],
        "requested_revision": revision,
        "seed": seed,
        "prompts": len(prompts),
        "intervention_site": site,
        "target_site": target_site,
        "feature_ids": list(feature_ids),
        "operation_effects": effects,
        "deterministic_repeat_max_abs_logit_error": deterministic_max_abs,
        "autograd_selected_logit_gradient_norm": gradient_norm,
        "hardware": resource.as_dict(),
        "passes": {
            "resolved_revision_matches_pin": adapter._metadata()["resolved_revision"] == revision,
            "deterministic_replay": deterministic_max_abs == 0.0,
            "all_operations_change_logits": direct_nonzero,
            "autograd_nonzero": np.isfinite(gradient_norm) and gradient_norm > 0.0,
        },
        "claim": (
            "Selected Qwen3-0.6B residual interventions directly change downstream hidden states "
            "and logits. This validates instrumentation only; it is not a circuit or workspace claim."
        ),
    }
    metrics["all_passed"] = bool(all(metrics["passes"].values()))
    output_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen3_0_6b_instrumentation_smoke.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path}",
        resource_profile=resource_profile,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output_path.with_suffix(".provenance.json"),
        provenance,
        extra={"metrics": output_path.as_posix(), "all_passed": metrics["all_passed"]},
    )
    if not metrics["all_passed"]:
        raise RuntimeError(f"Qwen instrumentation smoke failed: {metrics['passes']}")
    return metrics


def _prompts(count: int) -> list[str]:
    base = [
        "Paris is the capital city of France .",
        "Madrid is the capital city of Spain .",
        "Water freezes when the temperature becomes cold .",
        "Plants grow when they receive light and water .",
    ]
    return [base[index % len(base)] for index in range(count)]
