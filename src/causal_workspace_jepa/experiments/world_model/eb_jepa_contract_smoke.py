"""Official EB-JEPA source-contract and recurrent-decomposition smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from causal_workspace_jepa.adapters.eb_jepa_adapter import (
    EBJEPAAdapter,
    verify_eb_jepa_checkout,
)
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec


def run_eb_jepa_contract_smoke(config_path: str | Path) -> dict[str, Any]:
    """Run the exact public Impala/GRU object contract without training a model."""

    import torch

    config = load_config(config_path)
    source_root = Path(str(config["source_root"])).resolve()
    revision = str(config["revision"])
    source_metadata = verify_eb_jepa_checkout(source_root, expected_revision=revision)
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
    from eb_jepa.jepa import JEPAbase

    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    resource = require_free_disk(resource_profile)
    seed = int(config.get("seed", 23))
    steps = int(config.get("steps", 3))
    tolerance = float(config.get("reconstruction_atol", 1e-6))
    image_size = int(config.get("image_size", 65))
    device = str(config.get("device", "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, image_size, image_size),
    )
    predictor = RNNPredictor(hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln)
    model = JEPAbase(encoder, torch.nn.Identity(), predictor).to(device).eval()
    adapter = EBJEPAAdapter(model, device=device, upstream_metadata=source_metadata)

    observations = np.random.default_rng(seed).normal(
        size=(1, 2, image_size, image_size)
    ).astype(np.float32)
    actions = np.random.default_rng(seed + 6).normal(size=(1, steps, 2)).astype(np.float32)
    latent = adapter.encode(observations)
    reconstruction_error = adapter.validate_predictor_reconstruction(
        latent, actions, atol=tolerance
    )
    clean = adapter.predict(latent, actions, return_intermediates=True)
    intervention_step = min(1, steps - 1)
    edited = adapter.predict_with_intervention(
        latent,
        actions,
        InterventionSpec(
            site="predictor.update_gate",
            operation="zero",
            positions=(intervention_step,),
            feature_ids=(0,),
            seed=seed,
        ),
        return_intermediates=True,
    )
    clean_gate = clean.intermediates["predictor.update_gate"]
    edited_gate = edited.intermediates["predictor.update_gate"]
    upstream_gate_error = float(np.max(np.abs(edited_gate[:, 0] - clean_gate[:, 0])))
    same_step_other_feature_error = float(
        np.max(
            np.abs(
                edited_gate[:, intervention_step, 1:]
                - clean_gate[:, intervention_step, 1:]
            )
        )
    )
    downstream_latent_effect = float(
        np.linalg.norm(edited.predicted_latents - clean.predicted_latents)
    )
    torch_version = str(torch.__version__)
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    official_environment_exact = python_version.startswith("3.12.") and torch_version.startswith(
        "2.6."
    )
    passes = {
        "source_revision_matches_pin": source_metadata["resolved_revision"] == revision,
        "source_checkout_clean": bool(source_metadata["clean"]),
        "official_latent_shape": latent.tensor.shape == (1, 1, 512),
        "official_prediction_shape": clean.predicted_latents.shape == (1, steps, 512),
        "gate_trace_shape": clean_gate.shape == (1, steps, 512),
        "gru_reconstruction": reconstruction_error <= tolerance,
        "intervention_is_upstream_safe": upstream_gate_error == 0.0,
        "intervention_is_same_step_feature_specific": same_step_other_feature_error == 0.0,
        "gate_intervention_has_downstream_effect": downstream_latent_effect > 0.0,
    }
    metrics: dict[str, Any] = {
        "experiment_id": str(config.get("id", "WM-EBJEPA-CONTRACT-001")),
        "status": "SMOKE_VALIDATED" if all(passes.values()) else "FAILED_REGISTERED_GATES",
        "evidence_level": "Availability",
        "seed": seed,
        "source": source_metadata,
        "runtime": {
            "python": python_version,
            "torch": torch_version,
            "device": device,
            "official_requires_python": "3.12.*",
            "official_requires_torch": "2.6.0",
            "official_environment_exact": official_environment_exact,
        },
        "architecture": {
            "encoder": "official ImpalaEncoder",
            "latent_width": 512,
            "predictor": "official one-layer torch.nn.GRU",
            "action_width": 2,
            "steps": steps,
        },
        "shapes": {
            "latent": list(latent.tensor.shape),
            "prediction": list(clean.predicted_latents.shape),
            "update_gate": list(clean_gate.shape),
        },
        "max_gru_reconstruction_error": reconstruction_error,
        "reconstruction_atol": tolerance,
        "upstream_gate_error_after_later_edit": upstream_gate_error,
        "same_step_other_feature_error": same_step_other_feature_error,
        "downstream_latent_l2_after_gate_edit": downstream_latent_effect,
        "named_activation_points": list(adapter.named_activation_points()),
        "passes": passes,
        "all_passed": bool(all(passes.values())),
        "hardware": resource.as_dict(),
        "claim": (
            "The pinned official EB-JEPA Impala/one-layer-GRU object contract can be loaded, and "
            "the GRU transition can be decomposed and intervened on with numerical replay. This "
            "is an engineering smoke only: no checkpoint, planning competence, circuit, or "
            "workspace claim is evaluated."
        ),
        "limitations": [
            "The current runtime differs from upstream's exact Python 3.12/Torch 2.6 pin.",
            "Weights are random; no published or locally trained checkpoint is used.",
            "A gate edit with a downstream effect does not establish task mediation or specificity.",
        ],
    }
    output_path = Path(
        str(config.get("output_metrics", "artifacts/metrics/eb_jepa_contract_smoke.json"))
    )
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {Path(config_path).as_posix()}",
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
        raise RuntimeError(f"EB-JEPA contract smoke failed: {passes}")
    return metrics
