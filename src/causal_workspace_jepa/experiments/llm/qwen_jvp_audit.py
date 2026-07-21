"""FP32 autograd-JVP audit of the original Qwen nonlinear-advantage claim."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from causal_workspace_jepa.adapters.qwen_hf_adapter import QwenAdapterConfig, QwenHFAdapter
from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance
from causal_workspace_jepa.common.resources import require_free_disk
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.data.activation_store import read_hdf5_shards, write_hdf5_shards
from causal_workspace_jepa.experiments.llm.qwen_intervention_dataset import (
    OPERATIONS,
    _edited_source,
    qwen_intervention_prompts,
)
from causal_workspace_jepa.hooks.names import transformer_site
from causal_workspace_jepa.models.intervention_jepa import (
    NeuralInterventionJEPA,
    RidgeInterventionPredictor,
    TinyMLPInterventionPredictor,
    effect_correlation,
)


DEFAULT_EPSILONS = (2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1)


def run_qwen_jvp_audit(config_path: str | Path) -> dict[str, Any]:
    """Re-execute the frozen Qwen intervention grid in FP32 and audit exact JVPs."""

    import torch

    started = time.perf_counter()
    config_path = Path(config_path)
    config = load_config(config_path)
    resource_profile = str(config.get("resource_profile", "configs/resource/gpu_12gb.yaml"))
    hardware = require_free_disk(resource_profile)
    seed = int(config.get("seed", 127))
    provenance = collect_provenance(
        command=f"python scripts/run_experiment.py --config {config_path.as_posix()}",
        resource_profile=resource_profile,
        seed=seed,
    )
    if provenance.git_dirty:
        raise RuntimeError("LLM-QWEN-JVP-AUDIT-001 requires a clean committed worktree")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    original, records = read_hdf5_shards(str(config["dataset_dir"]))
    prompts, split_ids, _families = qwen_intervention_prompts()
    layers = tuple(int(value) for value in config.get("selected_layers", [7, 14, 21]))
    adapter = QwenHFAdapter(
        QwenAdapterConfig(
            model_name=str(config.get("model", "Qwen/Qwen3-0.6B")),
            revision=str(config["revision"]),
            device=str(config.get("device", "cuda")),
            dtype="float32",
            max_length=int(config.get("max_sequence_length", 24)),
            local_files_only=bool(config.get("local_files_only", True)),
            preserve_autograd=True,
            attn_implementation=str(config.get("attn_implementation", "eager")),
            token=False,
        )
    )
    adapter.model.eval()
    hidden_size = int(adapter.model.config.hidden_size)
    source_sites = [transformer_site(layer, "resid_post") for layer in layers]
    target_site = transformer_site(len(adapter.layers) - 1, "resid_post")
    capture_sites = [*source_sites, target_site, "logits"]
    with torch.no_grad():
        clean_runs = [
            adapter.forward_with_cache(adapter.tokenize([prompt]), capture_sites)
            for prompt in prompts
        ]
    train_indices = np.flatnonzero(np.asarray(split_ids) == 0)
    for site in source_sites:
        mean = torch.stack(
            [clean_runs[index].activations[site][0, -1] for index in train_indices]
        ).mean(dim=0)
        adapter.register_mean(site, mean)
        for prompt_index, run in enumerate(clean_runs):
            adapter.register_donor(f"prompt_{prompt_index}", site, run.activations[site])

    logit_ids = np.asarray(
        json.loads(Path(str(config["source_metrics"])).read_text(encoding="utf-8"))[
            "logit_target_ids"
        ],
        dtype=np.int64,
    )
    hidden_projection = _regenerate_hidden_projection(
        hidden_size=hidden_size,
        context_dim=int(config.get("context_projection_dim", 32)),
        hidden_target_dim=int(config.get("hidden_target_dim", 32)),
        seed=int(config.get("dataset_seed", 53)),
    )
    projection_tensor = torch.as_tensor(hidden_projection, device=adapter.device)
    logit_tensor = torch.as_tensor(logit_ids, device=adapter.device)
    clean_targets = [
        _target_vector(run, target_site, projection_tensor, logit_tensor)
        for run in clean_runs
    ]
    replay = adapter.forward_with_cache(clean_runs[0].token_batch, capture_sites)
    replay_target = _target_vector(replay, target_site, projection_tensor, logit_tensor)
    clean_replay_max_abs = float((replay_target - clean_targets[0]).abs().max().cpu())

    epsilons = tuple(float(value) for value in config.get("epsilons", DEFAULT_EPSILONS))
    validation_epsilon = float(config.get("validation_epsilon", 0.125))
    quadratic_epsilon = float(config.get("quadratic_epsilon", 0.125))
    if validation_epsilon not in epsilons or quadratic_epsilon not in epsilons:
        raise ValueError("validation and quadratic epsilons must be members of the fixed sweep")
    direct_effects: list[np.ndarray] = []
    jvps: list[np.ndarray] = []
    central_rows: list[np.ndarray] = []
    quadratic_rows: list[np.ndarray] = []
    semantic_reconstruction_errors: list[float] = []
    direct_source_semantic_errors: list[float] = []
    source_endpoint_errors: list[float] = []
    source_endpoint_tolerances: list[float] = []
    downstream_endpoint_relative_errors: list[float] = []
    fp32_clean_top_tokens: list[int] = []
    fp32_top_tokens: list[int] = []

    for index, record in enumerate(records):
        prompt_index = int(original["prompt_id"][index])
        site = str(record["site"])
        spec = InterventionSpec.from_dict(record["intervention"])
        clean = clean_runs[prompt_index]
        clean_source_tensor = clean.activations[site][0, -1].detach().float()
        clean_source = clean_source_tensor.cpu().numpy()
        donor = None
        if spec.donor_example_id is not None:
            donor = (
                adapter._donors[(spec.donor_example_id, site)][0, -1]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        edited = _edited_source(
            clean_source,
            spec,
            mean=adapter._means[site].detach().float().cpu().numpy(),
            donor=donor,
        )
        edited_tensor = torch.as_tensor(edited, device=adapter.device)
        direction = torch.as_tensor(edited - clean_source, device=adapter.device)
        with torch.no_grad():
            direct_run = adapter.forward_with_intervention(
                clean.token_batch, spec, [site, target_site, "logits"]
            )
            direct_target = _target_vector(
                direct_run, target_site, projection_tensor, logit_tensor
            )
        direct_source = direct_run.activations[site][0, -1].detach().float()
        direct_source_semantic_errors.append(
            float((direct_source - edited_tensor).abs().max().cpu())
        )
        source_endpoint = clean_source_tensor + direction
        source_endpoint_errors.append(
            float((source_endpoint - edited_tensor).abs().max().cpu())
        )
        source_endpoint_tolerances.append(
            _float32_endpoint_tolerance(clean_source_tensor, edited_tensor)
        )
        direct_effect = direct_target - clean_targets[prompt_index]
        scalar_function = _directional_output_function(
            adapter,
            clean.token_batch,
            source_site=site,
            target_site=target_site,
            direction=direction,
            hidden_projection=projection_tensor,
            logit_ids=logit_tensor,
        )
        scale = torch.zeros((), device=adapter.device, requires_grad=True)
        _, jvp = torch.autograd.functional.jvp(
            scalar_function,
            scale,
            torch.ones_like(scale),
            create_graph=False,
            strict=True,
        )
        central: list[torch.Tensor] = []
        outputs: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for epsilon in epsilons:
                plus = scalar_function(torch.tensor(epsilon, device=adapter.device))
                minus = scalar_function(torch.tensor(-epsilon, device=adapter.device))
                outputs[epsilon] = (plus, minus)
                central.append((plus - minus) / (2.0 * epsilon))
            dense_full = scalar_function(torch.ones((), device=adapter.device))
        semantic_reconstruction_errors.append(
            float((dense_full - direct_target).abs().max().cpu())
        )
        dense_effect = dense_full - clean_targets[prompt_index]
        downstream_endpoint_relative_errors.append(
            float(
                (dense_effect - direct_effect).norm().cpu()
                / max(float(direct_effect.norm().cpu()), 1e-8)
            )
        )
        plus, minus = outputs[quadratic_epsilon]
        second = (
            plus - 2.0 * clean_targets[prompt_index] + minus
        ) / (quadratic_epsilon**2)
        quadratic = jvp + 0.5 * second
        direct_effects.append(direct_effect.detach().cpu().numpy().astype(np.float32))
        jvps.append(jvp.detach().cpu().numpy().astype(np.float32))
        central_rows.append(
            torch.stack(central).detach().cpu().numpy().astype(np.float32)
        )
        quadratic_rows.append(quadratic.detach().cpu().numpy().astype(np.float32))
        fp32_clean_top_tokens.append(
            int(clean_runs[prompt_index].logits[0, -1].argmax().detach().cpu())
        )
        fp32_top_tokens.append(int(direct_run.logits[0, -1].argmax().detach().cpu()))

    evaluated = {
        "direct_effect_fp32": np.asarray(direct_effects, dtype=np.float32),
        "exact_jvp_fp32": np.asarray(jvps, dtype=np.float32),
        "central_differences_fp32": np.asarray(central_rows, dtype=np.float32),
        "quadratic_taylor_fp32": np.asarray(quadratic_rows, dtype=np.float32),
        "fp32_clean_top_token": np.asarray(fp32_clean_top_tokens, dtype=np.int64),
        "fp32_intervened_top_token": np.asarray(fp32_top_tokens, dtype=np.int64),
        "prompt_id": original["prompt_id"].astype(np.int64),
        "split_id": original["split_id"].astype(np.int64),
        "layer": original["layer"].astype(np.int64),
        "operation_id": original["operation_id"].astype(np.int64),
        "feature_id": original["feature_id"].astype(np.int64),
        "magnitude": original["magnitude"].astype(np.float32),
        "direct_source_semantic_error": np.asarray(
            direct_source_semantic_errors, dtype=np.float32
        ),
        "source_direction_endpoint_error": np.asarray(
            source_endpoint_errors, dtype=np.float32
        ),
        "source_direction_endpoint_tolerance": np.asarray(
            source_endpoint_tolerances, dtype=np.float32
        ),
        "downstream_endpoint_relative_error": np.asarray(
            downstream_endpoint_relative_errors, dtype=np.float32
        ),
    }
    storage = write_hdf5_shards(
        str(config.get("output_dir", "data/activations/qwen3_0_6b_jvp_audit_v1")),
        evaluated,
        records,
        dataset_id=str(config.get("id", "LLM-QWEN-JVP-AUDIT-001")),
        config_digest=_config_digest(config),
        max_shard_mb=float(config.get("max_shard_mb", 32)),
        budget_mb=float(config.get("activation_budget_mb", 64)),
        resume=bool(config.get("resume", True)),
    )
    metrics = _evaluate_audit(
        config,
        original,
        evaluated,
        epsilons,
        clean_replay_max_abs=clean_replay_max_abs,
        semantic_reconstruction_errors=np.asarray(semantic_reconstruction_errors),
        direct_source_semantic_errors=np.asarray(direct_source_semantic_errors),
        source_endpoint_errors=np.asarray(source_endpoint_errors),
        source_endpoint_tolerances=np.asarray(source_endpoint_tolerances),
        downstream_endpoint_relative_errors=np.asarray(
            downstream_endpoint_relative_errors
        ),
    )
    metrics.update(
        {
            "experiment_id": str(config.get("id", "LLM-QWEN-JVP-AUDIT-001")),
            "status": (
                "SMOKE_VALIDATED" if metrics["audit_passed"] else "REJECTED_NUMERICAL"
            ),
            "evidence_level": "Specificity",
            "model": str(config.get("model")),
            "model_revision": adapter._metadata()["resolved_revision"],
            "precision": "float32",
            "hardware": hardware.as_dict(),
            "storage": storage,
            "runtime_seconds": float(time.perf_counter() - started),
            "scientific_boundary": (
                "This audit can retain or withdraw the prior nonlinear-advantage claim. "
                "It cannot establish a circuit, behavior mechanism, or workspace."
            ),
        }
    )
    output_metrics = Path(
        str(config.get("output_metrics", "artifacts/metrics/qwen_jvp_audit_v1.json"))
    )
    output_manifest = Path(
        str(config.get("output_manifest", "data/manifests/qwen3_0_6b_jvp_audit_v1.json"))
    )
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_summary = {
        **storage,
        "model": str(config.get("model")),
        "model_revision": adapter._metadata()["resolved_revision"],
        "precision": "float32",
        "source_dataset": str(config.get("dataset_manifest")),
        "epsilon_sweep": list(epsilons),
        "local_data_root": str(config.get("output_dir")),
    }
    for shard in manifest_summary["shards"]:
        shard["path"] = f"{manifest_summary['local_data_root']}/{shard['path']}"
    output_manifest.write_text(
        json.dumps(manifest_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_provenance(
        output_metrics.with_suffix(".provenance.json"),
        provenance,
        extra={
            "metrics": output_metrics.as_posix(),
            "audit_passed": metrics["audit_passed"],
            "h_llm_01_retained": metrics["h_llm_01_retained"],
        },
    )
    if not metrics["audit_passed"]:
        raise RuntimeError("FP32 JVP audit failed its numerical-validity gates")
    return metrics


def _directional_output_function(
    adapter: QwenHFAdapter,
    batch: Any,
    *,
    source_site: str,
    target_site: str,
    direction: Any,
    hidden_projection: Any,
    logit_ids: Any,
) -> Callable[[Any], Any]:
    """Return differentiable F(alpha) for a dense residual edit alpha * direction."""

    source_layer = int(source_site.split(".")[1])
    target_layer = int(target_site.split(".")[1])
    if source_layer >= target_layer:
        raise ValueError("directional source must precede the target layer")

    def evaluate(scale: Any) -> Any:
        captured: dict[str, Any] = {}

        def source_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            updated = value.clone()
            updated[:, -1, :] = updated[:, -1, :] + scale.to(value) * direction.to(value)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated

        def target_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            captured["target"] = output[0] if isinstance(output, tuple) else output

        source_handle = adapter.layers[source_layer].register_forward_hook(source_hook)
        target_handle = adapter.layers[target_layer].register_forward_hook(target_hook)
        try:
            output = adapter.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
                logits_to_keep=0,
            )
        finally:
            source_handle.remove()
            target_handle.remove()
        hidden = captured["target"][0, -1].float() @ hidden_projection.float()
        logits = output.logits[0, -1, logit_ids].float()
        return adapter._torch.cat([hidden, logits])

    return evaluate


def _target_vector(run: Any, target_site: str, hidden_projection: Any, logit_ids: Any) -> Any:
    import torch

    hidden = run.activations[target_site][0, -1].float() @ hidden_projection.float()
    logits = run.logits[0, -1, logit_ids].float()
    return torch.cat([hidden, logits])


def _regenerate_hidden_projection(
    *, hidden_size: int, context_dim: int, hidden_target_dim: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rng.normal(0.0, 1.0 / np.sqrt(context_dim), size=(hidden_size, context_dim))
    rng.normal(0.0, 1.0 / np.sqrt(context_dim), size=(hidden_size, context_dim))
    return rng.normal(
        0.0,
        1.0 / np.sqrt(hidden_target_dim),
        size=(hidden_size, hidden_target_dim),
    ).astype(np.float32)


def _float32_endpoint_tolerance(clean: Any, edited: Any) -> float:
    """Return a scale-aware two-rounding tolerance for h + (edited - h)."""

    scale = max(
        1.0,
        float(clean.detach().abs().max().cpu()),
        float(edited.detach().abs().max().cpu()),
    )
    return float(2.0 * np.finfo(np.float32).eps * scale)


def _evaluate_audit(
    config: dict[str, Any],
    original: dict[str, np.ndarray],
    evaluated: dict[str, np.ndarray],
    epsilons: tuple[float, ...],
    *,
    clean_replay_max_abs: float,
    semantic_reconstruction_errors: np.ndarray,
    direct_source_semantic_errors: np.ndarray,
    source_endpoint_errors: np.ndarray,
    source_endpoint_tolerances: np.ndarray,
    downstream_endpoint_relative_errors: np.ndarray,
) -> dict[str, Any]:
    direct = evaluated["direct_effect_fp32"]
    jvp = evaluated["exact_jvp_fp32"]
    central = evaluated["central_differences_fp32"]
    quadratic = evaluated["quadratic_taylor_fp32"]
    validation_index = epsilons.index(float(config.get("validation_epsilon", 0.125)))
    adjacent_index = epsilons.index(float(config.get("adjacent_epsilon", 0.0625)))
    relative = _row_relative_error(central[:, validation_index], jvp)
    adjacent = _row_relative_error(
        central[:, validation_index], central[:, adjacent_index], denominator=jvp
    )
    norm_ratio = np.linalg.norm(jvp, axis=1) / np.maximum(np.linalg.norm(direct, axis=1), 1e-8)
    derivative_gates = {
        "clean_replay": clean_replay_max_abs <= float(config.get("clean_replay_atol", 1e-6)),
        "median_jvp_central_relative_error": float(np.median(relative))
        <= float(config.get("median_relative_error_max", 0.05)),
        "p95_jvp_central_relative_error": float(np.quantile(relative, 0.95))
        <= float(config.get("p95_relative_error_max", 0.15)),
        "adjacent_epsilon_agreement": float(np.median(adjacent))
        <= float(config.get("adjacent_relative_error_max", 0.10)),
        "jvp_norm_outlier_rate": float(np.mean(norm_ratio > 10.0))
        <= float(config.get("jvp_norm_outlier_rate_max", 0.05)),
    }
    semantic_validation_mode = str(
        config.get("semantic_validation_mode", "downstream_endpoint_v1")
    )
    if semantic_validation_mode == "downstream_endpoint_v1":
        derivative_gates["semantic_dense_edit_reconstruction"] = bool(
            float(np.max(semantic_reconstruction_errors))
            <= float(config.get("semantic_reconstruction_atol", 1e-5))
        )
    elif semantic_validation_mode == "source_semantics_v2":
        derivative_gates["direct_source_semantics"] = bool(
            float(np.max(direct_source_semantic_errors))
            <= float(config.get("direct_source_semantic_atol", 0.0))
        )
        derivative_gates["directional_source_endpoint"] = bool(
            np.all(source_endpoint_errors <= source_endpoint_tolerances)
        )
    else:
        raise ValueError(f"unknown semantic_validation_mode: {semantic_validation_mode}")
    split = original["split_id"]
    feature = original["feature_id"]
    operation = original["operation_id"]
    heldout_feature = int(config.get("heldout_feature", 256))
    heldout_operation = OPERATIONS.index(str(config.get("heldout_operation", "resample")))
    eligible = (feature != heldout_feature) & (operation != heldout_operation)
    train = (split == 0) & eligible
    validation = (split == 1) & eligible
    primary = split == 2
    context = original["context"].astype(np.float32)
    intervention = original["intervention"].astype(np.float32)
    seeds = tuple(int(value) for value in config.get("predictor_seeds", [61, 67, 71]))
    seed_predictions: list[np.ndarray] = []
    checkpoint_dir = Path(
        str(config.get("checkpoint_dir", "artifacts/checkpoints/qwen_jvp_audit_v1"))
    )
    for seed in seeds:
        model = NeuralInterventionJEPA.fit(
            context[train],
            intervention[train],
            direct[train],
            (context[validation], intervention[validation], direct[validation]),
            hidden_dim=int(config.get("hidden_dim", 64)),
            meta_dim=int(config.get("meta_dim", 24)),
            steps=int(config.get("steps", 1000)),
            learning_rate=float(config.get("learning_rate", 0.003)),
            seed=seed,
            device=str(config.get("training_device", "cpu")),
        )
        checkpoint = checkpoint_dir / f"seed-{seed}.npz"
        model.save(checkpoint)
        restored = NeuralInterventionJEPA.load(checkpoint)
        seed_predictions.append(restored.predict(context[primary], intervention[primary]))
    linear = RidgeInterventionPredictor.fit(
        context[train], intervention[train], direct[train], mode="linear", ridge=1e-2
    )
    bilinear = RidgeInterventionPredictor.fit(
        context[train], intervention[train], direct[train], mode="bilinear", ridge=1e-2
    )
    mlp = TinyMLPInterventionPredictor.fit(
        context[train],
        intervention[train],
        direct[train],
        hidden_dim=int(config.get("mlp_hidden_dim", 64)),
        steps=int(config.get("mlp_steps", 600)),
        learning_rate=float(config.get("mlp_learning_rate", 0.005)),
        seed=seeds[0],
    )
    train_mean = direct[train].mean(axis=0)
    ensemble = np.mean(seed_predictions, axis=0)
    raw_predictions = {
        "conditional_bottleneck": ensemble,
        "no_change": np.zeros_like(direct[primary]),
        "mean_effect": np.broadcast_to(train_mean, direct[primary].shape),
        "linear": linear.predict(context[primary], intervention[primary]),
        "bilinear": bilinear.predict(context[primary], intervention[primary]),
        "mlp": mlp.predict(context[primary], intervention[primary]),
        "nearest_neighbor": _nearest_neighbor(
            context[train],
            intervention[train],
            direct[train],
            context[primary],
            intervention[primary],
        ),
        "exact_jvp": jvp[primary],
        "quadratic_taylor": quadratic[primary],
        "bf16_one_sided_secant": original["local_jacobian_effect"][primary],
    }
    deduplicated = deduplicated_effect_mask(
        original["prompt_id"][primary],
        original["layer"][primary],
        original["source_delta"][primary],
        direct[primary],
    )
    scores = {
        "raw": {
            name: _scores(prediction, direct[primary])
            for name, prediction in raw_predictions.items()
        },
        "deduplicated": {
            name: _scores(prediction[deduplicated], direct[primary][deduplicated])
            for name, prediction in raw_predictions.items()
        },
    }
    operation_nonlinearity = {}
    for operation_id, name in enumerate(OPERATIONS):
        mask = primary & (operation == operation_id)
        operation_nonlinearity[name] = {
            "examples": int(mask.sum()),
            "jvp_normalized_mse": _normalized_mse(jvp[mask], direct[mask]),
            "quadratic_normalized_mse": _normalized_mse(quadratic[mask], direct[mask]),
        }
    nonlinearity = {
        "raw_jvp_normalized_mse": _normalized_mse(jvp[primary], direct[primary]),
        "deduplicated_jvp_normalized_mse": _normalized_mse(
            jvp[primary][deduplicated], direct[primary][deduplicated]
        ),
        "raw_quadratic_normalized_mse": _normalized_mse(
            quadratic[primary], direct[primary]
        ),
        "by_operation": operation_nonlinearity,
    }
    bf16_fp32 = {
        "normalized_mse": _normalized_mse(original["target_effect"], direct),
        "correlation": effect_correlation(original["target_effect"], direct),
        "mean_row_relative_error": float(
            np.mean(_row_relative_error(original["target_effect"], direct))
        ),
    }
    nonlinear_operation_count = sum(
        row["jvp_normalized_mse"] >= float(config.get("nonlinear_fraction_min", 0.10))
        for row in operation_nonlinearity.values()
    )
    nonlinearity["finite_amplitude_nonlinearity_pass"] = bool(
        nonlinearity["raw_jvp_normalized_mse"]
        >= float(config.get("nonlinear_fraction_min", 0.10))
        and nonlinearity["deduplicated_jvp_normalized_mse"]
        >= float(config.get("nonlinear_fraction_min", 0.10))
        and nonlinearity["raw_quadratic_normalized_mse"]
        >= float(config.get("quadratic_residual_min", 0.05))
        and nonlinear_operation_count >= int(config.get("nonlinear_operations_min", 3))
    )
    decisions = []
    for seed, prediction in zip(seeds, seed_predictions, strict=True):
        raw = _scores(prediction, direct[primary])
        dedup = _scores(prediction[deduplicated], direct[primary][deduplicated])
        decisions.append(
            {
                "seed": seed,
                "raw": raw,
                "deduplicated": dedup,
                "beats_exact_jvp_10pct": raw["mse"]
                <= 0.9 * scores["raw"]["exact_jvp"]["mse"],
                "beats_quadratic_10pct": raw["mse"]
                <= 0.9 * scores["raw"]["quadratic_taylor"]["mse"],
                "deduplicated_beats_exact_jvp_10pct": dedup["mse"]
                <= 0.9 * scores["deduplicated"]["exact_jvp"]["mse"],
                "correlation_gate": raw["correlation"]
                >= float(config.get("correlation_min", 0.60)),
            }
        )
    predictor_passing_seeds = sum(
        all(
            bool(row[key])
            for key in (
                "beats_exact_jvp_10pct",
                "beats_quadratic_10pct",
                "deduplicated_beats_exact_jvp_10pct",
                "correlation_gate",
            )
        )
        for row in decisions
    )
    audit_passed = bool(all(derivative_gates.values()))
    h_llm_01_retained = bool(
        audit_passed
        and nonlinearity["finite_amplitude_nonlinearity_pass"]
        and predictor_passing_seeds >= int(config.get("required_passing_seeds", 2))
    )
    disposition = _hypothesis_disposition(audit_passed, h_llm_01_retained)
    return {
        "audit_passed": audit_passed,
        "h_llm_01_retained": h_llm_01_retained,
        "prior_h_llm_01_disposition": disposition,
        "derivative_gates": derivative_gates,
        "derivative_diagnostics": {
            "clean_replay_max_abs": clean_replay_max_abs,
            "semantic_dense_edit_max_abs": float(np.max(semantic_reconstruction_errors)),
            "semantic_validation_mode": semantic_validation_mode,
            "direct_source_semantic_max_abs": float(
                np.max(direct_source_semantic_errors)
            ),
            "source_direction_endpoint_max_abs": float(
                np.max(source_endpoint_errors)
            ),
            "source_direction_endpoint_max_tolerance": float(
                np.max(source_endpoint_tolerances)
            ),
            "source_direction_endpoint_tolerance_violations": int(
                np.sum(source_endpoint_errors > source_endpoint_tolerances)
            ),
            "downstream_endpoint_median_relative_error": float(
                np.median(downstream_endpoint_relative_errors)
            ),
            "downstream_endpoint_p95_relative_error": float(
                np.quantile(downstream_endpoint_relative_errors, 0.95)
            ),
            "median_jvp_central_relative_error": float(np.median(relative)),
            "p95_jvp_central_relative_error": float(np.quantile(relative, 0.95)),
            "median_adjacent_epsilon_relative_error": float(np.median(adjacent)),
            "jvp_norm_outlier_rate": float(np.mean(norm_ratio > 10.0)),
            "epsilon_sweep": list(epsilons),
            "central_error_by_epsilon": {
                str(epsilon): {
                    "median_relative_error": float(
                        np.median(_row_relative_error(central[:, index], jvp))
                    ),
                    "p95_relative_error": float(
                        np.quantile(_row_relative_error(central[:, index], jvp), 0.95)
                    ),
                }
                for index, epsilon in enumerate(epsilons)
            },
        },
        "primary_examples_raw": int(primary.sum()),
        "primary_examples_deduplicated": int(deduplicated.sum()),
        "duplicate_primary_examples": int(primary.sum() - deduplicated.sum()),
        "scores": scores,
        "nonlinearity": nonlinearity,
        "bf16_fp32_direct_effect_agreement": bf16_fp32,
        "predictor_decisions": decisions,
        "predictor_passing_seeds": predictor_passing_seeds,
        "required_passing_seeds": int(config.get("required_passing_seeds", 2)),
        "fp32_top_token_changes": int(
            np.sum(
                evaluated["fp32_clean_top_token"]
                != evaluated["fp32_intervened_top_token"]
            )
        ),
    }


def _hypothesis_disposition(audit_passed: bool, retained: bool) -> str:
    if not audit_passed:
        return "UNRESOLVED_NUMERICAL_REJECT"
    return "RETAINED" if retained else "WITHDRAWN"


def deduplicated_effect_mask(
    prompt_ids: np.ndarray,
    layers: np.ndarray,
    source_delta: np.ndarray,
    direct_effect: np.ndarray,
) -> np.ndarray:
    """Keep the first exact semantic duplicate without repairing the frozen split."""

    keep = np.zeros(len(prompt_ids), dtype=bool)
    seen: set[bytes] = set()
    for index in range(len(prompt_ids)):
        digest = hashlib.sha256()
        digest.update(np.asarray([prompt_ids[index], layers[index]], dtype=np.int64).tobytes())
        digest.update(np.round(source_delta[index], decimals=6).astype(np.float32).tobytes())
        digest.update(np.round(direct_effect[index], decimals=6).astype(np.float32).tobytes())
        key = digest.digest()
        if key not in seen:
            seen.add(key)
            keep[index] = True
    return keep


def _row_relative_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    denominator: np.ndarray | None = None,
) -> np.ndarray:
    reference = target if denominator is None else denominator
    return np.linalg.norm(prediction - target, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-8
    )


def _normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2) / max(float(np.mean(target**2)), 1e-12))


def _scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "normalized_mse": _normalized_mse(prediction, target),
        "correlation": effect_correlation(prediction, target),
        "mean_prediction_l2": float(np.linalg.norm(prediction, axis=1).mean()),
        "mean_target_l2": float(np.linalg.norm(target, axis=1).mean()),
        "evidence_level": "Generalization",
    }


def _nearest_neighbor(
    train_context: np.ndarray,
    train_intervention: np.ndarray,
    train_target: np.ndarray,
    test_context: np.ndarray,
    test_intervention: np.ndarray,
) -> np.ndarray:
    train = np.concatenate([train_context, train_intervention], axis=1).astype(np.float64)
    test = np.concatenate([test_context, test_intervention], axis=1).astype(np.float64)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    distances = ((test[:, None] - train[None]) / scale[None, None]) ** 2
    indices = np.argmin(distances.sum(axis=-1), axis=1)
    return train_target[indices]


def _config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
