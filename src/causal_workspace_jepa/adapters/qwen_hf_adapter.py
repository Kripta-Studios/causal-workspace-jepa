"""Selected-site Hugging Face Qwen instrumentation.

The adapter deliberately uses module hooks instead of ``output_hidden_states`` so
experiments capture only preregistered sites.  Torch tensors are preserved when
``preserve_autograd`` is enabled; inference runs otherwise return detached NumPy
arrays suitable for bounded activation shards.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from causal_workspace_jepa.common.types import InterventionSpec, LLMRun, TokenBatch
from causal_workspace_jepa.hooks.names import transformer_site


class QwenHFAdapterUnavailable(RuntimeError):
    """Raised when the optional Hugging Face/Torch runtime is unavailable."""


@dataclass(frozen=True)
class QwenAdapterConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_length: int = 64
    local_files_only: bool = False
    preserve_autograd: bool = False
    token: bool | str | None = False


class QwenHFAdapter:
    """Instrument a dense Qwen3 causal LM through stable selected-layer hooks."""

    def __init__(
        self,
        config: QwenAdapterConfig | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise QwenHFAdapterUnavailable(
                "Hugging Face Qwen instrumentation requires torch and transformers"
            ) from exc

        self.config = config or QwenAdapterConfig()
        self._torch = torch
        self.device = torch.device(self.config.device)
        self.dtype = _torch_dtype(torch, self.config.dtype)
        if model is None:
            load_kwargs: dict[str, Any] = {
                "revision": self.config.revision,
                "local_files_only": self.config.local_files_only,
                "trust_remote_code": False,
                "dtype": self.dtype,
                "low_cpu_mem_usage": True,
                "token": self.config.token,
            }
            load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **load_kwargs)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
                token=self.config.token,
            )
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model = model.to(device=self.device)
        if not self.config.preserve_autograd:
            self.model.eval()
        self.tokenizer = tokenizer
        self.layers = _decoder_layers(self.model)
        self._donors: dict[tuple[str, str], Any] = {}
        self._means: dict[str, Any] = {}
        self._bases: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> "QwenHFAdapter":
        return cls(QwenAdapterConfig(**kwargs))

    def tokenize(self, prompts: Sequence[str]) -> TokenBatch:
        if not prompts:
            raise ValueError("at least one prompt is required")
        encoded = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens = tuple(
            tuple(str(token) for token in self.tokenizer.convert_ids_to_tokens(row.tolist()))
            for row in input_ids.detach().cpu()
        )
        return TokenBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompts=tuple(prompts),
            tokens=tokens,
        )

    def named_activation_points(self) -> Sequence[str]:
        sites: list[str] = []
        for layer in range(len(self.layers)):
            sites.extend(
                [
                    transformer_site(layer, "resid_pre"),
                    transformer_site(layer, "attn_out"),
                    transformer_site(layer, "mlp_out"),
                    transformer_site(layer, "resid_post"),
                ]
            )
        sites.append("logits")
        return tuple(sites)

    def register_donor(self, example_id: str, site: str, activation: Any) -> None:
        self._validate_site(site)
        self._donors[(example_id, site)] = _as_torch(self._torch, activation, self.device)

    def register_mean(self, site: str, mean: Any) -> None:
        self._validate_site(site)
        self._means[site] = _as_torch(self._torch, mean, self.device)

    def register_basis(self, site: str, basis: Any) -> None:
        self._validate_site(site)
        self._bases[site] = _as_torch(self._torch, basis, self.device)

    def forward_with_cache(self, batch: TokenBatch, sites: Sequence[str]) -> LLMRun:
        return self._forward(batch, sites, intervention=None)

    def forward_with_intervention(
        self,
        batch: TokenBatch,
        intervention: InterventionSpec,
        sites: Sequence[str],
    ) -> LLMRun:
        self._validate_site(intervention.site)
        return self._forward(batch, sites, intervention=intervention)

    def _forward(
        self,
        batch: TokenBatch,
        sites: Sequence[str],
        intervention: InterventionSpec | None,
    ) -> LLMRun:
        requested = set(sites)
        unknown = requested.difference(self.named_activation_points())
        if unknown:
            raise KeyError(f"unknown Qwen activation sites: {sorted(unknown)}")
        captured: dict[str, Any] = {}
        handles: list[Any] = []
        hook_sites = requested | ({intervention.site} if intervention is not None else set())

        for layer_index, layer in enumerate(self.layers):
            resid_pre = transformer_site(layer_index, "resid_pre")
            if resid_pre in hook_sites:
                handles.append(
                    layer.register_forward_pre_hook(
                        self._pre_hook(resid_pre, captured, requested, intervention),
                        with_kwargs=True,
                    )
                )
            for kind, module in (("attn_out", layer.self_attn), ("mlp_out", layer.mlp)):
                site = transformer_site(layer_index, kind)
                if site in hook_sites:
                    handles.append(
                        module.register_forward_hook(
                            self._forward_hook(site, captured, requested, intervention)
                        )
                    )
            resid_post = transformer_site(layer_index, "resid_post")
            if resid_post in hook_sites:
                handles.append(
                    layer.register_forward_hook(
                        self._forward_hook(resid_post, captured, requested, intervention)
                    )
                )

        context = nullcontext() if self.config.preserve_autograd else self._torch.inference_mode()
        try:
            with context:
                output = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                    logits_to_keep=0,
                )
                logits = output.logits
                if intervention is not None and intervention.site == "logits":
                    logits = self._apply_torch_intervention(logits, intervention)
                if "logits" in requested:
                    captured["logits"] = logits
        finally:
            for handle in handles:
                handle.remove()

        exported = {site: self._export(value) for site, value in captured.items()}
        return LLMRun(
            logits=self._export(logits),
            activations=exported,
            token_batch=batch,
            metadata=self._metadata(),
        )

    def _pre_hook(
        self,
        site: str,
        captured: dict[str, Any],
        requested: set[str],
        intervention: InterventionSpec | None,
    ) -> Any:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if args:
                value = args[0]
                rest = args[1:]
                updated = self._maybe_intervene(site, value, intervention)
                if site in requested:
                    captured[site] = updated
                return (updated, *rest), kwargs
            value = kwargs["hidden_states"]
            updated = self._maybe_intervene(site, value, intervention)
            if site in requested:
                captured[site] = updated
            updated_kwargs = dict(kwargs)
            updated_kwargs["hidden_states"] = updated
            return args, updated_kwargs

        return hook

    def _forward_hook(
        self,
        site: str,
        captured: dict[str, Any],
        requested: set[str],
        intervention: InterventionSpec | None,
    ) -> Any:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                value = output[0]
                updated = self._maybe_intervene(site, value, intervention)
                replaced = (updated, *output[1:])
            else:
                updated = self._maybe_intervene(site, output, intervention)
                replaced = updated
            if site in requested:
                captured[site] = updated
            return replaced

        return hook

    def _maybe_intervene(
        self,
        site: str,
        value: Any,
        intervention: InterventionSpec | None,
    ) -> Any:
        if intervention is None or intervention.site != site:
            return value
        return self._apply_torch_intervention(value, intervention)

    def _apply_torch_intervention(self, activation: Any, spec: InterventionSpec) -> Any:
        torch = self._torch
        result = activation.clone()
        target = _torch_select(result, spec)
        if spec.operation in {"zero", "suppress_module"}:
            updated = torch.zeros_like(target)
        elif spec.operation == "scale":
            updated = target * spec.magnitude
        elif spec.operation == "steer":
            updated = target + spec.magnitude
        elif spec.operation == "mean":
            if spec.site not in self._means:
                raise ValueError("mean intervention requires a registered training-split mean")
            updated = _broadcast_selected(self._means[spec.site], result, spec)
        elif spec.operation in {"patch", "replace_feature", "resample"}:
            if spec.donor_example_id is None:
                raise ValueError(f"{spec.operation} requires donor_example_id")
            key = (spec.donor_example_id, spec.site)
            if key not in self._donors:
                raise KeyError(f"unregistered donor {spec.donor_example_id!r} at {spec.site}")
            updated = _broadcast_selected(self._donors[key], result, spec)
        elif spec.operation == "project_out":
            if spec.site not in self._bases:
                raise ValueError("project_out requires a registered basis")
            updated = _project_out_torch(target, self._bases[spec.site], spec.magnitude)
        else:  # pragma: no cover - Literal and unit tests constrain operations
            raise ValueError(f"unsupported intervention operation: {spec.operation}")
        _torch_assign(result, spec, updated.to(device=result.device, dtype=result.dtype))
        return result

    def _validate_site(self, site: str) -> None:
        if site not in self.named_activation_points():
            raise KeyError(f"unknown Qwen activation site: {site}")

    def _export(self, value: Any) -> Any:
        if self.config.preserve_autograd:
            return value
        return value.detach().to(device="cpu", dtype=self._torch.float32).numpy()

    def _metadata(self) -> Mapping[str, Any]:
        config = self.model.config
        return {
            "model": self.config.model_name,
            "requested_revision": self.config.revision,
            "resolved_revision": getattr(config, "_commit_hash", None),
            "model_type": getattr(config, "model_type", None),
            "layers": len(self.layers),
            "hidden_size": int(config.hidden_size),
            "vocab_size": int(config.vocab_size),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "hidden_activations_available": True,
            "autograd_available": self.config.preserve_autograd,
        }


def _decoder_layers(model: Any) -> Sequence[Any]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise TypeError("unsupported Qwen model layout: decoder layers were not found")


def _torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _as_torch(torch: Any, value: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device=device)
    return torch.as_tensor(value, device=device)


def _torch_select(array: Any, spec: InterventionSpec) -> Any:
    if array.ndim == 3:
        positions = list(spec.positions) if spec.positions is not None else slice(None)
        features = list(spec.feature_ids) if spec.feature_ids is not None else slice(None)
        return array[:, positions, :][..., features]
    if array.ndim == 2:
        features = list(spec.feature_ids) if spec.feature_ids is not None else slice(None)
        return array[:, features]
    if spec.feature_ids is not None:
        return array[..., list(spec.feature_ids)]
    return array


def _torch_assign(array: Any, spec: InterventionSpec, value: Any) -> None:
    if array.ndim == 3:
        positions = list(spec.positions) if spec.positions is not None else slice(None)
        features = list(spec.feature_ids) if spec.feature_ids is not None else slice(None)
        view = array[:, positions, :].clone()
        view[..., features] = value
        array[:, positions, :] = view
        return
    if array.ndim == 2:
        features = list(spec.feature_ids) if spec.feature_ids is not None else slice(None)
        array[:, features] = value
        return
    if spec.feature_ids is not None:
        array[..., list(spec.feature_ids)] = value
    else:
        array[...] = value


def _broadcast_selected(source: Any, target: Any, spec: InterventionSpec) -> Any:
    selected = source.to(device=target.device, dtype=target.dtype)
    try:
        full = selected.expand_as(target)
    except RuntimeError:
        full = selected
    return _torch_select(full, spec).expand_as(_torch_select(target, spec))


def _project_out_torch(target: Any, basis: Any, magnitude: float) -> Any:
    flat = target.reshape(-1, target.shape[-1])
    for vector in basis.reshape(-1, basis.shape[-1]).to(flat):
        denominator = vector.square().sum().clamp_min(1e-12)
        flat = flat - magnitude * ((flat @ vector) / denominator).unsqueeze(-1) * vector
    return flat.reshape_as(target)
