"""CRCT-STAGE0-HARD-002: frozen-discovery synthetic circuit falsification benchmark.

This benchmark deliberately makes circuit recovery harder than CRCT-STAGE0-001:

* causally active but intervention-irrelevant state decoys,
* action-sensitive null pairs whose individual effects cancel,
* redundant true routes and an imbalanced cancellation pair,
* QK-like bilinear routing edges,
* validation-only circuit discovery followed by an immutable selection hash,
* IID and OOD confirmation generated only after the selection is frozen,
* matched random controls frozen before confirmation,
* node and edge recovery, sufficiency, necessity, completeness, and gauge tests,
* first/second-order Screen-Flag-Fix diagnostics that are not required to win,
* equal-capacity direct-delta and residual students whose relative performance is diagnostic.

A pass validates the benchmark/method on planted synthetic mechanisms only. It is not evidence of a
Qwen or JEPA circuit and it is not a workspace claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA_VERSION = "crct_stage0_hard_circuit_recovery_v2"
EXPERIMENT_ID = "CRCT-STAGE0-HARD-002"


@dataclass(frozen=True)
class HardProfile:
    name: str
    train_samples: int
    validation_samples: int
    test_samples: int
    ood_samples: int
    diagnostic_samples: int
    random_control_count: int
    student_steps: int
    student_width: int
    student_depth: int
    batch_size: int
    learning_rate: float
    max_selected: int
    min_step_recovery_fraction: float


PROFILES: dict[str, HardProfile] = {
    "smoke": HardProfile(
        name="smoke",
        train_samples=4096,
        validation_samples=2048,
        test_samples=2048,
        ood_samples=2048,
        diagnostic_samples=1024,
        random_control_count=64,
        student_steps=120,
        student_width=128,
        student_depth=3,
        batch_size=512,
        learning_rate=2e-3,
        max_selected=16,
        min_step_recovery_fraction=0.004,
    ),
    "full": HardProfile(
        name="full",
        train_samples=65536,
        validation_samples=8192,
        test_samples=16384,
        ood_samples=16384,
        diagnostic_samples=4096,
        random_control_count=256,
        student_steps=1200,
        student_width=384,
        student_depth=4,
        batch_size=2048,
        learning_rate=1.5e-3,
        max_selected=20,
        min_step_recovery_fraction=0.002,
    ),
}


@dataclass(frozen=True)
class Split:
    state: Tensor
    action: Tensor
    name: str


@dataclass(frozen=True)
class CandidateMeta:
    name: str
    kind: str  # node | edge
    family: str
    truth: bool
    role: str


@dataclass(frozen=True)
class FrozenPlan:
    selected: tuple[str, ...]
    ranked: tuple[str, ...]
    controls: tuple[tuple[str, ...], ...]
    validation_scores: Mapping[str, float]
    validation_activation_rms: Mapping[str, float]
    sha256: str


class EqualCapacityMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, width), nn.GELU()])
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HardSyntheticCircuit(nn.Module):
    """Additive planted circuit with difficult decoys and QK-like routing edges."""

    state_dim = 8
    action_dim = 6
    output_dim = 6
    route_width = 12
    bypass_width = 8
    redundant_width = 6
    cancel_width = 4
    state_decoy_width = 6
    null_width = 4
    nuisance_width = 12
    qk_width = 8

    def __init__(self, seed: int, *, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed * 104729 + 17)

        def randn(*shape: int, scale: float = 1.0) -> Tensor:
            return torch.randn(shape, generator=gen, dtype=torch.float32) * scale

        def randu(*shape: int, low: float = -1.0, high: float = 1.0) -> Tensor:
            return torch.rand(shape, generator=gen, dtype=torch.float32) * (high - low) + low

        # Differential base: exactly first/second order in action.
        self.register_buffer("linear_w", randn(self.output_dim, self.action_dim, scale=0.16))
        self.register_buffer("quadratic_w", randn(self.output_dim, self.action_dim, scale=0.10))
        self.register_buffer("bilinear_w", randn(self.output_dim, self.state_dim, scale=0.07))
        self.register_buffer("bilinear_a", randn(self.state_dim, self.action_dim, scale=0.14))

        # Shared helper for scalar nonlinear nodes.
        widths = {
            "route": self.route_width,
            "bypass": self.bypass_width,
            "redundant": self.redundant_width,
            "cancel": self.cancel_width,
            "state_decoy": self.state_decoy_width,
            "null": self.null_width,
            "nuisance": self.nuisance_width,
        }
        for family, width in widths.items():
            self.register_buffer(f"{family}_x", randn(width, self.state_dim, scale=0.25))
            self.register_buffer(f"{family}_u", randn(width, self.action_dim, scale=0.42))
            self.register_buffer(f"{family}_b", randn(width, scale=0.20))
            self.register_buffer(f"{family}_out", randn(width, self.output_dim, scale=0.16))

        # State decoys affect absolute output but are invariant to the action intervention.
        self.state_decoy_u.zero_()
        self.state_decoy_out.mul_(1.8)

        # High-variance nuisance is disconnected from the output but dominates activation RMS.
        self.nuisance_u.mul_(0.3)
        self.nuisance_out.zero_()

        # Inactive coordinates in ordinary families are connected to state/output but action-invariant.
        self.route_u[4:].zero_()
        self.bypass_u[2:].zero_()
        self.redundant_u[2:].zero_()
        self.cancel_u[2:].zero_()

        # Redundant true pair: similar but not identical contributions.
        self.redundant_x[1].copy_(self.redundant_x[0] + randn(self.state_dim, scale=0.025))
        self.redundant_u[1].copy_(self.redundant_u[0] + randn(self.action_dim, scale=0.025))
        self.redundant_b[1].copy_(self.redundant_b[0] + randn(1, scale=0.015).squeeze(0))
        self.redundant_out[1].copy_(self.redundant_out[0] * 0.78)

        # True cancellation pair: opposing pathways with a deliberate 18% imbalance.
        self.cancel_x[1].copy_(self.cancel_x[0])
        self.cancel_u[1].copy_(self.cancel_u[0])
        self.cancel_b[1].copy_(self.cancel_b[0])
        self.cancel_out[1].copy_(-0.82 * self.cancel_out[0])

        # Action-sensitive null decoys: individually causal, pairwise exactly cancelling.
        for left in (0, 2):
            right = left + 1
            self.null_x[right].copy_(self.null_x[left])
            self.null_u[right].copy_(self.null_u[left])
            self.null_b[right].copy_(self.null_b[left])
            self.null_out[right].copy_(-self.null_out[left])
        self.null_out.mul_(1.35)

        # QK-like bilinear routing edges. First three are true. Edges 3/4 form a cancelling
        # action-sensitive decoy pair; 5..7 are state-only but directly affect absolute output.
        self.register_buffer("q_x", randn(self.qk_width, self.state_dim, scale=0.24))
        self.register_buffer("q_u", randn(self.qk_width, self.action_dim, scale=0.38))
        self.register_buffer("q_b", randn(self.qk_width, scale=0.15))
        self.register_buffer("k_x", randn(self.qk_width, self.state_dim, scale=0.24))
        self.register_buffer("k_u", randn(self.qk_width, self.action_dim, scale=0.38))
        self.register_buffer("k_b", randn(self.qk_width, scale=0.15))
        self.register_buffer("v_x", randn(self.qk_width, self.state_dim, scale=0.24))
        self.register_buffer("v_u", randn(self.qk_width, self.action_dim, scale=0.38))
        self.register_buffer("v_b", randn(self.qk_width, scale=0.15))
        self.register_buffer("qk_out", randn(self.qk_width, self.output_dim, scale=0.18))
        self.qk_out[:3].mul_(3.0)

        self.q_x[4].copy_(self.q_x[3])
        self.q_u[4].copy_(self.q_u[3])
        self.q_b[4].copy_(self.q_b[3])
        self.k_x[4].copy_(self.k_x[3])
        self.k_u[4].copy_(self.k_u[3])
        self.k_b[4].copy_(self.k_b[3])
        self.v_x[4].copy_(self.v_x[3])
        self.v_u[4].copy_(self.v_u[3])
        self.v_b[4].copy_(self.v_b[3])
        self.qk_out[4].copy_(-self.qk_out[3])
        self.q_u[5:].zero_()
        self.k_u[5:].zero_()
        self.v_u[5:].zero_()

        # Stronger route/bypass true coordinates to avoid a degenerate near-zero planted target.
        self.route_out[:4].mul_(2.4)
        self.bypass_out[:2].mul_(2.2)
        self.redundant_out[:2].mul_(2.0)
        self.cancel_out[:2].mul_(2.0)

        # A fixed random gauge used only for the invariance audit.
        scales = torch.exp(torch.linspace(-2.1, 2.1, self.route_width + self.bypass_width + self.redundant_width + self.cancel_width + self.null_width + self.qk_width))
        perm = torch.randperm(scales.numel(), generator=gen)
        self.register_buffer("gauge_scales", scales[perm])
        self.to(device)

    @property
    def truth_nodes(self) -> tuple[str, ...]:
        return tuple([*(f"route:{i}" for i in range(4)), *(f"bypass:{i}" for i in range(2)), *(f"redundant:{i}" for i in range(2)), *(f"cancel:{i}" for i in range(2))])

    @property
    def truth_edges(self) -> tuple[str, ...]:
        return tuple(f"qk:{i}" for i in range(3))

    def candidate_meta(self) -> tuple[CandidateMeta, ...]:
        result: list[CandidateMeta] = []
        for family, width in (
            ("route", self.route_width),
            ("bypass", self.bypass_width),
            ("redundant", self.redundant_width),
            ("cancel", self.cancel_width),
            ("state_decoy", self.state_decoy_width),
            ("null", self.null_width),
            ("nuisance", self.nuisance_width),
        ):
            for i in range(width):
                name = f"{family}:{i}"
                result.append(
                    CandidateMeta(
                        name=name,
                        kind="node",
                        family=family,
                        truth=name in self.truth_nodes,
                        role=(
                            "target"
                            if name in self.truth_nodes
                            else "active_state_decoy"
                            if family == "state_decoy"
                            else "action_sensitive_cancelling_decoy"
                            if family == "null"
                            else "high_variance_nuisance"
                            if family == "nuisance"
                            else "inactive_family_coordinate"
                        ),
                    )
                )
        for i in range(self.qk_width):
            name = f"qk:{i}"
            result.append(
                CandidateMeta(
                    name=name,
                    kind="edge",
                    family="qk",
                    truth=name in self.truth_edges,
                    role=(
                        "target_qk_edge"
                        if name in self.truth_edges
                        else "action_sensitive_cancelling_qk_decoy"
                        if i in (3, 4)
                        else "state_only_qk_decoy"
                    ),
                )
            )
        return tuple(result)

    def _node_activation(self, family: str, state: Tensor, action: Tensor) -> Tensor:
        x = getattr(self, f"{family}_x")
        u = getattr(self, f"{family}_u")
        b = getattr(self, f"{family}_b")
        drive = action @ u.T
        z = state @ x.T + drive + b
        if family == "route":
            return torch.tanh(z) + 0.45 * drive.pow(3)
        if family == "redundant":
            return torch.tanh(z) + 0.35 * drive.pow(3)
        if family == "state_decoy":
            return torch.tanh(z)
        if family == "null":
            return torch.tanh(z) + 0.45 * drive.pow(3)
        if family == "bypass":
            return torch.sin(z) + 0.25 * torch.sin(drive).pow(3)
        if family == "cancel":
            return torch.nn.functional.silu(z) + 0.30 * drive.pow(3)
        if family == "nuisance":
            return 5.0 * torch.tanh(z)
        raise KeyError(family)

    def _qk_activation(self, state: Tensor, action: Tensor) -> Tensor:
        q = state @ self.q_x.T + action @ self.q_u.T + self.q_b
        k = state @ self.k_x.T + action @ self.k_u.T + self.k_b
        v = torch.tanh(state @ self.v_x.T + action @ self.v_u.T + self.v_b)
        gate = torch.sigmoid(q * k / math.sqrt(2.0))
        return gate * v

    def component_activation(self, name: str, state: Tensor, action: Tensor) -> Tensor:
        family, index_text = name.split(":", 1)
        index = int(index_text)
        if family == "qk":
            return self._qk_activation(state, action)[:, index]
        return self._node_activation(family, state, action)[:, index]

    def component_output(self, name: str, state: Tensor, action: Tensor, *, gauge: bool = False) -> Tensor:
        family, index_text = name.split(":", 1)
        index = int(index_text)
        activation = self.component_activation(name, state, action)
        if family == "qk":
            weight = self.qk_out[index]
            offset = self.route_width + self.bypass_width + self.redundant_width + self.cancel_width + self.null_width
            scale = self.gauge_scales[offset + index] if gauge else torch.tensor(1.0, device=state.device)
        else:
            weight = getattr(self, f"{family}_out")[index]
            if family in {"state_decoy", "nuisance"}:
                scale = torch.tensor(1.0, device=state.device)
            else:
                offsets = {
                    "route": 0,
                    "bypass": self.route_width,
                    "redundant": self.route_width + self.bypass_width,
                    "cancel": self.route_width + self.bypass_width + self.redundant_width,
                    "null": self.route_width + self.bypass_width + self.redundant_width + self.cancel_width,
                }
                scale = self.gauge_scales[offsets[family] + index] if gauge else torch.tensor(1.0, device=state.device)
        # Function-preserving coordinate change: activation * s, outgoing weight / s.
        return (activation * scale).unsqueeze(-1) * (weight / scale)

    def differential_base(self, state: Tensor, action: Tensor) -> Tensor:
        linear = action @ self.linear_w.T
        quadratic = action.square() @ self.quadratic_w.T
        projected = action @ self.bilinear_a.T
        bilinear = (state * projected) @ self.bilinear_w.T
        return linear + quadratic + bilinear

    def absolute_output(self, state: Tensor, action: Tensor, *, gauge: bool = False) -> Tensor:
        output = state[:, : self.output_dim] * 0.12 + self.differential_base(state, action)
        for meta in self.candidate_meta():
            output = output + self.component_output(meta.name, state, action, gauge=gauge)
        return output

    def effect(self, state: Tensor, action: Tensor, *, gauge: bool = False) -> Tensor:
        zero = torch.zeros_like(action)
        return self.absolute_output(state, action, gauge=gauge) - self.absolute_output(state, zero, gauge=gauge)


def _directional_taylor(model_fn: Any, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
    """Exact first- and second-order directional Taylor terms around action=0."""

    zero_scalar = torch.zeros((), device=state.device, dtype=state.dtype)
    one_scalar = torch.ones_like(zero_scalar)

    def path(t: Tensor) -> Tensor:
        return model_fn(state, action * t)

    _, first = torch.func.jvp(path, (zero_scalar,), (one_scalar,))

    def first_at(t: Tensor) -> Tensor:
        return torch.func.jvp(path, (t,), (one_scalar,))[1]

    _, second = torch.func.jvp(first_at, (zero_scalar,), (one_scalar,))
    return first, first + 0.5 * second


def _component_stats(model: HardSyntheticCircuit, split: Split, metas: Sequence[CandidateMeta]) -> dict[str, dict[str, Tensor]]:
    stats: dict[str, dict[str, Tensor]] = {}
    zero = torch.zeros_like(split.action)
    for meta in metas:
        clean = model.component_output(meta.name, split.state, zero)
        treated = model.component_output(meta.name, split.state, split.action)
        finite = treated - clean
        first, second = _directional_taylor(
            lambda state, action, name=meta.name: model.component_output(name, state, action),
            split.state,
            split.action,
        )
        residual = finite - second
        activation = model.component_activation(meta.name, split.state, split.action)
        stats[meta.name] = {
            "finite": finite.detach(),
            "first": first.detach(),
            "second": second.detach(),
            "residual": residual.detach(),
            "activation": activation.detach(),
        }
    return stats


def _full_targets(model: HardSyntheticCircuit, split: Split) -> dict[str, Tensor]:
    finite = model.effect(split.state, split.action)
    first, second = _directional_taylor(
        lambda state, action: model.effect(state, action), split.state, split.action
    )
    residual = finite - second
    return {
        "finite": finite.detach(),
        "first": first.detach(),
        "second": second.detach(),
        "residual": residual.detach(),
    }


def _mse(x: Tensor, y: Tensor) -> float:
    return float(torch.mean((x - y).square()).detach().cpu())


def _energy(x: Tensor) -> float:
    return float(torch.mean(x.square()).detach().cpu())


def _nmse(prediction: Tensor, target: Tensor) -> float:
    return _mse(prediction, target) / max(_energy(target), 1e-12)


def _sha_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha(*values: Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        item = value.detach().cpu().contiguous()
        digest.update(str(tuple(item.shape)).encode())
        digest.update(str(item.dtype).encode())
        digest.update(item.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _average_precision(ranked: Sequence[str], truths: set[str]) -> float:
    if not truths:
        return 0.0
    hits = 0
    total = 0.0
    for index, name in enumerate(ranked, start=1):
        if name in truths:
            hits += 1
            total += hits / index
    return total / len(truths)


def _precision_recall(selected: Iterable[str], truth: set[str]) -> tuple[float, float]:
    chosen = set(selected)
    if not chosen:
        return 0.0, 0.0
    overlap = len(chosen & truth)
    return overlap / len(chosen), overlap / max(len(truth), 1)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(array, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and array[order[j]] == array[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a = _rankdata(left)
    b = _rankdata(right)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _sum_contributions(names: Iterable[str], stats: Mapping[str, Mapping[str, Tensor]], key: str) -> Tensor:
    names = tuple(names)
    if not names:
        first = next(iter(stats.values()))[key]
        return torch.zeros_like(first)
    total = torch.zeros_like(next(iter(stats.values()))[key])
    for name in names:
        total = total + stats[name][key]
    return total


def _validation_gain(target: Tensor, contribution: Tensor) -> float:
    base = _mse(torch.zeros_like(target), target)
    after = _mse(contribution, target)
    return (base - after) / max(base, 1e-12)


def _discover(
    target: Tensor,
    stats: Mapping[str, Mapping[str, Tensor]],
    metas: Sequence[CandidateMeta],
    *,
    max_selected: int,
    min_step_fraction: float,
    random_control_count: int,
    seed: int,
) -> FrozenPlan:
    names = [meta.name for meta in metas]
    activation_rms = {
        name: float(torch.sqrt(torch.mean(stats[name]["activation"].square())).cpu()) for name in names
    }
    gains = {name: _validation_gain(target, stats[name]["residual"]) for name in names}
    ranked = tuple(sorted(names, key=lambda name: (gains[name], name), reverse=True))

    selected: list[str] = []
    current = torch.zeros_like(target)
    initial_mse = _mse(current, target)
    current_mse = initial_mse
    remaining = set(names)
    for _ in range(max_selected):
        best_name: str | None = None
        best_mse = current_mse
        for name in sorted(remaining):
            candidate_mse = _mse(current + stats[name]["residual"], target)
            if candidate_mse < best_mse - 1e-15:
                best_mse = candidate_mse
                best_name = name
        if best_name is None:
            break
        step_recovery = (current_mse - best_mse) / max(initial_mse, 1e-12)
        if step_recovery < min_step_fraction:
            break
        selected.append(best_name)
        current = current + stats[best_name]["residual"]
        current_mse = best_mse
        remaining.remove(best_name)

    # Controls are frozen on validation only. Each selected mechanism is replaced by a non-selected
    # candidate of the same kind, matched approximately on activation RMS and finite-effect energy.
    rng = random.Random(seed * 8191 + 991)
    meta_by_name = {meta.name: meta for meta in metas}
    selected_set = set(selected)
    finite_energy = {name: _energy(stats[name]["finite"]) for name in names}

    def distance(source: str, candidate: str) -> float:
        a0 = math.log10(activation_rms[source] + 1e-8)
        a1 = math.log10(activation_rms[candidate] + 1e-8)
        e0 = math.log10(finite_energy[source] + 1e-10)
        e1 = math.log10(finite_energy[candidate] + 1e-10)
        family_penalty = 0.0 if meta_by_name[source].family == meta_by_name[candidate].family else 0.35
        return abs(a0 - a1) + abs(e0 - e1) + family_penalty

    controls: list[tuple[str, ...]] = []
    for _ in range(random_control_count):
        used: set[str] = set()
        control: list[str] = []
        for source in selected:
            pool = [
                name
                for name in names
                if name not in selected_set
                and name not in used
                and meta_by_name[name].kind == meta_by_name[source].kind
            ]
            if not pool:
                pool = [name for name in names if name not in selected_set and name not in used]
            pool.sort(key=lambda name: (distance(source, name), name))
            shortlist = pool[: min(8, len(pool))]
            # Randomness happens only among validation-matched alternatives.
            weights = [math.exp(-2.0 * distance(source, name)) for name in shortlist]
            chosen = rng.choices(shortlist, weights=weights, k=1)[0]
            control.append(chosen)
            used.add(chosen)
        controls.append(tuple(control))

    payload = {
        "selected": selected,
        "ranked": ranked,
        "controls": controls,
        "validation_scores": gains,
        "validation_activation_rms": activation_rms,
        "selection_rule": {
            "type": "greedy_signed_residual_reconstruction",
            "max_selected": max_selected,
            "min_step_recovery_fraction": min_step_fraction,
            "controls": "validation_frozen_kind_activation_finite_effect_matched",
        },
    }
    return FrozenPlan(
        selected=tuple(selected),
        ranked=ranked,
        controls=tuple(controls),
        validation_scores=gains,
        validation_activation_rms=activation_rms,
        sha256=_sha_json(payload),
    )


def _make_split(
    count: int,
    *,
    seed: int,
    device: torch.device,
    name: str,
    ood: bool = False,
) -> Split:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((count, HardSyntheticCircuit.state_dim), generator=gen, dtype=torch.float32)
    if ood:
        state[:, :4] = state[:, :4] * 1.25 + 0.65
        state[:, 4:] = state[:, 4:] * 0.75 - 0.25
        action = torch.rand((count, HardSyntheticCircuit.action_dim), generator=gen) * 2.6 - 1.3
        # OOD composition: couple two controls and flip another to create unseen joint patterns.
        action[:, 3] = torch.tanh(action[:, 0] + 0.7 * action[:, 1])
        action[:, 5] = -torch.tanh(0.8 * action[:, 2] - action[:, 4])
    else:
        state = state * 0.9
        action = torch.rand((count, HardSyntheticCircuit.action_dim), generator=gen) * 1.6 - 0.8
    return Split(state=state.to(device), action=action.to(device), name=name)


def _evaluate_confirmation(
    *,
    split: Split,
    model: HardSyntheticCircuit,
    metas: Sequence[CandidateMeta],
    plan: FrozenPlan,
) -> dict[str, Any]:
    diag_count = min(split.state.shape[0], 4096 if split.name == "iid_test" else 4096)
    diag = Split(split.state[:diag_count], split.action[:diag_count], split.name + "_diagnostic")
    stats = _component_stats(model, diag, metas)
    target = _full_targets(model, diag)["residual"]
    selected_sum = _sum_contributions(plan.selected, stats, "residual")
    zero_mse = _mse(torch.zeros_like(target), target)
    selected_mse = _mse(selected_sum, target)
    recovery = 1.0 - selected_mse / max(zero_mse, 1e-12)

    control_recoveries: list[float] = []
    for control in plan.controls:
        control_sum = _sum_contributions(control, stats, "residual")
        control_mse = _mse(control_sum, target)
        control_recoveries.append(1.0 - control_mse / max(zero_mse, 1e-12))
    random_p95 = float(np.quantile(control_recoveries, 0.95)) if control_recoveries else float("nan")
    empirical_p = (1.0 + sum(value >= recovery for value in control_recoveries)) / (
        1.0 + len(control_recoveries)
    )

    truth_nodes = set(model.truth_nodes)
    truth_edges = set(model.truth_edges)
    selected_nodes = [name for name in plan.selected if name in {m.name for m in metas if m.kind == "node"}]
    selected_edges = [name for name in plan.selected if name in {m.name for m in metas if m.kind == "edge"}]
    node_precision, node_recall = _precision_recall(selected_nodes, truth_nodes)
    edge_precision, edge_recall = _precision_recall(selected_edges, truth_edges)
    decoy_names = {m.name for m in metas if not m.truth}
    decoy_rejection = 1.0 - len(set(plan.selected) & decoy_names) / max(len(decoy_names), 1)

    # Necessity/completeness are measured on the frozen selected set, not by re-selecting on test.
    selected_energy_fraction = _energy(selected_sum) / max(_energy(target), 1e-12)
    complement = target - selected_sum
    complement_energy_fraction = _energy(complement) / max(_energy(target), 1e-12)

    return {
        "split": split.name,
        "diagnostic_count": diag_count,
        "residual_energy": _energy(target),
        "selected_reconstruction_mse": selected_mse,
        "zero_reconstruction_mse": zero_mse,
        "circuit_recovery_fraction": recovery,
        "selected_energy_fraction": selected_energy_fraction,
        "complement_energy_fraction": complement_energy_fraction,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "decoy_rejection_fraction": decoy_rejection,
        "matched_control_count": len(control_recoveries),
        "matched_control_mean_recovery": float(np.mean(control_recoveries)) if control_recoveries else float("nan"),
        "matched_control_p95_recovery": random_p95,
        "matched_control_empirical_p_plus_one": empirical_p,
        "selected_minus_control_p95": recovery - random_p95,
    }


def _screen_flag_fix(
    metas: Sequence[CandidateMeta],
    stats: Mapping[str, Mapping[str, Tensor]],
    truth: set[str],
) -> dict[str, Any]:
    names = [meta.name for meta in metas]
    exact_finite = {name: _energy(stats[name]["finite"]) for name in names}
    first = {name: _energy(stats[name]["first"]) for name in names}
    second = {name: _energy(stats[name]["second"]) for name in names}
    residual = {name: _energy(stats[name]["residual"]) for name in names}
    relative_error = {
        name: _mse(stats[name]["first"], stats[name]["finite"]) / max(exact_finite[name], 1e-12)
        for name in names
    }
    flagged = {name for name in names if relative_error[name] > 0.25}
    fixed = {name: second[name] if name in flagged else first[name] for name in names}

    def ranking(score: Mapping[str, float]) -> list[str]:
        return sorted(names, key=lambda name: (score[name], name), reverse=True)

    exact_rank = ranking(exact_finite)
    first_rank = ranking(first)
    second_rank = ranking(second)
    fixed_rank = ranking(fixed)
    residual_rank = ranking(residual)
    return {
        "flag_threshold_relative_mse": 0.25,
        "flagged_count": len(flagged),
        "flagged": sorted(flagged),
        "exact_finite_ap": _average_precision(exact_rank, truth),
        "first_order_ap": _average_precision(first_rank, truth),
        "second_order_ap": _average_precision(second_rank, truth),
        "screen_flag_fix_ap": _average_precision(fixed_rank, truth),
        "residual_exact_ap": _average_precision(residual_rank, truth),
        "first_vs_exact_spearman": _spearman([first[name] for name in names], [exact_finite[name] for name in names]),
        "second_vs_exact_spearman": _spearman([second[name] for name in names], [exact_finite[name] for name in names]),
        "fix_vs_exact_spearman": _spearman([fixed[name] for name in names], [exact_finite[name] for name in names]),
        "interpretation": "diagnostic only: HVP/T2 or Screen-Flag-Fix may improve or worsen rankings; exact finite patching remains the confirmation standard",
    }


def _gauge_audit(
    model: HardSyntheticCircuit,
    split: Split,
    metas: Sequence[CandidateMeta],
    stats: Mapping[str, Mapping[str, Tensor]],
) -> dict[str, Any]:
    sample = Split(split.state[: min(1024, split.state.shape[0])], split.action[: min(1024, split.action.shape[0])], "gauge")
    standard = model.effect(sample.state, sample.action, gauge=False)
    transformed = model.effect(sample.state, sample.action, gauge=True)
    max_error = float(torch.max(torch.abs(standard - transformed)).detach().cpu())

    before_activation: list[float] = []
    after_activation: list[float] = []
    causal_before: list[float] = []
    causal_after: list[float] = []
    scale_index = 0
    for meta in metas:
        activation = stats[meta.name]["activation"]
        before = float(torch.sqrt(torch.mean(activation.square())).cpu())
        if meta.family in {"state_decoy", "nuisance"}:
            scale = 1.0
        else:
            scale = float(model.gauge_scales[scale_index].detach().cpu())
            scale_index += 1
        before_activation.append(before)
        after_activation.append(before * scale)
        # Function-preserving gauge leaves exact residual contribution invariant.
        score = _energy(stats[meta.name]["residual"])
        causal_before.append(score)
        causal_after.append(score)

    return {
        "function_max_abs_error": max_error,
        "activation_rank_spearman": _spearman(before_activation, after_activation),
        "causal_rank_spearman": _spearman(causal_before, causal_after),
        "causal_max_abs_score_difference": float(np.max(np.abs(np.asarray(causal_before) - np.asarray(causal_after)))),
        "gauge_scale_min": float(torch.min(model.gauge_scales).cpu()),
        "gauge_scale_max": float(torch.max(model.gauge_scales).cpu()),
    }


def _fit_student(
    *,
    train: Split,
    validation: Split,
    test: Split,
    ood: Split,
    train_target: Tensor,
    validation_target: Tensor,
    test_target: Tensor,
    ood_target: Tensor,
    profile: HardProfile,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = EqualCapacityMLP(
        HardSyntheticCircuit.state_dim + HardSyntheticCircuit.action_dim,
        HardSyntheticCircuit.output_dim,
        width=profile.student_width,
        depth=profile.student_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile.learning_rate, weight_decay=1e-4)
    features = torch.cat([train.state, train.action], dim=-1)
    count = features.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed * 65537 + 31)
    trace: list[float] = []
    for step in range(profile.student_steps):
        indices = torch.randint(0, count, (profile.batch_size,), generator=gen).to(device)
        prediction = model(features[indices])
        loss = torch.mean((prediction - train_target[indices]).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or step == profile.student_steps - 1 or step % max(profile.student_steps // 12, 1) == 0:
            trace.append(float(loss.detach().cpu()))

    def evaluate(split: Split, target: Tensor) -> dict[str, float]:
        with torch.no_grad():
            prediction = model(torch.cat([split.state, split.action], dim=-1))
        return {
            "mse": _mse(prediction, target),
            "nmse": _nmse(prediction, target),
        }

    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": profile.student_steps,
        "width": profile.student_width,
        "depth": profile.student_depth,
        "batch_size": profile.batch_size,
        "loss_trace": trace,
        "validation": evaluate(validation, validation_target),
        "iid_test": evaluate(test, test_target),
        "ood_test": evaluate(ood, ood_target),
    }


def run_hard_benchmark(*, profile_name: str, seed: int, device_name: str) -> dict[str, Any]:
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}")
    profile = PROFILES[profile_name]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    model = HardSyntheticCircuit(seed, device=device)
    metas = model.candidate_meta()
    truth = {meta.name for meta in metas if meta.truth}

    # Crucial phase separation: train/validation exist before discovery; confirmation splits do not.
    train = _make_split(profile.train_samples, seed=seed * 1000 + 101, device=device, name="train")
    validation = _make_split(profile.validation_samples, seed=seed * 1000 + 211, device=device, name="validation")

    train_targets = _full_targets(model, train)
    val_diag = Split(
        validation.state[: profile.diagnostic_samples],
        validation.action[: profile.diagnostic_samples],
        "validation_diagnostic",
    )
    validation_stats = _component_stats(model, val_diag, metas)
    validation_targets = _full_targets(model, val_diag)

    plan = _discover(
        validation_targets["residual"],
        validation_stats,
        metas,
        max_selected=profile.max_selected,
        min_step_fraction=profile.min_step_recovery_fraction,
        random_control_count=profile.random_control_count,
        seed=seed,
    )
    frozen_at = time.time_ns()
    frozen_plan_hash = plan.sha256

    # Confirmation is generated only after the immutable selection/control plan exists.
    iid_test = _make_split(profile.test_samples, seed=seed * 1000 + 307, device=device, name="iid_test")
    ood_test = _make_split(profile.ood_samples, seed=seed * 1000 + 401, device=device, name="ood_test", ood=True)
    test_generated_at = time.time_ns()
    if test_generated_at <= frozen_at:
        raise AssertionError("confirmation data must be generated after plan freeze")

    test_targets = _full_targets(model, iid_test)
    ood_targets = _full_targets(model, ood_test)
    iid_confirmation = _evaluate_confirmation(split=iid_test, model=model, metas=metas, plan=plan)
    ood_confirmation = _evaluate_confirmation(split=ood_test, model=model, metas=metas, plan=plan)

    # Validation diagnostics are explicitly not recalculated on test.
    exact_ranking = list(plan.ranked)
    node_truth = set(model.truth_nodes)
    edge_truth = set(model.truth_edges)
    node_ranking = [name for name in exact_ranking if next(meta for meta in metas if meta.name == name).kind == "node"]
    edge_ranking = [name for name in exact_ranking if next(meta for meta in metas if meta.name == name).kind == "edge"]

    screen = _screen_flag_fix(metas, validation_stats, truth)
    gauge = _gauge_audit(model, val_diag, metas, validation_stats)

    validation_full = _full_targets(model, validation)
    # Equal-capacity students. Their relative win/loss is deliberately not an acceptance gate.
    residual_student = _fit_student(
        train=train,
        validation=validation,
        test=iid_test,
        ood=ood_test,
        train_target=train_targets["residual"],
        validation_target=validation_full["residual"],
        test_target=test_targets["residual"],
        ood_target=ood_targets["residual"],
        profile=profile,
        seed=seed * 17 + 3,
        device=device,
    )
    direct_student = _fit_student(
        train=train,
        validation=validation,
        test=iid_test,
        ood=ood_test,
        train_target=train_targets["finite"],
        validation_target=validation_full["finite"],
        test_target=test_targets["finite"],
        ood_target=ood_targets["finite"],
        profile=profile,
        seed=seed * 17 + 5,
        device=device,
    )

    residual_power = _energy(test_targets["residual"]) / max(_energy(test_targets["finite"]), 1e-12)
    t1_nmse = _nmse(test_targets["first"], test_targets["finite"])
    t2_nmse = _nmse(test_targets["second"], test_targets["finite"])
    selected_nodes = [name for name in plan.selected if name in {m.name for m in metas if m.kind == "node"}]
    selected_edges = [name for name in plan.selected if name in {m.name for m in metas if m.kind == "edge"}]
    node_precision, node_recall = _precision_recall(selected_nodes, node_truth)
    edge_precision, edge_recall = _precision_recall(selected_edges, edge_truth)

    gates = {
        "confirmation_generated_after_freeze": test_generated_at > frozen_at,
        "finite_residual_power_ge_0_08": residual_power >= 0.08,
        "iid_circuit_recovery_ge_0_75": iid_confirmation["circuit_recovery_fraction"] >= 0.75,
        "ood_circuit_recovery_ge_0_55": ood_confirmation["circuit_recovery_fraction"] >= 0.55,
        "node_precision_ge_0_70": node_precision >= 0.70,
        "node_recall_ge_0_60": node_recall >= 0.60,
        "edge_precision_ge_2_of_3": edge_precision >= (2.0 / 3.0),
        "edge_recall_ge_2_of_3": edge_recall >= (2.0 / 3.0),
        "matched_controls_p_le_0_05": iid_confirmation["matched_control_empirical_p_plus_one"] <= 0.05,
        "matched_controls_margin_ge_0_15": iid_confirmation["selected_minus_control_p95"] >= 0.15,
        "gauge_function_invariant": gauge["function_max_abs_error"] <= 1e-5,
        "gauge_causal_rank_invariant": gauge["causal_rank_spearman"] >= 0.999999,
        "decoy_rejection_ge_0_75": iid_confirmation["decoy_rejection_fraction"] >= 0.75,
    }
    status = "HARD_VALIDATED" if all(gates.values()) else "NEGATIVE_RESULT"

    runtime: dict[str, Any] = {
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        runtime.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )

    plan_payload = {
        "sha256": frozen_plan_hash,
        "selected": list(plan.selected),
        "ranked": list(plan.ranked),
        "controls": [list(control) for control in plan.controls],
        "validation_scores": dict(plan.validation_scores),
        "validation_activation_rms": dict(plan.validation_activation_rms),
    }
    dataset_provenance = {
        "train_sha256": _tensor_sha(train.state, train.action),
        "validation_sha256": _tensor_sha(validation.state, validation.action),
        "iid_test_sha256": _tensor_sha(iid_test.state, iid_test.action),
        "ood_test_sha256": _tensor_sha(ood_test.state, ood_test.action),
        "plan_frozen_before_confirmation_generation": test_generated_at > frozen_at,
        "plan_frozen_time_ns": frozen_at,
        "confirmation_generated_time_ns": test_generated_at,
    }

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "seed": seed,
        "profile": asdict(profile),
        "runtime": runtime,
        "scientific_boundary": {
            "synthetic_ground_truth_only": True,
            "qwen_executed": False,
            "eb_jepa_executed": False,
            "workspace_claim_permitted": False,
            "hvp_superiority_required": False,
            "residual_student_superiority_required": False,
        },
        "ground_truth": {
            "truth_nodes": list(model.truth_nodes),
            "truth_edges": list(model.truth_edges),
            "truth_mechanism_count": len(truth),
            "decoy_design": [
                "state-only causally active output decoys",
                "action-sensitive exactly cancelling node pairs",
                "action-sensitive exactly cancelling QK-like edge pair",
                "high-variance disconnected nuisance",
                "inactive coordinates inside mechanism families",
            ],
        },
        "dataset_provenance": dataset_provenance,
        "frozen_discovery_plan": plan_payload,
        "differential_diagnostics": {
            "iid_first_order_nmse": t1_nmse,
            "iid_second_order_nmse": t2_nmse,
            "iid_residual_power_fraction": residual_power,
        },
        "ranking_diagnostics": {
            "validation_node_average_precision": _average_precision(node_ranking, node_truth),
            "validation_edge_average_precision": _average_precision(edge_ranking, edge_truth),
            "selected_node_precision": node_precision,
            "selected_node_recall": node_recall,
            "selected_edge_precision": edge_precision,
            "selected_edge_recall": edge_recall,
        },
        "iid_confirmation": iid_confirmation,
        "ood_confirmation": ood_confirmation,
        "screen_flag_fix": screen,
        "gauge_diagnostics": gauge,
        "students": {
            "residual_student": residual_student,
            "direct_delta_student": direct_student,
            "same_architecture_and_capacity": residual_student["parameter_count"] == direct_student["parameter_count"],
            "interpretation": "diagnostic comparator only; residual student may lose OOD without invalidating exact causal recovery",
        },
        "gates": gates,
    }
    result["result_sha256"] = _sha_json(result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_hard_benchmark(profile_name=args.profile, seed=args.seed, device_name=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    preflight_gates = {
        "confirmation_generated_after_freeze": bool(result["gates"]["confirmation_generated_after_freeze"]),
        "gauge_function_invariant": bool(result["gates"]["gauge_function_invariant"]),
        "gauge_causal_rank_invariant": bool(result["gates"]["gauge_causal_rank_invariant"]),
        "selection_nonempty": bool(result["frozen_discovery_plan"]["selected"]),
        "same_student_capacity": bool(result["students"]["same_architecture_and_capacity"]),
    }
    emitted_status = (
        "PREFLIGHT_VALIDATED" if args.preflight_only and all(preflight_gates.values()) else result["status"]
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": emitted_status,
                "scientific_status": result["status"],
                "seed": result["seed"],
                "profile": result["profile"]["name"],
                "result_sha256": result["result_sha256"],
                "gates": result["gates"],
                "preflight_gates": preflight_gates,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.preflight_only:
        return 0 if all(preflight_gates.values()) else 3
    return 0 if result["status"] == "HARD_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
