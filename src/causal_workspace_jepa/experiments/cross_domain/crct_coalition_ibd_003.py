"""CRCT-COALITION-IBD-003: interventional coalition recovery on an IBD plant.

IBD-001 remains smoke. IBD-002 remains PREREGISTERED_NOT_RUN and is not
executed. HARD-002 primary seeds are constructor-blocked.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from causal_workspace_jepa.interpretability.crct_coalition import (
    intervention_support_label,
    is_epsilon_sufficient,
    literal_recall,
    nmse,
    restoration_error,
)

SITE_NAMES = ("h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9")
FORWARD_SITES = ("h2", "h7", "h3", "h5", "h0", "h1", "h8")
C_FORWARD = frozenset({"h2", "h7", "h3", "h5"})
C_EQUIV = frozenset({"h4", "h9", "h3", "h5"})
LITERAL_PLANTED = ("h0", "h1", "h2", "h3", "h4", "h5", "h7", "h9")
DECOY_SITE = "h6"
NUISANCE_SITE = "h8"
CANCEL_PAIR = ("h0", "h1")

HARD002_PRIMARY_SEEDS = (1009, 2027, 4093)
IBD001_SEEDS = (11, 13, 17, 811, 823, 829)
IBD002_SEEDS = (21, 23, 29, 941, 947, 953)
FORBIDDEN_SEEDS = frozenset(HARD002_PRIMARY_SEEDS + IBD001_SEEDS + IBD002_SEEDS)
DEVELOPMENT_SEEDS = (31, 37, 41)
CONFIRMATION_SEEDS = (971, 977, 983)

FROZEN_THRESHOLDS = {
    "epsilon": 0.02,
    "necessity_delta_min": 0.05,
    "decoy_target_ablation_nmse_max": 1e-6,
    "decoy_activation_energy_min": 1.0,
    "cancellation_member_nmse_min": 1e-4,
    "random_control_count": 32,
    "random_control_plus_one_p_max": 0.05,
    "rms_control_count": 8,
    "rms_control_sufficient_max": 0,
    "specificity_ratio_min": 2.0,
    "gauge_function_mse_max": 1e-8,
    "gauge_causal_spearman_min": 0.99,
    "gauge_activation_spearman_max": 0.95,
    "uncompensated_energy_ratio_min": 10.0,
    "max_restore_size": 6,
}

GAUGE_SCALES = {"h2": 25.0, "h7": 0.04, "h3": 8.0, "h5": 0.125, "h0": 5.0}


def threshold_digest() -> str:
    payload = {"thresholds": FROZEN_THRESHOLDS, "gauge_scales": GAUGE_SCALES}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _energy(values: Tensor) -> float:
    return float(torch.mean(values.reshape(values.shape[0], -1).square()).item())


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> Tensor:
        order = sorted(range(len(values)), key=lambda i: (values[i], i))
        out = [0.0] * len(values)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return torch.tensor(out)

    a = ranks(left)
    b = ranks(right)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) == 0:
        return 0.0
    return float((a * b).sum().item() / denom.item())


class CoalitionRecoveryPlant(nn.Module):
    """IBD plant with distinct equivalent known paths and executed interventions."""

    state_dim = 8
    action_dim = 4
    out_dim = 6
    unknown_hidden = 8

    def __init__(self, seed: int, *, device: torch.device) -> None:
        super().__init__()
        if int(seed) in FORBIDDEN_SEEDS:
            raise ValueError(f"seed {seed} is frozen/forbidden and cannot be reused")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.seed = int(seed)
        self._device = device
        in_dim = self.state_dim + self.action_dim
        known_w = torch.randn(in_dim, self.out_dim, generator=generator)
        unknown_w1 = torch.randn(in_dim, self.unknown_hidden, generator=generator) * 0.55
        unknown_w2 = torch.randn(self.unknown_hidden, self.out_dim, generator=generator) * 0.55
        residual_w = torch.randn(in_dim, self.out_dim, generator=generator) * 0.5
        cancel_w = torch.randn(in_dim, self.out_dim, generator=generator) * 0.5
        decoy_w = torch.randn(in_dim, self.out_dim, generator=generator) * 3.0
        noise_w = torch.randn(in_dim, self.out_dim, generator=generator) * 0.03
        mask_a1 = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        mask_b1 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        q_raw = torch.randn(self.out_dim, self.out_dim, generator=generator)
        known_q, _ = torch.linalg.qr(q_raw)
        self.register_buffer("known_w", known_w.to(device))
        self.register_buffer("known_q", known_q.to(device))
        self.register_buffer("unknown_w1", unknown_w1.to(device))
        self.register_buffer("unknown_w2", unknown_w2.to(device))
        self.register_buffer("residual_w", residual_w.to(device))
        self.register_buffer("cancel_w", cancel_w.to(device))
        self.register_buffer("decoy_w", decoy_w.to(device))
        self.register_buffer("noise_w", noise_w.to(device))
        self.register_buffer("mask_a1", mask_a1.to(device))
        self.register_buffer("mask_a2", (1.0 - mask_a1).to(device))
        self.register_buffer("mask_b1", mask_b1.to(device))
        self.register_buffer("mask_b2", (1.0 - mask_b1).to(device))

    def _features(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.cat([state, action], dim=-1)

    def writes(
        self,
        state: Tensor,
        action: Tensor,
        *,
        gauge_scales: Mapping[str, float] | None = None,
        compensate: bool = True,
    ) -> dict[str, Tensor]:
        features = self._features(state, action)
        known = features @ self.known_w
        rotated = known @ self.known_q
        cancel = features @ self.cancel_w
        unknown = torch.tanh(features @ self.unknown_w1) @ self.unknown_w2
        residual = (features @ self.residual_w) * torch.tanh(features[:, : self.out_dim])
        payload = {
            "h0": cancel,
            "h1": -cancel,
            "h2": known * self.mask_a1,
            "h7": known * self.mask_a2,
            "h4": (rotated * self.mask_b1) @ self.known_q.T,
            "h9": (rotated * self.mask_b2) @ self.known_q.T,
            "h3": unknown,
            "h5": residual,
            "h6": features @ self.decoy_w,
            "h8": features @ self.noise_w,
        }
        out: dict[str, Tensor] = {}
        for name, tensor in payload.items():
            scale = float(gauge_scales.get(name, 1.0)) if gauge_scales else 1.0
            write_scale = (1.0 / scale) if (gauge_scales and compensate) else 1.0
            out[name] = tensor * scale * write_scale
        return out

    def activations(
        self,
        state: Tensor,
        action: Tensor,
        *,
        gauge_scales: Mapping[str, float] | None = None,
    ) -> dict[str, Tensor]:
        """Uncompensated site tensors (activation ranking / decoy energy)."""

        return self.writes(state, action, gauge_scales=gauge_scales, compensate=False)

    def target_output(
        self,
        state: Tensor,
        action: Tensor,
        *,
        only: Iterable[str] | None = None,
        ablate: Iterable[str] = (),
        gauge_scales: Mapping[str, float] | None = None,
        compensate: bool = True,
    ) -> Tensor:
        writes = self.writes(state, action, gauge_scales=gauge_scales, compensate=compensate)
        used = FORWARD_SITES if only is None else tuple(only)
        blocked = set(ablate)
        output = torch.zeros(state.shape[0], self.out_dim, device=state.device)
        for name in used:
            if name in blocked:
                continue
            output = output + writes[name]
        return output

    def spec_output(
        self,
        state: Tensor,
        action: Tensor,
        *,
        ablate: Iterable[str] = (),
    ) -> Tensor:
        writes = self.writes(state, action)
        blocked = set(ablate)
        output = torch.zeros(state.shape[0], self.out_dim, device=state.device)
        if DECOY_SITE not in blocked:
            output = output + writes[DECOY_SITE]
        if "h3" not in blocked:
            output = output + 0.2 * writes["h3"]
        return output


def _make_split(n: int, seed: int, device: torch.device, *, ood: bool = False) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    state = torch.randn(n, CoalitionRecoveryPlant.state_dim, generator=generator)
    action = torch.randn(n, CoalitionRecoveryPlant.action_dim, generator=generator)
    if ood:
        state = state * 1.7
        action = action * 1.7
    return state.to(device), action.to(device)


def restore_only(model: CoalitionRecoveryPlant, state: Tensor, action: Tensor, sites: Iterable[str]) -> Tensor:
    return model.target_output(state, action, only=tuple(sites))


def discover_minimal_restore_sets(
    model: CoalitionRecoveryPlant,
    state: Tensor,
    action: Tensor,
    *,
    epsilon: float,
    max_size: int,
) -> list[tuple[str, ...]]:
    full = model.target_output(state, action)
    found: list[tuple[str, ...]] = []
    for size in range(1, int(max_size) + 1):
        for combo in combinations(SITE_NAMES, size):
            restored = restore_only(model, state, action, combo)
            if not is_epsilon_sufficient(restoration_error(restored, full), epsilon=epsilon):
                continue
            combo_set = set(combo)
            if any(set(existing).issubset(combo_set) for existing in found):
                continue
            found.append(combo)
    return found


def _record(
    *,
    operation: str,
    site: str,
    magnitude: float,
    source: str,
    target: str,
    combination: Sequence[str],
    trained_operations: Sequence[str] = ("ablate", "restore_only", "substitute"),
    trained_sites: Sequence[str] = SITE_NAMES,
    magnitude_max: float = 1.0,
    trained_combination_size_max: int = 6,
) -> dict[str, Any]:
    label = intervention_support_label(
        operation=operation,
        site=site,
        magnitude=magnitude,
        combination=combination,
        trained_operations=trained_operations,
        trained_sites=trained_sites,
        magnitude_max=magnitude_max,
        trained_combination_size_max=trained_combination_size_max,
    )
    label["source"] = source
    label["target"] = target
    return label


def _activation_rms(acts: Mapping[str, Tensor], name: str) -> float:
    return float(torch.sqrt(torch.mean(acts[name].square())).item())


def _ablation_nmse(
    model: CoalitionRecoveryPlant,
    state: Tensor,
    action: Tensor,
    sites: Iterable[str],
    full: Tensor,
) -> float:
    ablated = model.target_output(state, action, ablate=tuple(sites))
    return restoration_error(ablated, full)


def _rms_replacements(
    selected: Sequence[str],
    acts: Mapping[str, Tensor],
    *,
    count: int,
) -> list[list[str]]:
    unused = [name for name in SITE_NAMES if name not in set(selected)]
    selected_key = tuple(sorted(selected))
    seen: set[tuple[str, ...]] = set()
    controls: list[list[str]] = []

    def consider(trial: list[str]) -> None:
        key = tuple(sorted(trial))
        if key == selected_key or key in seen:
            return
        seen.add(key)
        controls.append(trial)

    for offset in (0, 1):
        for index, source in enumerate(selected):
            ranked = sorted(
                unused,
                key=lambda name: (
                    abs(_activation_rms(acts, name) - _activation_rms(acts, source)),
                    name,
                ),
            )
            if offset >= len(ranked):
                continue
            trial = list(selected)
            trial[index] = ranked[offset]
            consider(trial)
            if len(controls) >= count:
                return controls
    return controls[:count]


def _magnitude_topk(acts: Mapping[str, Tensor], k: int) -> tuple[str, ...]:
    ranked = sorted(
        SITE_NAMES,
        key=lambda name: (_activation_rms(acts, name), name),
        reverse=True,
    )
    return tuple(ranked[:k])


def _freeze_controls(
    *,
    selected: Sequence[str],
    acts: Mapping[str, Tensor],
    seed: int,
    random_count: int,
    rms_count: int,
    exclude: Sequence[Sequence[str]] | None = None,
) -> dict[str, list[list[str]]]:
    selected_set = set(selected)
    size = len(selected)
    forbidden = {frozenset(selected_set)}
    for row in exclude or ():
        if len(tuple(row)) == size:
            forbidden.add(frozenset(row))
    rng = random.Random(int(seed) * 8191 + 991)
    pool = [combo for combo in combinations(SITE_NAMES, size) if frozenset(combo) not in forbidden]
    rng.shuffle(pool)
    decoy_row = [DECOY_SITE, *[name for name in selected if name != selected[-1]][: size - 1]]
    return {
        "random": [list(item) for item in pool[:random_count]],
        "rms_matched": _rms_replacements(selected, acts, count=rms_count),
        "cancel_pair": [list(CANCEL_PAIR)],
        "decoy_containing": [decoy_row],
    }


def _evaluate_controls(
    model: CoalitionRecoveryPlant,
    state: Tensor,
    action: Tensor,
    full: Tensor,
    controls: Mapping[str, list[list[str]]],
    *,
    epsilon: float,
) -> dict[str, Any]:
    def stats(rows: list[list[str]]) -> dict[str, Any]:
        errors = [restoration_error(restore_only(model, state, action, row), full) for row in rows]
        sufficient = [is_epsilon_sufficient(error, epsilon=epsilon) for error in errors]
        return {
            "count": len(rows),
            "unique_count": len({tuple(sorted(row)) for row in rows}),
            "errors": errors,
            "sufficient_count": int(sum(sufficient)),
            "mean_error": float(sum(errors) / max(len(errors), 1)),
        }

    random_stats = stats(controls["random"])
    plus_one_p = (random_stats["sufficient_count"] + 1) / (random_stats["count"] + 1)
    return {
        "random": {**random_stats, "plus_one_p": float(plus_one_p)},
        "rms_matched": stats(controls["rms_matched"]),
        "cancel_pair": stats(controls["cancel_pair"]),
        "decoy_containing": stats(controls["decoy_containing"]),
    }


def _specificity(
    model: CoalitionRecoveryPlant,
    state: Tensor,
    action: Tensor,
    selected: Sequence[str],
) -> dict[str, float]:
    y = model.target_output(state, action)
    spec = model.spec_output(state, action)
    y_wo_c = model.target_output(state, action, ablate=selected)
    spec_wo_c = model.spec_output(state, action, ablate=selected)
    y_wo_decoy = model.target_output(state, action, ablate=(DECOY_SITE,))
    spec_wo_decoy = model.spec_output(state, action, ablate=(DECOY_SITE,))
    c_on_target = restoration_error(y_wo_c, y)
    c_on_spec = restoration_error(spec_wo_c, spec)
    decoy_on_target = restoration_error(y_wo_decoy, y)
    decoy_on_spec = restoration_error(spec_wo_decoy, spec)
    return {
        "ablate_c_target_nmse": c_on_target,
        "ablate_c_spec_nmse": c_on_spec,
        "ablate_decoy_target_nmse": decoy_on_target,
        "ablate_decoy_spec_nmse": decoy_on_spec,
        "decoy_over_c_on_spec": decoy_on_spec / max(c_on_spec, 1e-12),
    }


def _gauge_audit(
    model: CoalitionRecoveryPlant,
    state: Tensor,
    action: Tensor,
) -> dict[str, Any]:
    full = model.target_output(state, action)
    acts = model.activations(state, action)
    gauged_full = model.target_output(state, action, gauge_scales=GAUGE_SCALES, compensate=True)
    gauged_acts = model.activations(state, action, gauge_scales=GAUGE_SCALES)
    uncompensated = model.target_output(
        state, action, gauge_scales={"h2": 25.0}, compensate=False
    )
    activation_before = [_activation_rms(acts, name) for name in SITE_NAMES]
    activation_after = [_activation_rms(gauged_acts, name) for name in SITE_NAMES]
    causal_before = [_ablation_nmse(model, state, action, (name,), full) for name in SITE_NAMES]
    gauged_causal = []
    for name in SITE_NAMES:
        ablated = model.target_output(
            state, action, ablate=(name,), gauge_scales=GAUGE_SCALES, compensate=True
        )
        gauged_causal.append(restoration_error(ablated, gauged_full))
    h2_base = _energy(acts["h2"])
    h2_uncomp = _energy(model.activations(state, action, gauge_scales={"h2": 25.0})["h2"])
    k = 4
    return {
        "function_mse": _energy(gauged_full - full),
        "activation_spearman": _spearman(activation_before, activation_after),
        "causal_spearman": _spearman(causal_before, gauged_causal),
        "uncompensated_function_mse": _energy(uncompensated - full),
        "uncompensated_h2_energy": h2_uncomp,
        "baseline_h2_energy": h2_base,
        "uncompensated_h2_ratio": h2_uncomp / max(h2_base, 1e-12),
        "magnitude_topk": list(_magnitude_topk(acts, k)),
        "gauged_magnitude_topk": list(_magnitude_topk(gauged_acts, k)),
    }


def _seed_gates(
    *,
    recovered: Sequence[str],
    iid_full: Tensor,
    iid_restored: Tensor,
    ood_full: Tensor,
    ood_restored: Tensor,
    minimality: Mapping[str, float],
    necessity: Mapping[str, float],
    discovered_sets: Sequence[Sequence[str]],
    cancel: Mapping[str, float],
    decoy_activation: float,
    control_eval: Mapping[str, Any],
    specificity: Mapping[str, float],
    gauge: Mapping[str, Any],
) -> dict[str, bool]:
    eps = FROZEN_THRESHOLDS["epsilon"]
    recovered_set = set(recovered)
    discovered_frozen = {frozenset(item) for item in discovered_sets}
    class_members = {C_FORWARD, C_EQUIV}
    return {
        "recovered_class_member": recovered_set in class_members,
        "both_class_members_found": class_members.issubset(discovered_frozen),
        "iid_sufficient": is_epsilon_sufficient(restoration_error(iid_restored, iid_full), epsilon=eps),
        "ood_sufficient": is_epsilon_sufficient(restoration_error(ood_restored, ood_full), epsilon=eps),
        "minimal": all(error > eps for error in minimality.values()),
        "necessary": all(
            delta >= FROZEN_THRESHOLDS["necessity_delta_min"] for delta in necessity.values()
        ),
        "cancel_members_visible": all(
            cancel[name] >= FROZEN_THRESHOLDS["cancellation_member_nmse_min"] for name in CANCEL_PAIR
        ),
        "decoy_activation_high": decoy_activation > FROZEN_THRESHOLDS["decoy_activation_energy_min"],
        "random_controls": control_eval["random"]["plus_one_p"]
        <= FROZEN_THRESHOLDS["random_control_plus_one_p_max"],
        "rms_controls": control_eval["rms_matched"]["sufficient_count"]
        <= FROZEN_THRESHOLDS["rms_control_sufficient_max"]
        and control_eval["rms_matched"]["count"] >= 2
        and control_eval["rms_matched"]["unique_count"]
        == control_eval["rms_matched"]["count"],
        "decoy_control_fails": control_eval["decoy_containing"]["sufficient_count"] == 0,
        "specificity_decoy_on_spec": specificity["decoy_over_c_on_spec"]
        >= FROZEN_THRESHOLDS["specificity_ratio_min"],
        "gauge_activation_moved": gauge["activation_spearman"]
        <= FROZEN_THRESHOLDS["gauge_activation_spearman_max"],
        "magnitude_selector_misses": set(gauge["magnitude_topk"]) != recovered_set,
        "gauged_magnitude_selector_misses": set(gauge["gauged_magnitude_topk"]) != recovered_set,
    }


def run_coalition_recovery(
    *,
    seed: int,
    stage: str,
    device_name: str = "cpu",
    samples: int = 128,
) -> dict[str, Any]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be development or confirmation")
    if stage == "development" and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("development must use the frozen development seeds")
    if stage == "confirmation" and seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation must use the frozen confirmation seeds")
    device = torch.device(device_name)
    model = CoalitionRecoveryPlant(seed, device=device)
    val_state, val_action = _make_split(samples, seed * 1000 + 211, device)
    discovered = discover_minimal_restore_sets(
        model,
        val_state,
        val_action,
        epsilon=FROZEN_THRESHOLDS["epsilon"],
        max_size=int(FROZEN_THRESHOLDS["max_restore_size"]),
    )
    size_min = min((len(item) for item in discovered), default=0)
    size_minimal = [item for item in discovered if len(item) == size_min]
    recovered = tuple(size_minimal[0]) if size_minimal else ()
    frozen_at = time.time_ns()
    val_acts = model.activations(val_state, val_action)
    if not recovered:
        return {
            "experiment_id": "CRCT-COALITION-IBD-003",
            "stage": stage,
            "seed": seed,
            "threshold_digest": threshold_digest(),
            "thresholds": dict(FROZEN_THRESHOLDS),
            "discovered_minimal_sets": [list(item) for item in discovered],
            "recovered_circuit": [],
            "status": "NEGATIVE_RESULT",
            "evidence_level": "None",
            "gates": {"recovered_class_member": False},
            "claim_boundary": (
                "synthetic IBD mechanism-recovery test only; not a Qwen, JEPA, workspace, "
                "Platonic, or planning claim; does not alter HARD-002, IBD-001, or IBD-002"
            ),
            "hard002_primary_seeds_reused": False,
            "ibd002_executed": False,
        }
    controls = _freeze_controls(
        selected=recovered,
        acts=val_acts,
        seed=seed,
        random_count=int(FROZEN_THRESHOLDS["random_control_count"]),
        rms_count=int(FROZEN_THRESHOLDS["rms_control_count"]),
        exclude=discovered,
    )
    iid_state, iid_action = _make_split(samples, seed * 1000 + 307, device)
    ood_state, ood_action = _make_split(samples, seed * 1000 + 401, device, ood=True)
    if time.time_ns() <= frozen_at:
        raise AssertionError("confirmation splits must be generated after discovery freeze")
    iid_full = model.target_output(iid_state, iid_action)
    ood_full = model.target_output(ood_state, ood_action)
    iid_restored = restore_only(model, iid_state, iid_action, recovered)
    ood_restored = restore_only(model, ood_state, ood_action, recovered)
    minimality = {
        name: restoration_error(
            restore_only(model, iid_state, iid_action, [item for item in recovered if item != name]),
            iid_full,
        )
        for name in recovered
    }
    with_error = restoration_error(iid_restored, iid_full)
    necessity = {
        name: _ablation_nmse(model, iid_state, iid_action, (name,), iid_full) - with_error
        for name in recovered
    }
    restore_equiv = restore_only(model, iid_state, iid_action, tuple(C_EQUIV))
    steered = model.target_output(
        iid_state, iid_action, gauge_scales={"h2": 8.0}, compensate=False
    )
    cancel = {
        "h0": _ablation_nmse(model, iid_state, iid_action, ("h0",), iid_full),
        "h1": _ablation_nmse(model, iid_state, iid_action, ("h1",), iid_full),
        "joint": _ablation_nmse(model, iid_state, iid_action, CANCEL_PAIR, iid_full),
    }
    signed = {
        name: {
            "mean": float(val_acts[name].mean().item()),
            "energy": _energy(val_acts[name]),
            "sign": float(torch.sign(val_acts[name].mean()).item()),
        }
        for name in SITE_NAMES
    }
    decoy_target_nmse = _ablation_nmse(model, iid_state, iid_action, (DECOY_SITE,), iid_full)
    decoy_activation = _energy(model.activations(iid_state, iid_action)[DECOY_SITE])
    control_eval = _evaluate_controls(
        model, iid_state, iid_action, iid_full, controls, epsilon=FROZEN_THRESHOLDS["epsilon"]
    )
    specificity = _specificity(model, iid_state, iid_action, recovered)
    gauge = _gauge_audit(model, iid_state, iid_action)
    literal = literal_recall(recovered, LITERAL_PLANTED)
    path_a = model.activations(iid_state, iid_action)["h2"] + model.activations(iid_state, iid_action)["h7"]
    path_b = model.activations(iid_state, iid_action)["h4"] + model.activations(iid_state, iid_action)["h9"]
    steer_error = restoration_error(steered, iid_full)
    interventions = [
        _record(
            operation="restore_only",
            site=recovered[0],
            magnitude=1.0,
            source="validation_discovery",
            target="iid_y",
            combination=recovered,
        ),
        _record(
            operation="restore_only",
            site="h4",
            magnitude=1.0,
            source="h4",
            target="iid_y",
            combination=tuple(C_EQUIV),
        ),
        _record(
            operation="ablate",
            site=DECOY_SITE,
            magnitude=1.0,
            source=DECOY_SITE,
            target="iid_y",
            combination=(DECOY_SITE,),
        ),
        _record(
            operation="steer",
            site="h2",
            magnitude=8.0,
            source="h2",
            target="iid_y",
            combination=("h2",),
            trained_operations=("ablate", "restore_only"),
            magnitude_max=1.0,
            trained_combination_size_max=6,
        ),
    ]
    interventions[-1]["executed_nmse"] = steer_error
    gates = _seed_gates(
        recovered=recovered,
        iid_full=iid_full,
        iid_restored=iid_restored,
        ood_full=ood_full,
        ood_restored=ood_restored,
        minimality=minimality,
        necessity=necessity,
        discovered_sets=discovered,
        cancel=cancel,
        decoy_activation=decoy_activation,
        control_eval=control_eval,
        specificity=specificity,
        gauge=gauge,
    )
    passed = all(gates.values())
    return {
        "experiment_id": "CRCT-COALITION-IBD-003",
        "stage": stage,
        "seed": seed,
        "threshold_digest": threshold_digest(),
        "thresholds": dict(FROZEN_THRESHOLDS),
        "discovered_minimal_sets": [list(item) for item in discovered],
        "recovered_circuit": list(recovered),
        "planted_forward_circuit": sorted(C_FORWARD),
        "planted_equiv_circuit": sorted(C_EQUIV),
        "literal": literal,
        "iid_sufficiency_error": restoration_error(iid_restored, iid_full),
        "ood_sufficiency_error": restoration_error(ood_restored, ood_full),
        "minimality_drop_errors": minimality,
        "necessity_delta": necessity,
        "restore_equiv_error": restoration_error(restore_equiv, iid_full),
        "steer_executed_nmse": steer_error,
        "path_ab_nmse": nmse(path_a, path_b),
        "path_a_vs_h2_nmse": nmse(
            model.activations(iid_state, iid_action)["h2"],
            model.activations(iid_state, iid_action)["h4"],
        ),
        "cancellation": cancel,
        "signed": signed,
        "decoy_target_ablation_nmse": decoy_target_nmse,
        "decoy_activation_energy": decoy_activation,
        "matched_controls": control_eval,
        "specificity": specificity,
        "gauge": gauge,
        "interventions": interventions,
        "out_of_support_labeled": any(not item["in_support"] for item in interventions),
        "gates": gates,
        "status": "MECHANISM_RECOVERY_PASSED" if passed else "NEGATIVE_RESULT",
        "evidence_level": "Causal effect" if passed else "None",
        "claim_boundary": (
            "synthetic IBD mechanism-recovery test only; not a Qwen, JEPA, workspace, "
            "Platonic, or planning claim; does not alter HARD-002, IBD-001, or IBD-002"
        ),
        "hard002_primary_seeds_reused": False,
        "ibd002_executed": False,
    }


def run_stage(stage: str, *, device_name: str = "cpu", samples: int = 128) -> dict[str, Any]:
    seeds = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    rows = [
        run_coalition_recovery(seed=seed, stage=stage, device_name=device_name, samples=samples)
        for seed in seeds
    ]
    passed = all(row["status"] == "MECHANISM_RECOVERY_PASSED" for row in rows)
    return {
        "experiment_id": "CRCT-COALITION-IBD-003",
        "stage": stage,
        "threshold_digest": threshold_digest(),
        "seeds": list(seeds),
        "all_seeds_passed": passed,
        "status": "MECHANISM_RECOVERY_PASSED" if passed else "NEGATIVE_RESULT",
        "evidence_level": "Causal effect" if passed else "None",
        "rows": rows,
        "claim_boundary": rows[0]["claim_boundary"],
        "ibd002_executed": False,
        "hard002_status_preserved": "NEGATIVE_RESULT",
    }


def main() -> int:
    import argparse
    from pathlib import Path

    from causal_workspace_jepa.common.provenance import collect_provenance, write_provenance

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    payload = run_stage(args.stage, device_name=args.device, samples=args.samples)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        write_provenance(
            path.with_suffix(".provenance.json"),
            collect_provenance(
                command=(
                    "python -m causal_workspace_jepa.experiments.cross_domain."
                    f"crct_coalition_ibd_003 --stage {args.stage}"
                ),
                resource_profile="configs/resource/cpu_vps.yaml",
                seed=payload["seeds"][0],
            ),
            extra={"metrics": path.as_posix(), "experiment_id": "CRCT-COALITION-IBD-003"},
        )
    print(text)
    return 0 if payload["all_seeds_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
