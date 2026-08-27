"""CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006: conditional downstream mediation.

Does not mutate 001–005. Learned training is forbidden until freeze.
Not a JEPA-objective experiment. 005 remains INCONCLUSIVE. Level 3 is not
authorized. Residual-unit membership is not required by fiat.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as parent
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta as wm002
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta_003 as wm003
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta_005 as wm005

EXPERIMENT_ID = "CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006"
MODULE = "causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_conditional_mediator_006"
PARENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-005"
CONFIG_PATH = Path("configs/experiments/crct_learned_wm_conditional_mediator_v6.json")

LADDER_RUNGS = (800, 2000, 5000)
DEVELOPMENT_SEEDS = (173, 179, 181)
CONFIRMATION_SEEDS = (1171, 1181, 1187)
PROPOSED_DEVELOPMENT_SEEDS = DEVELOPMENT_SEEDS
PROPOSED_CONFIRMATION_SEEDS = CONFIRMATION_SEEDS
FORBIDDEN_SEEDS = frozenset(wm005.FORBIDDEN_SEEDS) | {
    109,
    113,
    127,
    1103,
    1109,
    1117,
}

PHYSICS_DEPENDENCY = dict(wm003.PHYSICS_DEPENDENCY)
INDEPENDENT_CONTROLS = wm003.INDEPENDENT_CONTROLS
DOWNSTREAM_OF_AX = wm003.DOWNSTREAM_OF_AX

FROZEN_THRESHOLDS = {
    "competence_nmse_max": 0.05,
    "sufficiency_nmse_max": 0.05,
    "necessity_nmse_min": 0.10,
    "specificity_ratio_min": 2.0,
    "max_coalition": 4,
    "min_step_nmse": 0.02,
    "random_control_count": 32,
    "random_control_sufficient_max": 0,
    "counterfactual_gap_min": 0.50,
    "counterfactual_pairs": 64,
    "gauge_function_mse_max": 1e-8,
    "cancellation_member_nmse_min": 0.02,
    "n_down_min": 0.50,
    "s_down_min": 0.50,
    "interaction_abs_min": 0.20,
    "gap_eps": 1e-8,
    "cf_effect_floor": 1e-8,
    "ladder_rungs": list(LADDER_RUNGS),
}
DRAFT_THRESHOLDS = FROZEN_THRESHOLDS

LEVEL2B_CLASSES = frozenset(
    {"DIRECT", "DOWNSTREAM_F1", "DOWNSTREAM_F2", "DOWNSTREAM_F1_F2"}
)
LEVEL2B_STATUSES = frozenset(
    {
        "DIRECT_TRANSMISSION_PASSED",
        "DOWNSTREAM_F1_MEDIATION_PASSED",
        "DOWNSTREAM_F2_MEDIATION_PASSED",
        "DOWNSTREAM_F1_F2_MEDIATION_PASSED",
    }
)

PLANT_MODES = (
    "early_carrier_f1",
    "early_carrier_f2",
    "early_carrier_f1_f2",
    "true_direct",
    "redundant_downstream",
    "interacting_downstream",
    "early_carrier_f1_gauge",
)

V_UP = ("act_0",)


def _claim_boundary() -> str:
    return (
        "supervised residual-MLP PointMass world model only; not a JEPA objective; "
        "Level 2B conditional downstream mediation; Level 3 is not authorized; "
        "does not reinterpret 001/002/003/004/005; "
        "005 remains INCONCLUSIVE with confirmation CLOSED; "
        "does not require residual-unit membership by fiat; "
        "cached r2_P does not assign Stage-2B class; "
        "mean-fill MSRS is not the same object as recompute G_V; "
        "does not alter HARD-002, IBD-002, or IBD-003"
    )


def _require_execution_authorized() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("config experiment_id mismatch")
    if config.get("execution_authorized") is not True:
        raise ValueError("CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006 execution is not authorized")
    if config.get("status") == "DRAFT_NOT_PREREGISTERED":
        raise ValueError("CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006 is not frozen")


def n_down(g_held: float, g_v: float, *, eps: float | None = None) -> float:
    """Conditional downstream necessity: 1 - effect(V_up patch + hold D) / effect(V_up patch)."""

    floor = FROZEN_THRESHOLDS["gap_eps"] if eps is None else float(eps)
    if g_v <= floor:
        return float("nan")
    return 1.0 - (float(g_held) / float(g_v))


def s_down(g_restore: float, g_damaged: float, g_v: float, *, eps: float | None = None) -> float:
    """Fraction of mediator-transmitted effect restored from a factual-held downstream reference."""

    floor = FROZEN_THRESHOLDS["gap_eps"] if eps is None else float(eps)
    denom = float(g_v) - float(g_damaged)
    if denom <= floor:
        return float("nan")
    return (float(g_restore) - float(g_damaged)) / denom


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _high(value: float, bar: float) -> bool:
    return _finite(value) and value >= bar


class ConditionalPlantedPredictor(torch.nn.Module):
    """Known-mechanism residual plants. Not scientific evidence about learned models."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in PLANT_MODES:
            raise ValueError(f"unknown plant {mode}")
        self.mode = mode
        self.out = torch.nn.Linear(6, 4, bias=False)
        with torch.no_grad():
            self.out.weight.zero_()
            readout = 3 if mode == "early_carrier_f1_gauge" else 0
            self.out.weight[2, readout] = 1.0

    def _zeros(self, action: Tensor) -> Tensor:
        return torch.zeros(action.shape[0], 6, device=action.device, dtype=action.dtype)

    def encode_h0(self, action: Tensor, overrides: Mapping[str, Tensor] | None = None) -> Tensor:
        over = overrides or {}
        e = self._zeros(action)
        e[:, 0] = action[:, 0]
        e = parent._override_units(e, "act", over)
        h0 = self._zeros(action)
        if self.mode == "true_direct":
            h0[:, 0] = e[:, 0]
        else:
            h0[:, 1] = e[:, 0]
        return h0

    def f1(self, h0: Tensor) -> Tensor:
        r1 = torch.zeros_like(h0)
        if self.mode in {"early_carrier_f1", "early_carrier_f1_gauge", "redundant_downstream"}:
            dest = 3 if self.mode == "early_carrier_f1_gauge" else 0
            r1[:, dest] = h0[:, 1]
            r1[:, 5] = 7.0
        elif self.mode == "early_carrier_f1_f2":
            r1[:, 2] = h0[:, 1]
        elif self.mode == "interacting_downstream":
            r1[:, 0] = 3.0 * h0[:, 1]
        return r1

    def f2(self, h1: Tensor) -> Tensor:
        r2 = torch.zeros_like(h1)
        if self.mode == "early_carrier_f2":
            r2[:, 0] = h1[:, 1]
        elif self.mode == "early_carrier_f1_f2":
            r2[:, 0] = h1[:, 2]
        elif self.mode == "redundant_downstream":
            r2[:, 0] = h1[:, 1]
        elif self.mode == "interacting_downstream":
            r2[:, 0] = -2.0 * h1[:, 1]
        return r2

    def compose(self, h0: Tensor, r1: Tensor, r2: Tensor) -> Tensor:
        return self.out(h0 + r1 + r2)

    def forward_path(
        self,
        state: Tensor,
        action: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
        path_holds: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        del state, path_holds
        over = overrides or {}
        h0 = self.encode_h0(action, over)
        r1 = parent._override_units(self.f1(h0), "b1", over)
        h1 = h0 + r1
        r2 = parent._override_units(self.f2(h1), "b2", over)
        e = self._zeros(action)
        e[:, 0] = action[:, 0]
        e = parent._override_units(e, "act", over)
        sites = {f"act_{i}": e[:, i] for i in range(parent.HIDDEN)}
        sites.update({f"b1_{i}": r1[:, i] for i in range(parent.HIDDEN)})
        sites.update({f"b2_{i}": r2[:, i] for i in range(parent.HIDDEN)})
        paths = {"h0": h0, "r1": r1, "h1": h1, "r2": r2, "h2": h1 + r2, "hid1": r1, "hid2": r2}
        return self.compose(h0, r1, r2), sites, paths


@dataclass(frozen=True)
class ResidualTraj:
    h0: Tensor
    r1: Tensor
    r2: Tensor
    y: Tensor
    sites: dict[str, Tensor]


def _traj_from_paths(
    y: Tensor, sites: dict[str, Tensor], paths: Mapping[str, Tensor]
) -> ResidualTraj:
    return ResidualTraj(
        h0=paths["h0"],
        r1=paths["r1"],
        r2=paths["r2"],
        y=y,
        sites=sites,
    )


def _cf_forwards(
    model: Any,
    v_up: Sequence[str],
    seed: int,
) -> tuple[ResidualTraj, ResidualTraj, ResidualTraj, Tensor, Tensor, Tensor]:
    state, action_a, action_b = wm003._cf_pairs(seed)
    with torch.no_grad():
        y_a, sites_a, path_a = model.forward_path(state, action_a, None, None)
        y_b, sites_b, path_b = model.forward_path(state, action_b, None, None)
        patch = {name: sites_b[name] for name in v_up}
        y_p, sites_p, path_p = model.forward_path(state, action_a, patch, None)
    traj_a = _traj_from_paths(y_a, sites_a, path_a)
    traj_b = _traj_from_paths(y_b, sites_b, path_b)
    traj_p = _traj_from_paths(y_p, sites_p, path_p)
    return traj_a, traj_b, traj_p, state, action_a, action_b


def _b2_overrides(traj_p: ResidualTraj, v_up: Sequence[str]) -> dict[str, Tensor]:
    return {name: traj_p.sites[name] for name in v_up if str(name).startswith("b2_")}


def _recompute_r2(model: Any, h1: Tensor, traj_p: ResidualTraj, v_up: Sequence[str]) -> Tensor:
    """F2(h1), then re-apply any V_up b2_* hid2 patches from the P forward."""

    overrides = _b2_overrides(traj_p, v_up)
    inner = getattr(model, "inner", None)
    if inner is not None and hasattr(inner, "b2_w1"):
        hid2 = torch.tanh(inner.b2_w1(h1)) @ inner.q_b2
        hid2 = parent._override_units(hid2, "b2", overrides)
        return inner.b2_w2(hid2)
    return parent._override_units(model.f2(h1), "b2", overrides)


def _y_hold(
    model: Any,
    traj_p: ResidualTraj,
    traj_a: ResidualTraj,
    hold: Sequence[str],
    *,
    recompute_descendants: bool,
    v_up: Sequence[str] = (),
) -> Tensor:
    held = set(hold)
    h0 = traj_p.h0
    r1 = traj_a.r1 if "r1" in held else traj_p.r1
    if "r2" in held:
        r2 = traj_a.r2
    elif recompute_descendants:
        r2 = _recompute_r2(model, h0 + r1, traj_p, v_up)
    else:
        r2 = traj_p.r2
    return model.compose(h0, r1, r2)


def _y_enable(
    model: Any,
    traj_p: ResidualTraj,
    traj_a: ResidualTraj,
    enable: Sequence[str],
    *,
    recompute_descendants: bool,
    v_up: Sequence[str] = (),
) -> Tensor:
    """Factual-hold residual messages, then re-enable D from current parents."""

    enabled = set(enable)
    h0 = traj_p.h0
    r1 = traj_p.r1 if "r1" in enabled else traj_a.r1
    r2 = traj_a.r2
    if "r2" in enabled:
        r2 = _recompute_r2(model, h0 + r1, traj_p, v_up)
    elif recompute_descendants and "r1" in enabled:
        r2 = _recompute_r2(model, h0 + r1, traj_p, v_up)
    return model.compose(h0, r1, r2)


def _cf_effect(y_a: Tensor, y_b: Tensor, channel: str) -> float:
    t = parent.CHANNELS.index(channel)
    return float((y_b[:, t] - y_a[:, t]).square().median().item())


def conditional_downstream_report(
    model: Any,
    *,
    v_up: Sequence[str] = V_UP,
    seed: int = 173,
) -> dict[str, Any]:
    """Pearl-style conditional downstream mediation on a planted residual net."""

    traj_a, traj_b, traj_p, state, action_a, _action_b = _cf_forwards(model, v_up, seed)
    bar = float(FROZEN_THRESHOLDS["counterfactual_gap_min"])
    gaps: dict[str, float] = {}
    for channel in parent.CHANNELS:
        gaps[f"g_v_{channel}"] = wm003._gap_closed(traj_a.y, traj_b.y, traj_p.y, channel)
        gaps[f"cf_effect_{channel}"] = _cf_effect(traj_a.y, traj_b.y, channel)
    g_v = gaps["g_v_dvx"]

    y_hold_r1 = _y_hold(model, traj_p, traj_a, ("r1",), recompute_descendants=True, v_up=v_up)
    y_hold_r2 = _y_hold(model, traj_p, traj_a, ("r2",), recompute_descendants=True, v_up=v_up)
    y_hold_both = _y_hold(model, traj_p, traj_a, ("r1", "r2"), recompute_descendants=True, v_up=v_up)
    y_hold_r1_cached_r2 = _y_hold(
        model, traj_p, traj_a, ("r1",), recompute_descendants=False, v_up=v_up
    )

    g_hold_r1 = wm003._gap_closed(traj_a.y, traj_b.y, y_hold_r1, parent.PRIMARY)
    g_hold_r2 = wm003._gap_closed(traj_a.y, traj_b.y, y_hold_r2, parent.PRIMARY)
    g_hold_both = wm003._gap_closed(traj_a.y, traj_b.y, y_hold_both, parent.PRIMARY)
    g_damaged = g_hold_both

    y_en_r1_hold = _y_enable(
        model, traj_p, traj_a, ("r1",), recompute_descendants=False, v_up=v_up
    )
    y_en_r2_hold = _y_enable(
        model, traj_p, traj_a, ("r2",), recompute_descendants=False, v_up=v_up
    )
    y_en_r1_desc = _y_enable(
        model, traj_p, traj_a, ("r1",), recompute_descendants=True, v_up=v_up
    )
    y_en_both = _y_enable(
        model, traj_p, traj_a, ("r1", "r2"), recompute_descendants=True, v_up=v_up
    )

    g_en_r1_hold = wm003._gap_closed(traj_a.y, traj_b.y, y_en_r1_hold, parent.PRIMARY)
    g_en_r2_hold = wm003._gap_closed(traj_a.y, traj_b.y, y_en_r2_hold, parent.PRIMARY)
    g_en_r1_desc = wm003._gap_closed(traj_a.y, traj_b.y, y_en_r1_desc, parent.PRIMARY)
    g_en_both = wm003._gap_closed(traj_a.y, traj_b.y, y_en_both, parent.PRIMARY)

    y_pp = traj_p.y
    y_pa = _y_hold(model, traj_p, traj_a, ("r2",), recompute_descendants=False, v_up=v_up)
    y_ap = _y_hold(model, traj_p, traj_a, ("r1",), recompute_descendants=True, v_up=v_up)
    y_aa = y_hold_both
    t = parent.CHANNELS.index(parent.PRIMARY)
    interaction = float((y_pp[:, t] - y_pa[:, t] - y_ap[:, t] + y_aa[:, t]).median().item())

    n_r1 = n_down(g_hold_r1, g_v)
    n_r2 = n_down(g_hold_r2, g_v)
    n_both = n_down(g_hold_both, g_v)
    s_r1_hold = s_down(g_en_r1_hold, g_damaged, g_v)
    s_r2_hold = s_down(g_en_r2_hold, g_damaged, g_v)
    s_r1_desc = s_down(g_en_r1_desc, g_damaged, g_v)
    s_both = s_down(g_en_both, g_damaged, g_v)

    with torch.no_grad():
        meanfill = _meanfill_overrides(traj_a.sites, keep=(), batch=state.shape[0])
        meanfill.update({name: traj_p.sites[name] for name in v_up})
        y_meanfill_act, _, _ = model.forward_path(state, action_a, meanfill, None)
        restore_a = _meanfill_overrides(traj_a.sites, keep=(), batch=state.shape[0])
        restore_a.update({name: traj_a.sites[name] for name in v_up})
        y_restore_a, _, _ = model.forward_path(state, action_a, restore_a, None)
    g_meanfill_act = wm003._gap_closed(traj_a.y, traj_b.y, y_meanfill_act, parent.PRIMARY)
    meanfill_err = parent.restoration_error(
        parent._channel(y_restore_a, parent.PRIMARY),
        parent._channel(traj_a.y, parent.PRIMARY),
    )
    recompute_err = parent.restoration_error(
        parent._channel(traj_p.y, parent.PRIMARY),
        parent._channel(traj_b.y, parent.PRIMARY),
    )

    spec = specificity_conditional(gaps)
    path_class = classify_conditional_downstream(
        {
            "g_v": g_v,
            "g_damaged": g_damaged,
            "n_r1": n_r1,
            "n_r2": n_r2,
            "s_r1_hold": s_r1_hold,
            "s_r2_hold": s_r2_hold,
            "s_r1_desc": s_r1_desc,
            "s_both": s_both,
            "interaction": interaction,
        }
    )
    return {
        "mode": model.mode,
        "v_up": list(v_up),
        "g_v": g_v,
        "g_hold_r1": g_hold_r1,
        "g_hold_r2": g_hold_r2,
        "g_hold_both": g_hold_both,
        "g_hold_r1_cached_r2": wm003._gap_closed(traj_a.y, traj_b.y, y_hold_r1_cached_r2, parent.PRIMARY),
        "g_meanfill_act": g_meanfill_act,
        "meanfill_restore_nmse_act": meanfill_err,
        "recompute_patch_nmse_to_b": recompute_err,
        "n_r1": n_r1,
        "n_r2": n_r2,
        "n_both": n_both,
        "s_r1_hold": s_r1_hold,
        "s_r2_hold": s_r2_hold,
        "s_r1_desc": s_r1_desc,
        "s_both": s_both,
        "interaction_r1_r2": interaction,
        "path_class": path_class,
        "specificity": spec,
        "channel_gaps": gaps,
        "msrs_recompute_finds_v_up": g_v >= bar,
        "msrs_meanfill_act_only": g_meanfill_act >= bar,
        "claim_boundary": _claim_boundary(),
    }


def _meanfill_overrides(
    sites: Mapping[str, Tensor],
    *,
    keep: Sequence[str],
    batch: int,
) -> dict[str, Tensor]:
    kept = set(keep)
    means = {name: sites[name].mean() for name in parent.SITE_NAMES}
    return {name: parent._as_batch(means[name], batch) for name in parent.SITE_NAMES if name not in kept}


def specificity_conditional(gaps: Mapping[str, float]) -> dict[str, Any]:
    """Independent controls are Δvy and Δy. Δx is downstream of ax, not a negative control."""

    floor = float(FROZEN_THRESHOLDS["cf_effect_floor"])
    ratio_bar = float(FROZEN_THRESHOLDS["specificity_ratio_min"])
    g_v = float(gaps["g_v_dvx"])
    rows: dict[str, Any] = {}
    failed = False
    for name in INDEPENDENT_CONTROLS:
        effect = float(gaps[f"cf_effect_{name}"])
        g_ctrl = float(gaps[f"g_v_{name}"])
        inactive = effect < floor
        if inactive:
            ratio = float("inf")
            ok = True
        else:
            ratio = g_v / max(g_ctrl, floor)
            ok = ratio >= ratio_bar
            failed = failed or (not ok)
        rows[name] = {
            "cf_effect": effect,
            "g_v": g_ctrl,
            "inactive": inactive,
            "ratio_dvx_over_control": ratio,
            "passed": ok,
        }
    dx_effect = float(gaps["cf_effect_dx"])
    return {
        "failed": failed,
        "independent": rows,
        "dx_downstream_not_a_negative_control": True,
        "dx_cf_effect": dx_effect,
        "physics_dependency": dict(PHYSICS_DEPENDENCY),
    }


def classify_conditional_downstream(metrics: Mapping[str, float]) -> str | None:
    bar = float(FROZEN_THRESHOLDS["counterfactual_gap_min"])
    n_bar = float(FROZEN_THRESHOLDS["n_down_min"])
    s_bar = float(FROZEN_THRESHOLDS["s_down_min"])
    i_bar = float(FROZEN_THRESHOLDS["interaction_abs_min"])
    g_v = float(metrics["g_v"])
    if (not math.isfinite(g_v)) or g_v < bar:
        return None
    n1 = _high(float(metrics["n_r1"]), n_bar)
    n2 = _high(float(metrics["n_r2"]), n_bar)
    s1 = _high(float(metrics["s_r1_hold"]), s_bar)
    s2 = _high(float(metrics["s_r2_hold"]), s_bar)
    s1d = _high(float(metrics["s_r1_desc"]), s_bar)
    sb = _high(float(metrics["s_both"]), s_bar)
    interacting = abs(float(metrics["interaction"])) >= i_bar
    g_damaged = float(metrics["g_damaged"])
    if (not n1) and (not n2) and g_damaged >= bar:
        return "DIRECT"
    if n1 and (not n2) and s1:
        return "DOWNSTREAM_F1"
    if n2 and (not n1) and s2:
        return "DOWNSTREAM_F2"
    if (not n1) and (not n2) and s1 and s2:
        return "REDUNDANT_DOWNSTREAM"
    if n1 and n2 and sb and interacting and s1d:
        return "DOWNSTREAM_F1_F2"
    if n1 and n2 and sb and (not s1) and (not s2) and (not interacting):
        return "INTERACTING_DOWNSTREAM"
    return "DOWNSTREAM_UNRESOLVED"


def ontology_identifiability(seed: int = 173) -> dict[str, Any]:
    """Compare unit-level vs branch-message interventions on planted residual nets."""

    model_f1 = ConditionalPlantedPredictor("early_carrier_f1")
    traj_a, traj_b, traj_p, _state, _action_a, _action_b = _cf_forwards(model_f1, V_UP, seed)
    y_msg = _y_hold(model_f1, traj_p, traj_a, ("r1",), recompute_descendants=True)
    y_unit = model_f1.compose(traj_p.h0, traj_a.r1, model_f1.f2(traj_p.h0 + traj_a.r1))
    t = parent.CHANNELS.index(parent.PRIMARY)
    unit_equals_message = float(torch.mean((y_msg[:, t] - y_unit[:, t]).square()).item())
    r1_hold_decoy = traj_p.r1.clone()
    r1_hold_decoy[:, 5] = traj_a.r1[:, 5]
    y_decoy_unit = model_f1.compose(
        traj_p.h0, r1_hold_decoy, model_f1.f2(traj_p.h0 + r1_hold_decoy)
    )
    decoy_still_mediated = wm003._gap_closed(traj_a.y, traj_b.y, y_decoy_unit, parent.PRIMARY)

    model_f2 = ConditionalPlantedPredictor("early_carrier_f2")
    a2, b2, p2, *_ = _cf_forwards(model_f2, V_UP, seed)
    y_r2 = _y_hold(model_f2, p2, a2, ("r2",), recompute_descendants=True)
    y_skip2 = model_f2.compose(a2.h0, a2.r1, p2.r2)
    skip2_vs_r2 = float(torch.mean((y_r2[:, t] - y_skip2[:, t]).square()).item())
    g_skip2 = wm003._gap_closed(a2.y, b2.y, y_skip2, parent.PRIMARY)
    g_r2 = wm003._gap_closed(a2.y, b2.y, y_r2, parent.PRIMARY)
    report = conditional_downstream_report(model_f1, seed=seed)
    gauge = conditional_downstream_report(
        ConditionalPlantedPredictor("early_carrier_f1_gauge"), seed=seed
    )
    return {
        "ontology_a_node_v_down_identifiable": False,
        "ontology_b_branch_message_v_down_identifiable": True,
        "preferred_ontology": "B",
        "reason": (
            "h1 = h0 + F1(h0) and h2 = h1 + F2(h1) couple node state, residual "
            "message, and skip message. Intervening b1_* overwrites hid1 and "
            "therefore r1 and skip2=h1 together. do(r1) and do(r2) name the "
            "residual messages; skip2 remains a separate hold."
        ),
        "unit_r1_hold_mse": unit_equals_message,
        "decoy_b1_5_still_transmits_g_v": decoy_still_mediated,
        "f2_skip2_hold_gap": g_skip2,
        "f2_r2_hold_gap": g_r2,
        "skip2_hold_vs_r2_hold_mse": skip2_vs_r2,
        "f1_class": report["path_class"],
        "f1_gauge_class": gauge["path_class"],
        "gauge_literal_readout_changed": True,
        "gauge_downstream_role_survives": gauge["path_class"] == report["path_class"],
    }


def planted_suite(seed: int = 173) -> dict[str, dict[str, Any]]:
    return {
        mode: conditional_downstream_report(ConditionalPlantedPredictor(mode), seed=seed)
        for mode in (
            "early_carrier_f1",
            "early_carrier_f2",
            "early_carrier_f1_f2",
            "true_direct",
            "redundant_downstream",
            "interacting_downstream",
        )
    }


def critical_early_carrier_f1(seed: int = 173) -> dict[str, Any]:
    """Global recompute-MSRS finds the action stem; conditional mediation finds F1."""

    row = conditional_downstream_report(ConditionalPlantedPredictor("early_carrier_f1"), seed=seed)
    row["old_stage_a_recompute_object"] = "action_stem"
    row["new_conditional_object"] = "r1"
    row["critical_pass"] = bool(
        row["path_class"] == "DOWNSTREAM_F1"
        and row["msrs_recompute_finds_v_up"]
        and (not row["msrs_meanfill_act_only"])
        and _high(float(row["n_r1"]), float(FROZEN_THRESHOLDS["n_down_min"]))
        and (not _high(float(row["n_r2"]), float(FROZEN_THRESHOLDS["n_down_min"])))
    )
    return row


def mechanism_hierarchy() -> dict[str, str]:
    return {
        "L1_C": "information carrier (may be an early bottleneck; not automatically the computation)",
        "L2A_V_up": "upstream causal mediator: node coalition whose B-patch into A produces m_up",
        "L2B_V_down": "conditional downstream mediator: branch-message coalition necessary/sufficient for transmitting m_up",
        "L3_E": "identifiable causal edges/paths; only after V_up and V_down are validated",
    }


def refined_mechanism_tuple() -> dict[str, str]:
    return {
        "previous": "M = (V, E, I) with V = global MSRS",
        "refined": "M = (C, V_up, V_down, E, I)",
        "C": "Level-1 carrier; diagnostic, not a circuit claim",
        "V_up": "Level-2A node coalition / upstream mediator",
        "V_down": "Level-2B branch-message coalition under V_up patch",
        "E": "Level-3 identifiable edges after V_down",
        "I": "Pearl do-semantics: hold D at A with descendant recompute; factual-hold sufficiency reference",
    }


def msrs_early_bottleneck_bias() -> dict[str, Any]:
    """Methodological finding from plants. Not a learned-model result and not a 005 reinterpretation."""

    return {
        "global_meanfill_msrs_biased_toward_early_bottlenecks": False,
        "recompute_or_g_full_biased_toward_early_bottlenecks": True,
        "causal_reason": (
            "Patching an early carrier V_up from B into A and recomputing descendants "
            "(G_full / intact downstream) restores the target whenever downstream F1/F2 "
            "can still transform that carrier. Mean-filling non-coalition hidden units "
            "disables that downstream machinery, so mean-fill MSRS is not the same object."
        ),
        "conditional_v_down_needed": True,
    }


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_digest() -> str:
    parts = [
        Path(parent.__file__).read_bytes(),
        Path(wm002.__file__).read_bytes(),
        Path(wm003.__file__).read_bytes(),
        Path(wm005.__file__).read_bytes(),
        Path(__file__).read_bytes(),
    ]
    return hashlib.sha256(b"\n".join(parts)).hexdigest()


class LearnedResidualAdapter:
    """Expose F1/F2 maps of the 001–005 residual MLP without changing 005."""

    def __init__(self, model: wm005.PathAwareActionDeltaPredictor) -> None:
        self.inner = model
        self.mode = "learned"
        self.out = model.out

    def f1(self, h0: Tensor) -> Tensor:
        hid1 = torch.tanh(self.inner.b1_w1(h0)) @ self.inner.q_b1
        return self.inner.b1_w2(hid1)

    def f2(self, h1: Tensor) -> Tensor:
        hid2 = torch.tanh(self.inner.b2_w1(h1)) @ self.inner.q_b2
        return self.inner.b2_w2(hid2)

    def compose(self, h0: Tensor, r1: Tensor, r2: Tensor) -> Tensor:
        return self.inner.out(h0 + r1 + r2)

    def forward_path(
        self,
        state: Tensor,
        action: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
        path_holds: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        return self.inner.forward_path(state, action, overrides, path_holds)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _status_for_class(path_class: str) -> str:
    mapping = {
        "DIRECT": "DIRECT_TRANSMISSION_PASSED",
        "DOWNSTREAM_F1": "DOWNSTREAM_F1_MEDIATION_PASSED",
        "DOWNSTREAM_F2": "DOWNSTREAM_F2_MEDIATION_PASSED",
        "DOWNSTREAM_F1_F2": "DOWNSTREAM_F1_F2_MEDIATION_PASSED",
        "REDUNDANT_DOWNSTREAM": "REDUNDANT_DOWNSTREAM",
        "INTERACTING_DOWNSTREAM": "INTERACTING_DOWNSTREAM",
        "DOWNSTREAM_UNRESOLVED": "DOWNSTREAM_UNRESOLVED",
    }
    return mapping[path_class]


def _level_for(status: str) -> int:
    if status in LEVEL2B_STATUSES:
        return 2
    if status in {
        "DOWNSTREAM_UNRESOLVED",
        "REDUNDANT_DOWNSTREAM",
        "INTERACTING_DOWNSTREAM",
    }:
        return 2
    return 0


def adjudicate_seed(
    *,
    coalition: Sequence[str],
    sufficiency_dvx: float,
    drop_still_sufficient: bool,
    necessity_dvx: float,
    spec_failed: bool,
    random_sufficient: int,
    act_random_sufficient: int,
    g_v: float,
    gauge_fn: float,
    g_suff: float,
    g_nec: float,
    g_path: str | None,
    path_class: str | None,
    branch_control_failed: bool,
    stage_2b_ran: bool,
) -> tuple[str, int]:
    if not coalition:
        status = "LOCALIZATION_FAILED"
    elif sufficiency_dvx > FROZEN_THRESHOLDS["sufficiency_nmse_max"]:
        status = "SUFFICIENCY_FAILED"
    elif drop_still_sufficient:
        status = "MINIMALITY_FAILED"
    elif necessity_dvx < FROZEN_THRESHOLDS["necessity_nmse_min"]:
        status = "NECESSITY_FAILED"
    elif spec_failed:
        status = "SPECIFICITY_FAILED"
    elif random_sufficient > FROZEN_THRESHOLDS["random_control_sufficient_max"]:
        status = "INCONCLUSIVE"
    elif act_random_sufficient > FROZEN_THRESHOLDS["random_control_sufficient_max"]:
        status = "INCONCLUSIVE"
    elif g_v < FROZEN_THRESHOLDS["counterfactual_gap_min"]:
        status = "INCONCLUSIVE"
    elif gauge_fn > FROZEN_THRESHOLDS["gauge_function_mse_max"]:
        status = "GAUGE_FAILED"
    elif (
        g_suff > FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        or g_nec < FROZEN_THRESHOLDS["necessity_nmse_min"]
    ):
        status = "GAUGE_FAILED"
    elif not stage_2b_ran or path_class is None:
        status = "INCONCLUSIVE"
    elif g_path != path_class:
        status = "DOWNSTREAM_CLASS_GAUGE_UNSTABLE"
    elif path_class == "DOWNSTREAM_UNRESOLVED":
        status = "DOWNSTREAM_UNRESOLVED"
    elif path_class == "REDUNDANT_DOWNSTREAM":
        status = "REDUNDANT_DOWNSTREAM"
    elif path_class == "INTERACTING_DOWNSTREAM":
        status = "INTERACTING_DOWNSTREAM"
    elif branch_control_failed:
        status = "BRANCH_CONTROL_FAILED"
    elif path_class in LEVEL2B_CLASSES:
        status = _status_for_class(path_class)
    else:
        status = "INCONCLUSIVE"
    return status, _level_for(status)


def train_model(
    seed: int,
    state: Tensor,
    action: Tensor,
    delta: Tensor,
    steps: int,
) -> tuple[wm005.PathAwareActionDeltaPredictor, list[float], str]:
    if int(seed) in FORBIDDEN_SEEDS:
        raise ValueError(f"forbidden seed {seed}")
    _require_execution_authorized()
    model = wm005.PathAwareActionDeltaPredictor(seed)
    opt = torch.optim.Adam(model.parameters(), lr=parent.LR)
    n = state.shape[0]
    gen = torch.Generator().manual_seed(int(seed) + 13)
    curve: list[float] = []
    mark = max(int(steps) // 10, 1)
    last = 0.0
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(parent.BATCH, n),), generator=gen)
        pred = model(state[idx], action[idx])
        loss = torch.mean((pred - delta[idx]).square())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
        if (step + 1) % mark == 0 or step + 1 == int(steps):
            curve.append(last)
    payload = []
    for key, tensor in model.state_dict().items():
        payload.append(key.encode("utf-8"))
        payload.append(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest = hashlib.sha256(b"".join(payload)).hexdigest()
    return model, curve[:10], digest


def _act_random_sufficient(
    model: wm005.PathAwareActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    means: Mapping[str, Tensor],
    coalition: Sequence[str],
    original_delta: Tensor,
    rng: random.Random,
) -> int:
    size = max(len(coalition), 1)
    pool = [list(item) for item in combinations(parent.ACT_SITES, size) if set(item) != set(coalition)]
    rng.shuffle(pool)
    rows = pool[: int(FROZEN_THRESHOLDS["random_control_count"])]
    return int(
        sum(
            parent._predict_nmse(
                model, state, action, parent._mean_except(means, row, state.shape[0]), parent.PRIMARY, original_delta
            )
            <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
            for row in rows
        )
    )


def _shuffled_r1_s_down(
    adapter: LearnedResidualAdapter,
    traj_a: ResidualTraj,
    traj_b: ResidualTraj,
    traj_p: ResidualTraj,
    g_v: float,
    g_damaged: float,
) -> float:
    perm = torch.randperm(traj_p.r1.shape[0])
    y_shuf = adapter.compose(traj_p.h0, traj_p.r1[perm], traj_a.r2)
    g_shuf = wm003._gap_closed(traj_a.y, traj_b.y, y_shuf, parent.PRIMARY)
    return s_down(g_shuf, g_damaged, g_v)


def _run_mechanism(
    model: wm005.PathAwareActionDeltaPredictor,
    seed: int,
    stage: str,
    train_s: Tensor,
    train_a: Tensor,
    dev_s: Tensor,
    dev_a: Tensor,
) -> dict[str, Any]:
    means = parent._site_means(model, train_s, train_a)
    raw_coalition, _ = parent.greedy_restore(model, dev_s, dev_a, means, target=parent.PRIMARY)
    coalition = parent.prune_inclusion_minimal(
        model, dev_s, dev_a, means, raw_coalition, target=parent.PRIMARY
    )
    eval_s, eval_a = dev_s, dev_a
    if stage == "confirmation":
        eval_s, eval_a, _ = parent._transitions(seed * 1000 + 71, 64)
    with torch.no_grad():
        y0, _original = model.forward_intervene(eval_s, eval_a, None)
    metrics = parent._evaluate_coalition(model, eval_s, eval_a, means, coalition, y0)
    spec = {
        name: metrics["necessity"][parent.PRIMARY] / max(metrics["necessity"][name], 1e-6)
        for name in INDEPENDENT_CONTROLS
    }
    rng = random.Random(int(seed) * 8191 + 3)
    plus_one_p, _control_sets, random_sufficient = parent._random_controls(
        model, eval_s, eval_a, means, coalition, y0, rng
    )
    action_only = set(coalition) <= set(parent.ACT_SITES) and bool(coalition)
    act_random_sufficient = 0
    if action_only:
        act_random_sufficient = _act_random_sufficient(
            model, eval_s, eval_a, means, coalition, y0, rng
        )
    drop_still_sufficient = any(
        err <= FROZEN_THRESHOLDS["sufficiency_nmse_max"] for err in metrics["minimality_drop"].values()
    )
    spec_failed = any(spec[name] < FROZEN_THRESHOLDS["specificity_ratio_min"] for name in INDEPENDENT_CONTROLS)
    stage_a_ok = (
        bool(coalition)
        and metrics["sufficiency"][parent.PRIMARY] <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        and not drop_still_sufficient
        and metrics["necessity"][parent.PRIMARY] >= FROZEN_THRESHOLDS["necessity_nmse_min"]
        and not spec_failed
        and int(random_sufficient) <= FROZEN_THRESHOLDS["random_control_sufficient_max"]
        and int(act_random_sufficient) <= FROZEN_THRESHOLDS["random_control_sufficient_max"]
    )
    adapter = LearnedResidualAdapter(model)
    report: dict[str, Any] = {}
    if coalition:
        report = conditional_downstream_report(adapter, v_up=coalition, seed=seed)
    g_v = float(report.get("g_v", 0.0))
    stage_2b_ran = bool(stage_a_ok and g_v >= FROZEN_THRESHOLDS["counterfactual_gap_min"])
    path_class = str(report["path_class"]) if stage_2b_ran else None
    branch_control_failed = False
    shuffled_s = float("nan")
    if stage_2b_ran and path_class == "DOWNSTREAM_F1":
        traj_a, traj_b, traj_p, *_ = _cf_forwards(adapter, coalition, seed)
        shuffled_s = _shuffled_r1_s_down(
            adapter, traj_a, traj_b, traj_p, g_v, float(report["g_hold_both"])
        )
        branch_control_failed = _high(shuffled_s, float(FROZEN_THRESHOLDS["s_down_min"]))

    gauged = copy.deepcopy(model)
    gauged.apply_hidden_gauge(parent._orthogonal(seed, 7), parent._orthogonal(seed, 11), parent._orthogonal(seed, 13))
    with torch.no_grad():
        gauge_fn = float(torch.mean((gauged(eval_s, eval_a) - y0).square()).item())
    g_means = parent._site_means(gauged, train_s, train_a)
    g_coal, _ = parent.greedy_restore(gauged, dev_s, dev_a, g_means, target=parent.PRIMARY)
    g_coal = parent.prune_inclusion_minimal(gauged, dev_s, dev_a, g_means, g_coal, target=parent.PRIMARY)
    with torch.no_grad():
        g_y0, _ = gauged.forward_intervene(eval_s, eval_a, None)
    g_metrics = parent._evaluate_coalition(gauged, eval_s, eval_a, g_means, g_coal, g_y0)
    g_suff = g_metrics["sufficiency"][parent.PRIMARY]
    g_nec = g_metrics["necessity"][parent.PRIMARY]
    g_adapter = LearnedResidualAdapter(gauged)
    g_report = (
        conditional_downstream_report(g_adapter, v_up=g_coal, seed=seed) if g_coal else {"g_v": 0.0, "path_class": None}
    )
    g_stage_2b = bool(
        g_coal
        and g_suff <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        and g_nec >= FROZEN_THRESHOLDS["necessity_nmse_min"]
        and float(g_report.get("g_v", 0.0)) >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
    )
    g_path = str(g_report["path_class"]) if g_stage_2b else None
    status, level = adjudicate_seed(
        coalition=coalition,
        sufficiency_dvx=metrics["sufficiency"][parent.PRIMARY],
        drop_still_sufficient=drop_still_sufficient,
        necessity_dvx=metrics["necessity"][parent.PRIMARY],
        spec_failed=spec_failed,
        random_sufficient=int(random_sufficient),
        act_random_sufficient=int(act_random_sufficient),
        g_v=g_v,
        gauge_fn=gauge_fn,
        g_suff=float(g_suff),
        g_nec=float(g_nec),
        g_path=g_path,
        path_class=path_class,
        branch_control_failed=bool(branch_control_failed),
        stage_2b_ran=bool(stage_2b_ran),
    )
    return {
        "msrs": list(coalition),
        "v_up": list(coalition),
        "v_down_messages": ["r1", "r2"],
        "pre_prune_circuit": list(raw_coalition),
        "sufficiency": metrics["sufficiency"],
        "necessity": metrics["necessity"],
        "specificity_independent": spec,
        "minimality_drop": metrics["minimality_drop"],
        "random_plus_one_p": float(plus_one_p),
        "random_control_sufficient_count": int(random_sufficient),
        "act_random_control_sufficient_count": int(act_random_sufficient),
        "g_v": _json_float(g_v),
        "g_damaged": _json_float(report.get("g_hold_both", 0.0)),
        "n_r1": _json_float(report.get("n_r1")),
        "n_r2": _json_float(report.get("n_r2")),
        "n_both": _json_float(report.get("n_both")),
        "s_r1_hold": _json_float(report.get("s_r1_hold")),
        "s_r2_hold": _json_float(report.get("s_r2_hold")),
        "s_r1_desc": _json_float(report.get("s_r1_desc")),
        "s_both": _json_float(report.get("s_both")),
        "interaction_r1_r2": _json_float(report.get("interaction_r1_r2")),
        "g_hold_r1": _json_float(report.get("g_hold_r1")),
        "g_hold_r2": _json_float(report.get("g_hold_r2")),
        "g_hold_r1_cached_r2": _json_float(report.get("g_hold_r1_cached_r2")),
        "cached_r2_assigns_class": False,
        "require_residual_units_by_fiat": False,
        "downstream_class": path_class,
        "path_class": path_class,
        "stage_a_ok": bool(stage_a_ok),
        "stage_2b_ran": bool(stage_2b_ran),
        "branch_control_failed": bool(branch_control_failed),
        "shuffled_r1_s_down": _json_float(shuffled_s),
        "action_embedding_only": action_only,
        "gauge_function_mse": gauge_fn,
        "gauge_v_up": list(g_coal),
        "gauge_sufficiency_dvx": float(g_suff),
        "gauge_necessity_dvx": float(g_nec),
        "gauge_downstream_class": g_path,
        "literal_jaccard_vs_gauge": (
            len(set(coalition) & set(g_coal)) / max(len(set(coalition) | set(g_coal)), 1)
        ),
        "status": status,
        "level": level,
        "hierarchy_level": "2B" if status in LEVEL2B_STATUSES else ("2A" if level == 2 else "0"),
        "level3_authorized": False,
        "seed_97_retrospective_pass": False,
        "mechanism_object": "M=(C,V_up,V_down,E,I)_conditional_branch_messages",
        "stage2b_specificity": report.get("specificity"),
    }


def _fit_seed(
    seed: int, stage: str, train_steps: int
) -> tuple[wm005.PathAwareActionDeltaPredictor, dict[str, Any]]:
    if int(seed) in FORBIDDEN_SEEDS:
        raise ValueError(f"forbidden seed {seed}")
    if stage == "development" and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("development seed mismatch")
    if stage == "confirmation" and seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed mismatch")
    train_s, train_a, train_d = parent._transitions(seed * 1000 + 61, 256)
    dev_s, dev_a, dev_d = parent._transitions(seed * 1000 + 67, 64)
    model, curve, ckpt = train_model(seed, train_s, train_a, train_d, train_steps)
    with torch.no_grad():
        train_pred = model(train_s, train_a)
        train_loss = float(torch.mean((train_pred - train_d).square()).item())
        train_channels = wm002._channel_report(train_pred, train_d)
    bundle = wm002.competence_bundle(model, dev_s, dev_a, dev_d)
    row: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_ID,
        "stage": stage,
        "seed": seed,
        "train_steps": int(train_steps),
        "checkpoint_sha256": ckpt,
        "train_loss_final": curve[-1] if curve else train_loss,
        "train_loss_fullsplit": train_loss,
        "train_loss_curve": curve,
        "train_channels": train_channels,
        "competence": bundle,
        "circuit_search_ran": False,
        "claim_boundary": _claim_boundary(),
        "substrate": "supervised_residual_mlp_not_jepa_objective",
        "seed_97_retrospective_pass": False,
        "level3_authorized": False,
    }
    if not bundle["passed"]:
        row["status"] = "MODEL_INCOMPETENT"
        row["evidence_level"] = "None"
    else:
        row["status"] = "COMPETENT_NOT_INTERPRETED"
        row["evidence_level"] = "None"
    return model, row


def _literal_jaccard(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    recovered = {row["seed"]: set(row.get("v_up") or row.get("msrs") or []) for row in rows if row.get("msrs")}
    out: dict[str, float] = {}
    seeds = sorted(recovered)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            union = recovered[a] | recovered[b]
            out[f"{a}_{b}"] = len(recovered[a] & recovered[b]) / max(len(union), 1)
    return out


def _shared_level2b_class(rows: Sequence[Mapping[str, Any]]) -> str | None:
    classes = {row.get("downstream_class") or row.get("path_class") for row in rows}
    if len(classes) == 1:
        only = next(iter(classes))
        if only in LEVEL2B_CLASSES:
            return str(only)
    return None


def _h_equivalent(rows: Sequence[Mapping[str, Any]], shared: str | None, passed: bool) -> bool:
    if not passed or shared is None or len(rows) < 2:
        return False
    jaccards = list(_literal_jaccard(rows).values())
    return bool(jaccards) and any(value < 1.0 for value in jaccards)


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
    train_steps: int,
    *,
    required_shared_class: str | None = None,
) -> dict[str, Any]:
    incompetent = [row for row in rows if row["status"] == "MODEL_INCOMPETENT"]
    statuses = {row["status"] for row in rows}
    shared = _shared_level2b_class(rows) if not incompetent else None
    if stage == "confirmation" and incompetent:
        status = "MODEL_INCOMPETENT_CONFIRMATION"
    elif incompetent:
        status = "MODEL_INCOMPETENT"
    elif statuses <= LEVEL2B_STATUSES and len(statuses) == 1 and shared is not None:
        status = "CONDITIONAL_DOWNSTREAM_MEDIATION_PASSED"
    elif statuses == {"REDUNDANT_DOWNSTREAM"}:
        status = "REDUNDANT_DOWNSTREAM"
    elif statuses == {"INTERACTING_DOWNSTREAM"}:
        status = "INTERACTING_DOWNSTREAM"
    elif len(statuses) == 1:
        status = next(iter(statuses))
    else:
        status = "INCONCLUSIVE"
    passed = status == "CONDITIONAL_DOWNSTREAM_MEDIATION_PASSED"
    if (
        stage == "confirmation"
        and required_shared_class is not None
        and passed
        and shared != required_shared_class
    ):
        status = "CONFIRMATION_DOWNSTREAM_CLASS_MISMATCH"
        passed = False
    h_eq = _h_equivalent(rows, shared, passed)
    return {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_ID,
        "stage": stage,
        "train_steps": int(train_steps),
        "selected_rung": int(train_steps) if not incompetent else None,
        "threshold_digest": threshold_digest(),
        "source_digest": source_digest(),
        "seeds": [row["seed"] for row in rows],
        "all_seeds_passed": passed,
        "all_seeds_competent": not bool(incompetent),
        "shared_downstream_class": shared if passed else None,
        "shared_path_class": shared if passed else None,
        "required_shared_class": required_shared_class,
        "h_equivalent": h_eq,
        "status": status,
        "evidence_level": "Causal effect" if passed else "None",
        "rows": list(rows),
        "literal_overlap_jaccard": _literal_jaccard(rows) if not incompetent else {},
        "functional_convergence": bool(h_eq),
        "physics_dependency": PHYSICS_DEPENDENCY,
        "claim_boundary": _claim_boundary(),
        "substrate": "supervised_residual_mlp_not_jepa_objective",
        "parent_005_status_preserved": "INCONCLUSIVE",
        "parent_004_status_preserved": "INCONCLUSIVE",
        "parent_003_status_preserved": "MODEL_INCOMPETENT",
        "parent_002_status_preserved": "INCONCLUSIVE",
        "parent_001_status_preserved": "MODEL_INCOMPETENT",
        "hard002_status_preserved": "NEGATIVE_RESULT",
        "ibd002_executed": False,
        "seed_97_retrospective_pass": False,
        "level3_authorized": False,
        "require_residual_units_by_fiat": False,
        "nomenclature": "learned WM, not JEPA objective",
        "mechanism_object": "M=(C,V_up,V_down,E,I)",
    }


def _trained_row(
    seed: int, stage: str, train_steps: int
) -> tuple[wm005.PathAwareActionDeltaPredictor | None, dict[str, Any]]:
    model, row = _fit_seed(seed, stage, train_steps)
    if row["status"] == "MODEL_INCOMPETENT":
        return None, row
    return model, row


def _authorize_previous(path: str, current_rung: int) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("previous artifact is not LEARNED-WM-CONDITIONAL-MEDIATOR-006")
    if payload.get("stage") != "development":
        raise ValueError("previous artifact must be development")
    if payload.get("status") != "MODEL_INCOMPETENT":
        raise ValueError("cannot climb: previous rung was not competence-failed")
    if payload.get("threshold_digest") != threshold_digest():
        raise ValueError("previous threshold digest mismatch")
    if payload.get("source_digest") != source_digest():
        raise ValueError("previous source digest mismatch")
    previous = int(payload.get("train_steps"))
    if previous not in LADDER_RUNGS:
        raise ValueError("previous rung is not on the ladder")
    expected = LADDER_RUNGS[LADDER_RUNGS.index(previous) + 1] if previous != LADDER_RUNGS[-1] else None
    if expected != int(current_rung):
        raise ValueError(f"rung {current_rung} does not follow {previous}")


def run_learned_development(train_steps: int = 800, *, previous_path: str | None = None) -> dict[str, Any]:
    _require_execution_authorized()
    if int(train_steps) not in LADDER_RUNGS:
        raise ValueError(f"rung {train_steps} is not on the frozen ladder")
    if int(train_steps) == LADDER_RUNGS[0]:
        if previous_path:
            raise ValueError("rung 800 must not receive --require-previous")
    else:
        if not previous_path:
            raise ValueError("climbing the ladder requires previous_path")
        _authorize_previous(previous_path, int(train_steps))
    fitted = [_trained_row(seed, "development", train_steps) for seed in DEVELOPMENT_SEEDS]
    rows = [row for _, row in fitted]
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        return _aggregate(rows, "development", train_steps)
    interpreted: list[dict[str, Any]] = []
    for seed, (model, row) in zip(DEVELOPMENT_SEEDS, fitted, strict=True):
        if model is None:
            raise RuntimeError("competent rung missing model")
        train_s, train_a, _ = parent._transitions(seed * 1000 + 61, 256)
        dev_s, dev_a, _ = parent._transitions(seed * 1000 + 67, 64)
        mechanism = _run_mechanism(model, seed, "development", train_s, train_a, dev_s, dev_a)
        merged = dict(row)
        merged.update(mechanism)
        merged["circuit_search_ran"] = True
        merged["evidence_level"] = "Causal effect" if mechanism["status"] in LEVEL2B_STATUSES else "None"
        interpreted.append(merged)
    return _aggregate(interpreted, "development", train_steps)


def _authorize_confirmation(dev_path: str) -> int:
    payload = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("development artifact is not LEARNED-WM-CONDITIONAL-MEDIATOR-006")
    if payload.get("stage") != "development":
        raise ValueError("require-development file is not development")
    if payload.get("status") != "CONDITIONAL_DOWNSTREAM_MEDIATION_PASSED":
        raise ValueError("confirmation closed: development did not pass shared downstream class")
    if payload.get("all_seeds_passed") is not True:
        raise ValueError("confirmation closed: development all_seeds_passed is not true")
    if payload.get("shared_downstream_class") not in LEVEL2B_CLASSES:
        raise ValueError("confirmation closed: no shared Level-2B class")
    if payload.get("seeds") != list(DEVELOPMENT_SEEDS):
        raise ValueError("development seeds do not match freeze")
    if payload.get("threshold_digest") != threshold_digest():
        raise ValueError("development threshold digest mismatch")
    if payload.get("source_digest") != source_digest():
        raise ValueError("development source digest mismatch")
    if payload.get("level3_authorized") is True:
        raise ValueError("006 must not authorize Level 3")
    selected = int(payload["selected_rung"])
    if selected not in LADDER_RUNGS:
        raise ValueError("selected rung is not on the frozen ladder")
    sidecar = Path(dev_path).with_suffix(".provenance.json")
    if sidecar.exists():
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if prov.get("stage") != "development":
            raise ValueError("development provenance stage mismatch")
        if prov.get("seed") != DEVELOPMENT_SEEDS[0]:
            raise ValueError("development provenance seed must be 173")
        if "--stage confirmation" in str(prov.get("command", "")):
            raise ValueError("development provenance fuses confirmation")
    return selected


def run_confirmation(development_path: str) -> dict[str, Any]:
    _require_execution_authorized()
    train_steps = _authorize_confirmation(development_path)
    development = json.loads(Path(development_path).read_text(encoding="utf-8"))
    required_class = development.get("shared_downstream_class")
    fitted = [_trained_row(seed, "confirmation", train_steps) for seed in CONFIRMATION_SEEDS]
    rows = [row for _, row in fitted]
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        return _aggregate(rows, "confirmation", train_steps, required_shared_class=required_class)
    interpreted: list[dict[str, Any]] = []
    for seed, (model, row) in zip(CONFIRMATION_SEEDS, fitted, strict=True):
        if model is None:
            raise RuntimeError("competent confirmation seed missing model")
        train_s, train_a, _ = parent._transitions(seed * 1000 + 61, 256)
        dev_s, dev_a, _ = parent._transitions(seed * 1000 + 67, 64)
        mechanism = _run_mechanism(model, seed, "confirmation", train_s, train_a, dev_s, dev_a)
        merged = dict(row)
        merged.update(mechanism)
        merged["circuit_search_ran"] = True
        merged["evidence_level"] = "Causal effect" if mechanism["status"] in LEVEL2B_STATUSES else "None"
        interpreted.append(merged)
    return _aggregate(interpreted, "confirmation", train_steps, required_shared_class=required_class)


def _relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def main() -> int:
    import argparse

    from causal_workspace_jepa.common.provenance import collect_provenance, stage_cli_command, write_provenance

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--rung", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--require-previous", default="")
    parser.add_argument("--require-development", default="")
    args = parser.parse_args()
    extra = ""
    if args.stage == "development":
        if int(args.rung) not in LADDER_RUNGS:
            raise ValueError("development requires a frozen --rung")
        if int(args.rung) != LADDER_RUNGS[0] and not args.require_previous:
            raise ValueError("climbing the ladder requires --require-previous")
        extra = f"--rung {int(args.rung)}"
        if args.require_previous:
            extra += f" --require-previous {_relative_posix(Path(args.require_previous))}"
        payload = run_learned_development(int(args.rung), previous_path=args.require_previous or None)
        require_dev = None
    else:
        if not args.require_development:
            raise ValueError("confirmation requires --require-development")
        require_dev = _relative_posix(Path(args.require_development))
        payload = run_confirmation(args.require_development)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        rel = _relative_posix(path)
        command = stage_cli_command(
            MODULE,
            args.stage,
            rel,
            require_development=require_dev,
            extra_args=extra,
        )
        provenance = collect_provenance(
            command,
            "configs/resource/cpu_vps.yaml",
            seed=int(payload["seeds"][0]),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        write_provenance(
            path.with_suffix(".provenance.json"),
            provenance,
            extra={
                "experiment_id": EXPERIMENT_ID,
                "stage": args.stage,
                "command_stage": args.stage,
                "train_steps": payload.get("train_steps"),
                "seeds": list(payload["seeds"]),
                "metrics": rel,
                "threshold_digest": payload["threshold_digest"],
                "source_digest": payload.get("source_digest"),
            },
        )
    print(text)
    return 0 if payload["all_seeds_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

