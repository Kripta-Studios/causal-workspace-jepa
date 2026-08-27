"""CRCT-LEARNED-WM-ACTION-DELTA-005: edge identifiability after 004 Level-2 stop.

Does not mutate 001/002/003/004. Seed 97 is not Level 3. Seed 101 redundancy
is not promoted. Not a JEPA-objective experiment.
004 G_skip does not assign 005 path class. Residual messages are cached.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as parent
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta as wm002
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta_003 as wm003
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta_004 as wm004
from pathlib import Path

EXPERIMENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-005"
MODULE = "causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_005"
PARENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-004"
CONFIG_PATH = Path("configs/experiments/crct_learned_wm_action_delta_v5.json")

LADDER_RUNGS = (800, 2000, 5000)
DEVELOPMENT_SEEDS = (109, 113, 127)
CONFIRMATION_SEEDS = (1103, 1109, 1117)

FORBIDDEN_SEEDS = frozenset(wm004.FORBIDDEN_SEEDS) | {97, 101, 107, 1063, 1069, 1087}

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
    "ladder_rungs": list(LADDER_RUNGS),
}

LEVEL3_STATUSES = frozenset(
    {
        "DIRECT_PATH_MECHANISM_PASSED",
        "DISTRIBUTED_F1_PATH_MECHANISM_PASSED",
    }
)
LEVEL3_CLASSES = frozenset({"DIRECT", "DISTRIBUTED_F1"})
LEVEL2_STATUSES = frozenset(
    {"MEDIATOR_FOUND_PATH_UNRESOLVED", "INFORMATION_GATEWAY_ONLY", "REDUNDANT_ROUTES"}
)


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_digest() -> str:
    parts = [
        Path(parent.__file__).read_bytes(),
        Path(wm002.__file__).read_bytes(),
        Path(wm003.__file__).read_bytes(),
        Path(wm004.__file__).read_bytes(),
        Path(__file__).read_bytes(),
    ]
    return hashlib.sha256(b"\n".join(parts)).hexdigest()


def _require_execution_authorized() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("config experiment_id mismatch")
    if config.get("execution_authorized") is not True:
        raise ValueError("CRCT-LEARNED-WM-ACTION-DELTA-005 execution is not authorized")
    if config.get("status") == "DRAFT_NOT_PREREGISTERED":
        raise ValueError("CRCT-LEARNED-WM-ACTION-DELTA-005 is not frozen")


def _claim_boundary() -> str:
    return (
        "supervised residual-MLP PointMass world model only; not a JEPA objective; "
        "does not reinterpret 001/002/003/004; seed 97 is not a retrospective Level-3 pass; "
        "004 G_skip does not assign 005 path class; action-stem MSRS cannot be Level 3; "
        "INTERACTING is not Level 3; REDUNDANT_ROUTES is not Level 3; "
        "cached r2_P is not a Level-3 F2 edge; residual messages cached from A/P; "
        "does not alter HARD-002, IBD-002, or IBD-003"
    )


class PathAwareActionDeltaPredictor(wm003.PathAwareActionDeltaPredictor):
    """Same topology as 001–004. Exposes cached residual messages r1/r2."""

    def forward_path(
        self,
        state: Tensor,
        action: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
        path_holds: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        over = overrides or {}
        holds = path_holds or {}
        z = torch.tanh(self.enc2(torch.tanh(self.enc1(state))))
        e = torch.tanh(self.act1(action)) @ self.q_act
        e = parent._override_units(e, "act", over)
        h0_mix = torch.tanh(self.mix(torch.cat([z, e], dim=-1)))
        h0_branch = holds["h0_branch"] if "h0_branch" in holds else h0_mix
        skip1 = holds["skip1"] if "skip1" in holds else h0_mix
        hid1 = torch.tanh(self.b1_w1(h0_branch)) @ self.q_b1
        hid1 = parent._override_units(hid1, "b1", over)
        if "hid1" in holds:
            hid1 = holds["hid1"]
        r1 = self.b1_w2(hid1)
        h1 = skip1 + r1
        if "h1" in holds:
            h1 = holds["h1"]
        hid2 = torch.tanh(self.b2_w1(h1)) @ self.q_b2
        hid2 = parent._override_units(hid2, "b2", over)
        if "hid2" in holds:
            hid2 = holds["hid2"]
        r2 = self.b2_w2(hid2)
        skip2 = holds["skip2"] if "skip2" in holds else h1
        h2 = skip2 + r2
        sites = {f"act_{i}": e[:, i] for i in range(parent.HIDDEN)}
        sites.update({f"b1_{i}": hid1[:, i] for i in range(parent.HIDDEN)})
        sites.update({f"b2_{i}": hid2[:, i] for i in range(parent.HIDDEN)})
        paths = {
            "h0": h0_mix,
            "hid1": hid1,
            "r1": r1,
            "h1": h1,
            "hid2": hid2,
            "r2": r2,
            "s1": skip1,
            "s2": skip2,
            "h2": h2,
        }
        return self.out(h2), sites, paths


def compose_output(
    model: Any,
    s1: Tensor,
    r1: Tensor,
    r2: Tensor,
    s2: Tensor | None = None,
) -> Tensor:
    h1 = s1 + r1
    skip2 = h1 if s2 is None else s2
    return model.out(skip2 + r2)


def classify_edge_path(gaps: Mapping[str, float], bar: float | None = None) -> str | None:
    """Block-1 skip vs F1 message vs cached F2 message. Does not use 004 G_skip.

    ``g_skip2`` is recorded as an alias of ``g_both1`` (h1_P + r2_A) and is
    **not** an independent skip2 edge. DISTRIBUTED_F1 means the F1 message
    carries V and then rides the additive stream; it is not F1 exclusive of
    skip2. Action-stem V cannot be Level 3 in ``adjudicate_seed``.
    """

    thresh = FROZEN_THRESHOLDS["counterfactual_gap_min"] if bar is None else bar
    g_v = float(gaps["g_v"])
    if g_v < thresh:
        return None
    skip1 = float(gaps["g_skip1"]) >= thresh
    res1 = float(gaps["g_res1"]) >= thresh
    res2 = float(gaps["g_res2"]) >= thresh
    n_indep = int(skip1) + int(res1) + int(res2)
    if n_indep >= 2:
        return "REDUNDANT_ROUTES"
    if skip1 and not res1 and not res2:
        return "DIRECT"
    if res1 and not skip1 and not res2:
        return "DISTRIBUTED_F1"
    if res2 and not skip1 and not res1:
        return "F2_CACHED_UNIDENTIFIED"
    return "INTERACTING"


class PlantedRoutePredictor(torch.nn.Module):
    """Deterministic known-route plants for instrument tests. Not scientific evidence."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        allowed = {"direct", "f1", "f2", "redundant_skip_f1", "interacting", "f1_copied_by_f2"}
        if mode not in allowed:
            raise ValueError(f"unknown plant {mode}")
        self.mode = mode
        self.out = torch.nn.Linear(6, 4, bias=False)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.weight[2, 0] = 1.0

    def forward_path(
        self,
        state: Tensor,
        action: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
        path_holds: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        del state, path_holds
        over = overrides or {}
        n = int(action.shape[0])
        e = torch.zeros(n, 6, device=action.device, dtype=action.dtype)
        e[:, 0] = action[:, 0]
        e = parent._override_units(e, "act", over)
        h0 = torch.zeros(n, 6, device=action.device, dtype=action.dtype)
        r1 = torch.zeros(n, 6, device=action.device, dtype=action.dtype)
        r2 = torch.zeros(n, 6, device=action.device, dtype=action.dtype)
        sig = e[:, 0]
        if self.mode == "direct":
            h0[:, 0] = sig
        elif self.mode == "f1":
            r1[:, 0] = sig
        elif self.mode == "f1_copied_by_f2":
            r1[:, 1] = sig
        elif self.mode == "f2":
            h0[:, 1] = sig
        elif self.mode == "redundant_skip_f1":
            h0[:, 0] = sig
            r1[:, 0] = sig
        else:
            h0[:, 0] = 3.0 * sig
            r1[:, 0] = -2.0 * sig
        r1 = parent._override_units(r1, "b1", over)
        h1 = h0 + r1
        if self.mode in {"f2", "f1_copied_by_f2"}:
            r2[:, 0] = h1[:, 1]
        r2 = parent._override_units(r2, "b2", over)
        h2 = h1 + r2
        sites = {f"act_{i}": e[:, i] for i in range(parent.HIDDEN)}
        sites.update({f"b1_{i}": r1[:, i] for i in range(parent.HIDDEN)})
        sites.update({f"b2_{i}": r2[:, i] for i in range(parent.HIDDEN)})
        paths = {"h0": h0, "r1": r1, "h1": h1, "r2": r2, "h2": h2, "hid1": r1, "hid2": r2}
        return self.out(h2), sites, paths


def _level_for(status: str) -> int:
    if status in LEVEL3_STATUSES:
        return 3
    if status in LEVEL2_STATUSES:
        return 2
    return 0


def _status_for_class(path_class: str) -> str:
    mapping = {
        "DIRECT": "DIRECT_PATH_MECHANISM_PASSED",
        "DISTRIBUTED_F1": "DISTRIBUTED_F1_PATH_MECHANISM_PASSED",
        "F2_CACHED_UNIDENTIFIED": "MEDIATOR_FOUND_PATH_UNRESOLVED",
        "REDUNDANT_ROUTES": "REDUNDANT_ROUTES",
        "INTERACTING": "MEDIATOR_FOUND_PATH_UNRESOLVED",
    }
    return mapping[path_class]


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
    action_only: bool,
    edge_control_failed: bool,
    stage_b_ran: bool,
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
    elif not stage_b_ran or path_class is None:
        status = "INFORMATION_GATEWAY_ONLY" if action_only else "MEDIATOR_FOUND_PATH_UNRESOLVED"
    elif action_only:
        status = "INFORMATION_GATEWAY_ONLY"
    elif path_class == "INTERACTING":
        status = "MEDIATOR_FOUND_PATH_UNRESOLVED"
    elif path_class == "REDUNDANT_ROUTES":
        status = "REDUNDANT_ROUTES"
    elif path_class == "F2_CACHED_UNIDENTIFIED":
        status = "MEDIATOR_FOUND_PATH_UNRESOLVED"
    elif edge_control_failed:
        status = "EDGE_CONTROL_FAILED"
    elif g_path != path_class:
        status = "PATH_CLASS_GAUGE_UNSTABLE"
    elif path_class in LEVEL3_CLASSES:
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
) -> tuple[PathAwareActionDeltaPredictor, list[float], str]:
    if int(seed) in FORBIDDEN_SEEDS:
        raise ValueError(f"forbidden seed {seed}")
    model = PathAwareActionDeltaPredictor(seed)
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
    model: PathAwareActionDeltaPredictor,
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


def edge_factorial(
    model: PathAwareActionDeltaPredictor,
    coalition: Sequence[str],
    seed: int,
) -> dict[str, float]:
    state, action_a, action_b = wm003._cf_pairs(seed)
    with torch.no_grad():
        y_a, _sites_a, path_a = model.forward_path(state, action_a, None, None)
        y_b, sites_b, _ = model.forward_path(state, action_b, None, None)
        patch = {name: sites_b[name] for name in coalition}
        y_p, _sites_p, path_p = model.forward_path(state, action_a, patch, None)
        y_aa = compose_output(model, path_a["h0"], path_a["r1"], path_a["r2"])
        y_skip1 = compose_output(model, path_p["h0"], path_a["r1"], path_a["r2"])
        y_res1 = compose_output(model, path_a["h0"], path_p["r1"], path_a["r2"])
        y_both1 = compose_output(model, path_p["h0"], path_p["r1"], path_a["r2"])
        y_skip2 = compose_output(model, path_p["h0"], path_p["r1"], path_a["r2"], s2=path_p["h1"])
        y_res2 = compose_output(model, path_a["h0"], path_a["r1"], path_p["r2"], s2=path_a["h1"])
        y_both2 = compose_output(model, path_p["h0"], path_p["r1"], path_p["r2"], s2=path_p["h1"])
        y_f1f2 = compose_output(model, path_a["h0"], path_p["r1"], path_p["r2"])
        y_ctrl_r2 = compose_output(model, path_a["h0"], path_a["r1"], path_p["r2"])
        y_ctrl_r1 = compose_output(model, path_a["h0"], path_p["r1"], path_a["r2"])
        perm = torch.randperm(path_p["r1"].shape[0])
        y_shuf = compose_output(model, path_a["h0"], path_p["r1"][perm], path_a["r2"])
        i1 = (y_both1 - y_skip1 - y_res1 + y_aa)[:, parent.CHANNELS.index(parent.PRIMARY)]
    return {
        "g_v": wm003._gap_closed(y_a, y_b, y_p, parent.PRIMARY),
        "g_skip1": wm003._gap_closed(y_a, y_b, y_skip1, parent.PRIMARY),
        "g_res1": wm003._gap_closed(y_a, y_b, y_res1, parent.PRIMARY),
        "g_both1": wm003._gap_closed(y_a, y_b, y_both1, parent.PRIMARY),
        "g_skip2": wm003._gap_closed(y_a, y_b, y_skip2, parent.PRIMARY),
        "g_skip2_is_alias_of_both1": float(torch.mean((y_skip2 - y_both1).square()).item()),
        "g_res2": wm003._gap_closed(y_a, y_b, y_res2, parent.PRIMARY),
        "g_both2": wm003._gap_closed(y_a, y_b, y_both2, parent.PRIMARY),
        "g_f1f2": wm003._gap_closed(y_a, y_b, y_f1f2, parent.PRIMARY),
        "g_control_r2_only": wm003._gap_closed(y_a, y_b, y_ctrl_r2, parent.PRIMARY),
        "g_control_r1_only": wm003._gap_closed(y_a, y_b, y_ctrl_r1, parent.PRIMARY),
        "g_control_shuffled_r1": wm003._gap_closed(y_a, y_b, y_shuf, parent.PRIMARY),
        "reconstruct_a_mse": float(torch.mean((y_aa - y_a).square()).item()),
        "reconstruct_p_mse": float(torch.mean((y_both2 - y_p).square()).item()),
        "interaction_block1_dvx_median": float(i1.median().item()),
    }


def _edge_control_failed(path_class: str | None, gaps: Mapping[str, float]) -> bool:
    bar = FROZEN_THRESHOLDS["counterfactual_gap_min"]
    if path_class not in LEVEL3_CLASSES:
        return False
    return float(gaps["g_control_shuffled_r1"]) >= bar


def _run_mechanism(
    model: PathAwareActionDeltaPredictor,
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
    plus_one_p, control_sets, random_sufficient = parent._random_controls(
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
    gaps = (
        edge_factorial(model, coalition, seed)
        if coalition
        else {
            "g_v": 0.0,
            "g_skip1": 0.0,
            "g_res1": 0.0,
            "g_both1": 0.0,
            "g_skip2": 0.0,
            "g_res2": 0.0,
            "g_both2": 0.0,
            "g_f1f2": 0.0,
            "g_control_r2_only": 0.0,
            "g_control_r1_only": 0.0,
            "g_control_shuffled_r1": 0.0,
            "reconstruct_a_mse": 0.0,
            "reconstruct_p_mse": 0.0,
            "interaction_block1_dvx_median": 0.0,
        }
    )
    stage_b_ran = bool(stage_a_ok and float(gaps["g_v"]) >= FROZEN_THRESHOLDS["counterfactual_gap_min"])
    path_class = classify_edge_path(gaps) if stage_b_ran else None
    edge_control_failed = _edge_control_failed(path_class, gaps) if stage_b_ran else False
    legacy = (
        wm003.counterfactual_paths(model, coalition, seed)
        if coalition
        else {"full": 0.0, "skip": 0.0, "residual": 0.0}
    )
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
    g_gaps = edge_factorial(gauged, g_coal, seed) if g_coal else {"g_v": 0.0, "g_skip1": 0.0, "g_res1": 0.0, "g_skip2": 0.0, "g_res2": 0.0, "g_f1f2": 0.0}
    g_stage_b = bool(
        g_coal
        and g_suff <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        and g_nec >= FROZEN_THRESHOLDS["necessity_nmse_min"]
        and float(g_gaps["g_v"]) >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
    )
    g_path = classify_edge_path(g_gaps) if g_stage_b else None
    status, level = adjudicate_seed(
        coalition=coalition,
        sufficiency_dvx=metrics["sufficiency"][parent.PRIMARY],
        drop_still_sufficient=drop_still_sufficient,
        necessity_dvx=metrics["necessity"][parent.PRIMARY],
        spec_failed=spec_failed,
        random_sufficient=int(random_sufficient),
        act_random_sufficient=int(act_random_sufficient),
        g_v=float(gaps["g_v"]),
        gauge_fn=gauge_fn,
        g_suff=float(g_suff),
        g_nec=float(g_nec),
        g_path=g_path,
        path_class=path_class,
        action_only=action_only,
        edge_control_failed=bool(edge_control_failed),
        stage_b_ran=bool(stage_b_ran),
    )
    return {
        "msrs": list(coalition),
        "mediator_v": list(coalition),
        "causal_edges": {
            "E_skip1": ["s1"],
            "E_F1_message": ["r1"],
            "r2_P_cached_diagnostic": ["r2"],
        },
        "skip2_note": "g_skip2 aliases g_both1; not an independent edge",
        "residual_message_semantics": "cached_from_reference_forwards_A_and_P",
        "pre_prune_circuit": list(raw_coalition),
        "sufficiency": metrics["sufficiency"],
        "necessity": metrics["necessity"],
        "specificity_independent": spec,
        "minimality_drop": metrics["minimality_drop"],
        "random_plus_one_p": float(plus_one_p),
        "random_control_sufficient_count": int(random_sufficient),
        "act_random_control_sufficient_count": int(act_random_sufficient),
        "g_v": float(gaps["g_v"]),
        "g_full_diagnostic_004": float(legacy["full"]),
        "g_skip_diagnostic_004": float(legacy["skip"]),
        "g_res_diagnostic_004": float(legacy["residual"]),
        "edge_gaps": {key: float(gaps[key]) for key in gaps},
        "path_class": path_class,
        "stage_b_ran": bool(stage_b_ran),
        "edge_control_failed": bool(edge_control_failed),
        "action_stem_msrs_cannot_be_level3": True,
        "gauge_function_mse": gauge_fn,
        "gauge_msrs": list(g_coal),
        "gauge_sufficiency_dvx": float(g_suff),
        "gauge_necessity_dvx": float(g_nec),
        "gauge_path_class": g_path,
        "literal_jaccard_vs_gauge": (
            len(set(coalition) & set(g_coal)) / max(len(set(coalition) | set(g_coal)), 1)
        ),
        "status": status,
        "level": level,
        "action_embedding_only": action_only,
        "004_g_skip_assigns_path_class": False,
        "seed_97_retrospective_pass": False,
        "mechanism_object": "M=(V,E,I)_cached_edge_factorial",
    }


def _fit_seed(
    seed: int, stage: str, train_steps: int
) -> tuple[PathAwareActionDeltaPredictor, dict[str, Any]]:
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
        "seed_59_retrospective_pass": False,
    }
    if not bundle["passed"]:
        row["status"] = "MODEL_INCOMPETENT"
        row["evidence_level"] = "None"
    else:
        row["status"] = "COMPETENT_NOT_INTERPRETED"
        row["evidence_level"] = "None"
    return model, row


def _literal_jaccard(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    recovered = {row["seed"]: set(row.get("msrs") or []) for row in rows if row.get("msrs")}
    out: dict[str, float] = {}
    seeds = sorted(recovered)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            union = recovered[a] | recovered[b]
            out[f"{a}_{b}"] = len(recovered[a] & recovered[b]) / max(len(union), 1)
    return out


def _shared_level3_class(rows: Sequence[Mapping[str, Any]]) -> str | None:
    classes = {row.get("path_class") for row in rows}
    if len(classes) == 1:
        only = next(iter(classes))
        if only in LEVEL3_CLASSES:
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
    required_shared_path_class: str | None = None,
) -> dict[str, Any]:
    incompetent = [row for row in rows if row["status"] == "MODEL_INCOMPETENT"]
    statuses = {row["status"] for row in rows}
    shared = _shared_level3_class(rows) if not incompetent else None
    if stage == "confirmation" and incompetent:
        status = "MODEL_INCOMPETENT_CONFIRMATION"
    elif incompetent:
        status = "MODEL_INCOMPETENT"
    elif statuses <= LEVEL3_STATUSES and len(statuses) == 1 and shared is not None:
        status = "PATH_MECHANISM_RECOVERY_PASSED"
    elif statuses == {"REDUNDANT_ROUTES"}:
        status = "REDUNDANT_ROUTES"
    elif len(statuses) == 1:
        status = next(iter(statuses))
    else:
        status = "INCONCLUSIVE"
    passed = status == "PATH_MECHANISM_RECOVERY_PASSED"
    if (
        stage == "confirmation"
        and required_shared_path_class is not None
        and passed
        and shared != required_shared_path_class
    ):
        status = "CONFIRMATION_PATH_CLASS_MISMATCH"
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
        "shared_path_class": shared if passed else None,
        "required_shared_path_class": required_shared_path_class,
        "h_equivalent": h_eq,
        "status": status,
        "evidence_level": "Causal effect" if passed else "None",
        "rows": list(rows),
        "literal_overlap_jaccard": _literal_jaccard(rows) if not incompetent else {},
        "functional_convergence": bool(h_eq),
        "physics_dependency": PHYSICS_DEPENDENCY,
        "claim_boundary": _claim_boundary(),
        "substrate": "supervised_residual_mlp_not_jepa_objective",
        "parent_004_status_preserved": "INCONCLUSIVE",
        "parent_003_status_preserved": "MODEL_INCOMPETENT",
        "parent_002_status_preserved": "INCONCLUSIVE",
        "parent_001_status_preserved": "MODEL_INCOMPETENT",
        "hard002_status_preserved": "NEGATIVE_RESULT",
        "ibd002_executed": False,
        "seed_97_retrospective_pass": False,
        "nomenclature": "learned WM, not JEPA objective",
        "004_g_skip_assigns_path_class": False,
    }


def _trained_row(
    seed: int, stage: str, train_steps: int
) -> tuple[PathAwareActionDeltaPredictor | None, dict[str, Any]]:
    model, row = _fit_seed(seed, stage, train_steps)
    if row["status"] == "MODEL_INCOMPETENT":
        return None, row
    return model, row


def _authorize_previous(path: str, current_rung: int) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("previous artifact is not LEARNED-WM-ACTION-DELTA-005")
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


def run_development_rung(train_steps: int, *, previous_path: str | None = None) -> dict[str, Any]:
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
        merged["evidence_level"] = "Causal effect" if mechanism["status"] in LEVEL3_STATUSES else "None"
        interpreted.append(merged)
    return _aggregate(interpreted, "development", train_steps)


def _authorize_confirmation(dev_path: str) -> int:
    payload = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("development artifact is not LEARNED-WM-ACTION-DELTA-005")
    if payload.get("stage") != "development":
        raise ValueError("require-development file is not development")
    if payload.get("status") != "PATH_MECHANISM_RECOVERY_PASSED":
        raise ValueError("confirmation closed: development did not pass shared path class")
    if payload.get("all_seeds_passed") is not True:
        raise ValueError("confirmation closed: development all_seeds_passed is not true")
    if payload.get("shared_path_class") not in LEVEL3_CLASSES:
        raise ValueError("confirmation closed: no shared Level-3 path class")
    if payload.get("seeds") != list(DEVELOPMENT_SEEDS):
        raise ValueError("development seeds do not match freeze")
    if payload.get("threshold_digest") != threshold_digest():
        raise ValueError("development threshold digest mismatch")
    if payload.get("source_digest") != source_digest():
        raise ValueError("development source digest mismatch")
    selected = int(payload["selected_rung"])
    if selected not in LADDER_RUNGS:
        raise ValueError("selected rung is not on the frozen ladder")
    sidecar = Path(dev_path).with_suffix(".provenance.json")
    if sidecar.exists():
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if prov.get("stage") != "development":
            raise ValueError("development provenance stage mismatch")
        if prov.get("seed") != DEVELOPMENT_SEEDS[0]:
            raise ValueError("development provenance seed must be 109")
        if "--stage confirmation" in str(prov.get("command", "")):
            raise ValueError("development provenance fuses confirmation")
    return selected


def run_confirmation(development_path: str) -> dict[str, Any]:
    _require_execution_authorized()
    train_steps = _authorize_confirmation(development_path)
    development = json.loads(Path(development_path).read_text(encoding="utf-8"))
    required_class = development.get("shared_path_class")
    fitted = [_trained_row(seed, "confirmation", train_steps) for seed in CONFIRMATION_SEEDS]
    rows = [row for _, row in fitted]
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        return _aggregate(rows, "confirmation", train_steps, required_shared_path_class=required_class)
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
        merged["evidence_level"] = "Causal effect" if mechanism["status"] in LEVEL3_STATUSES else "None"
        interpreted.append(merged)
    return _aggregate(interpreted, "confirmation", train_steps, required_shared_path_class=required_class)


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
        payload = run_development_rung(int(args.rung), previous_path=args.require_previous or None)
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
