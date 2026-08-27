"""CRCT-LEARNED-WM-ACTION-DELTA-003: gateway vs pathway for learned ax→Δvx.

Does not mutate 001/002. Not a JEPA-objective experiment.
Seed 59 is not a retrospective pass. ARCHITECTURE_CUTSET is not an automatic fail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as parent
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta as wm002

EXPERIMENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-003"
MODULE = "causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_003"
PARENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-002"
TRAIN_STEPS = 800
DEVELOPMENT_SEEDS = (79, 83, 89)
CONFIRMATION_SEEDS = (1049, 1051, 1061)

FORBIDDEN_SEEDS = frozenset(wm002.FORBIDDEN_SEEDS) | {
    59,
    71,
    73,
    1031,
    1033,
    1039,
}

# Direct D / independent 0. Downstream of ax is Δx (not a negative control).
PHYSICS_DEPENDENCY = {
    "dx": {"ax": "D", "ay": "0", "vx": "D", "vy": "0"},
    "dy": {"ax": "0", "ay": "D", "vx": "0", "vy": "D"},
    "dvx": {"ax": "D", "ay": "0", "vx": "D", "vy": "0"},
    "dvy": {"ax": "0", "ay": "D", "vx": "0", "vy": "D"},
}
INDEPENDENT_CONTROLS = ("dvy", "dy")
DOWNSTREAM_OF_AX = ("dx",)

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
    "train_steps": TRAIN_STEPS,
}


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_digest() -> str:
    parts = [
        Path(parent.__file__).read_bytes(),
        Path(wm002.__file__).read_bytes(),
        Path(__file__).read_bytes(),
    ]
    return hashlib.sha256(b"\n".join(parts)).hexdigest()


def _claim_boundary() -> str:
    return (
        "supervised residual-MLP PointMass world model only; not a JEPA objective; "
        "does not reinterpret 002; seed 59 is not a retrospective pass; "
        "does not alter HARD-002, IBD-002, or IBD-003"
    )


class PathAwareActionDeltaPredictor(parent.ActionDeltaPredictor):
    """Same weights as 001/002, with residual/skip path holds for 003."""

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
        h1 = skip1 + self.b1_w2(hid1)
        if "h1" in holds:
            h1 = holds["h1"]
        hid2 = torch.tanh(self.b2_w1(h1)) @ self.q_b2
        hid2 = parent._override_units(hid2, "b2", over)
        if "hid2" in holds:
            hid2 = holds["hid2"]
        skip2 = holds["skip2"] if "skip2" in holds else h1
        h2 = skip2 + self.b2_w2(hid2)
        sites = {f"act_{i}": e[:, i] for i in range(parent.HIDDEN)}
        sites.update({f"b1_{i}": hid1[:, i] for i in range(parent.HIDDEN)})
        sites.update({f"b2_{i}": hid2[:, i] for i in range(parent.HIDDEN)})
        paths = {"h0": h0_mix, "hid1": hid1, "h1": h1, "hid2": hid2, "h2": h2}
        return self.out(h2), sites, paths

    def forward_intervene(
        self,
        state: Tensor,
        action: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        delta, sites, _ = self.forward_path(state, action, overrides, None)
        return delta, sites


def train_model(
    seed: int,
    state: Tensor,
    action: Tensor,
    delta: Tensor,
    steps: int = TRAIN_STEPS,
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


def _probe_r2(feature: Tensor, target: Tensor) -> float:
    x = torch.stack([feature, torch.ones_like(feature)], dim=1)
    sol = torch.linalg.lstsq(x, target.unsqueeze(-1)).solution[:, 0]
    pred = x @ sol
    ss_res = torch.sum((target - pred).square())
    ss_tot = torch.sum((target - target.mean()).square()).clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).item())


def information_r2(
    model: PathAwareActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        _, sites = model.forward_intervene(state, action)
    ax = action[:, 0].detach()
    return {name: _probe_r2(sites[name].detach(), ax) for name in parent.SITE_NAMES}


def _cf_pairs(seed: int) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(int(seed) + 409)
    n = int(FROZEN_THRESHOLDS["counterfactual_pairs"])
    state = torch.empty(n, 4)
    state[:, 0:2] = (torch.rand(n, 2, generator=gen) * 2.0) - 1.0
    state[:, 2:4] = (torch.rand(n, 2, generator=gen) * 0.4) - 0.2
    ax = (torch.rand(n, 1, generator=gen) * 2.0) - 1.0
    ay = (torch.rand(n, 1, generator=gen) * 2.0) - 1.0
    ax2 = torch.clamp(ax + 0.7, -1.0, 1.0)
    same = (ax2 - ax).abs() < 1e-6
    ax2 = torch.where(same, torch.clamp(ax - 0.7, -1.0, 1.0), ax2)
    action_a = torch.cat([ax, ay], dim=-1)
    action_b = torch.cat([ax2, ay], dim=-1)
    return state, action_a, action_b


def _gap_closed(y_a: Tensor, y_b: Tensor, y_p: Tensor, channel: str) -> float:
    t = parent.CHANNELS.index(channel)
    base = (y_b[:, t] - y_a[:, t]).square()
    after = (y_b[:, t] - y_p[:, t]).square()
    closed = 1.0 - after / base.clamp_min(1e-12)
    return float(closed.median().item())


def counterfactual_paths(
    model: PathAwareActionDeltaPredictor,
    coalition: Sequence[str],
    seed: int,
) -> dict[str, float]:
    state, action_a, action_b = _cf_pairs(seed)
    with torch.no_grad():
        y_a, _sites_a, path_a = model.forward_path(state, action_a, None, None)
        y_b, sites_b, _ = model.forward_path(state, action_b, None, None)
        patch = {name: sites_b[name] for name in coalition}
        y_full, _, _ = model.forward_path(state, action_a, patch, None)
        y_skip, _, _ = model.forward_path(
            state,
            action_a,
            patch,
            {"hid1": path_a["hid1"], "hid2": path_a["hid2"]},
        )
        y_res, _, _ = model.forward_path(
            state,
            action_a,
            patch,
            {"skip1": path_a["h0"]},
        )
    return {
        "full": _gap_closed(y_a, y_b, y_full, parent.PRIMARY),
        "skip": _gap_closed(y_a, y_b, y_skip, parent.PRIMARY),
        "residual": _gap_closed(y_a, y_b, y_res, parent.PRIMARY),
    }


def classify_path(gaps: Mapping[str, float]) -> str | None:
    bar = FROZEN_THRESHOLDS["counterfactual_gap_min"]
    if float(gaps["full"]) < bar:
        return None
    skip_ok = float(gaps["skip"]) >= bar
    res_ok = float(gaps["residual"]) >= bar
    if skip_ok and not res_ok:
        return "DIRECT"
    if res_ok and not skip_ok:
        return "DISTRIBUTED"
    if skip_ok and res_ok:
        return "REDUNDANT_ROUTES"
    return "INTERACTING"


def _hypothesis(status: str, path_class: str | None) -> str:
    if status == "INFORMATION_GATEWAY_ONLY":
        return "H_GATEWAY"
    if status != "PATH_MECHANISM_RECOVERY_PASSED" or path_class is None:
        return "H_UNASSIGNED"
    if path_class in {"DIRECT", "REDUNDANT_ROUTES"}:
        return "H_DIRECT"
    if path_class in {"DISTRIBUTED", "INTERACTING"}:
        return "H_DISTRIBUTED"
    return "H_UNASSIGNED"


def _mean_r2(r2: Mapping[str, float], names: Sequence[str]) -> float:
    if not names:
        return 0.0
    return float(sum(r2[name] for name in names) / len(names))


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
    downstream_only, _ = parent.greedy_restore(
        model,
        dev_s,
        dev_a,
        means,
        target=parent.PRIMARY,
        forbidden=parent.ACT_SITES,
    )
    downstream_only = parent.prune_inclusion_minimal(
        model, dev_s, dev_a, means, downstream_only, target=parent.PRIMARY
    )
    eval_s, eval_a = dev_s, dev_a
    if stage == "confirmation":
        eval_s, eval_a, _ = parent._transitions(seed * 1000 + 71, 64)
    with torch.no_grad():
        y0, original = model.forward_intervene(eval_s, eval_a, None)
    metrics = parent._evaluate_coalition(model, eval_s, eval_a, means, coalition, y0)
    spec = {
        name: metrics["necessity"][parent.PRIMARY] / max(metrics["necessity"][name], 1e-6)
        for name in INDEPENDENT_CONTROLS
    }
    rng = random.Random(int(seed) * 8191 + 3)
    plus_one_p, _, random_sufficient = parent._random_controls(
        model, eval_s, eval_a, means, coalition, y0, rng
    )
    r2 = information_r2(model, train_s, train_a)
    k = max(len(coalition), 1)
    probe_topk = sorted(parent.SITE_NAMES, key=lambda name: r2[name], reverse=True)[:k]
    probe_eval = parent._evaluate_coalition(model, eval_s, eval_a, means, probe_topk, y0)
    probe_gap = parent.counterfactual_gap(
        model, means, probe_topk, seed=seed, vary="ax", target=parent.PRIMARY
    )
    gaps = counterfactual_paths(model, coalition, seed) if coalition else {"full": 0.0, "skip": 0.0, "residual": 0.0}
    path_class = classify_path(gaps) if coalition else None
    grads = parent._site_grads(
        model, eval_s[: min(64, eval_s.shape[0])], eval_a[: min(64, eval_a.shape[0])]
    )
    orig_small = {key: original[key][: min(64, original[key].shape[0])] for key in parent.SITE_NAMES}
    base = parent._baselines(orig_small, grads, k)
    base_eval = {}
    for label, sites in base.items():
        fill = parent._mean_except(means, sites, eval_s.shape[0])
        abl = parent._mean_on(means, sites, eval_s.shape[0])
        base_eval[label] = {
            "sites": sites,
            "sufficiency_dvx": parent._predict_nmse(model, eval_s, eval_a, fill, parent.PRIMARY, y0),
            "necessity_dvx": parent._predict_nmse(model, eval_s, eval_a, abl, parent.PRIMARY, y0),
        }
    gauged = copy.deepcopy(model)
    gauged.apply_hidden_gauge(parent._orthogonal(seed, 7), parent._orthogonal(seed, 11), parent._orthogonal(seed, 13))
    with torch.no_grad():
        gauge_fn = float(torch.mean((gauged(eval_s, eval_a) - y0).square()).item())
    g_means = parent._site_means(gauged, train_s, train_a)
    g_coal, _ = parent.greedy_restore(gauged, dev_s, dev_a, g_means, target=parent.PRIMARY)
    g_coal = parent.prune_inclusion_minimal(gauged, dev_s, dev_a, g_means, g_coal, target=parent.PRIMARY)
    with torch.no_grad():
        g_y0, _ = gauged.forward_intervene(eval_s, eval_a, None)
    g_rest = parent._mean_except(g_means, g_coal, eval_s.shape[0])
    g_suff = parent._predict_nmse(gauged, eval_s, eval_a, g_rest, parent.PRIMARY, g_y0)
    g_gaps = counterfactual_paths(gauged, g_coal, seed) if g_coal else {"full": 0.0, "skip": 0.0, "residual": 0.0}
    g_path = classify_path(g_gaps) if g_coal else None
    act_cut = parent._evaluate_coalition(model, eval_s, eval_a, means, list(parent.ACT_SITES), y0)
    cancel = parent._cancellation(model, eval_s, eval_a, means, coalition, y0)
    drop_still_sufficient = any(
        err <= FROZEN_THRESHOLDS["sufficiency_nmse_max"] for err in metrics["minimality_drop"].values()
    )
    spec_failed = any(spec[name] < FROZEN_THRESHOLDS["specificity_ratio_min"] for name in INDEPENDENT_CONTROLS)
    msrs_r2 = _mean_r2(r2, coalition)
    probe_r2 = _mean_r2(r2, probe_topk)
    if not coalition:
        status = "LOCALIZATION_FAILED"
    elif metrics["sufficiency"][parent.PRIMARY] > FROZEN_THRESHOLDS["sufficiency_nmse_max"]:
        status = "SUFFICIENCY_FAILED"
    elif drop_still_sufficient:
        status = "MINIMALITY_FAILED"
    elif metrics["necessity"][parent.PRIMARY] < FROZEN_THRESHOLDS["necessity_nmse_min"]:
        status = "NECESSITY_FAILED"
    elif spec_failed:
        status = "SPECIFICITY_FAILED"
    elif random_sufficient > FROZEN_THRESHOLDS["random_control_sufficient_max"]:
        status = "INCONCLUSIVE"
    elif gaps["full"] < FROZEN_THRESHOLDS["counterfactual_gap_min"] and probe_r2 >= msrs_r2:
        status = "INFORMATION_GATEWAY_ONLY"
    elif gaps["full"] < FROZEN_THRESHOLDS["counterfactual_gap_min"]:
        status = "INCONCLUSIVE"
    elif gauge_fn > FROZEN_THRESHOLDS["gauge_function_mse_max"]:
        status = "INCONCLUSIVE"
    elif g_suff > FROZEN_THRESHOLDS["sufficiency_nmse_max"]:
        status = "INCONCLUSIVE"
    elif g_gaps["full"] < FROZEN_THRESHOLDS["counterfactual_gap_min"]:
        status = "INCONCLUSIVE"
    else:
        status = "PATH_MECHANISM_RECOVERY_PASSED"
    action_carrier = [name for name in coalition if name in parent.ACT_SITES]
    downstream = [name for name in coalition if name not in parent.ACT_SITES]
    return {
        "msrs": list(coalition),
        "mcp": {"msrs": list(coalition), "path_class": path_class},
        "action_carrier_set": action_carrier,
        "downstream_msrs": downstream,
        "downstream_only_diagnostic": list(downstream_only),
        "pre_prune_circuit": list(raw_coalition),
        "sufficiency": metrics["sufficiency"],
        "necessity": metrics["necessity"],
        "specificity_independent": spec,
        "specificity_vs_dx_diagnostic": metrics["necessity"][parent.PRIMARY]
        / max(metrics["necessity"]["dx"], 1e-6),
        "minimality_drop": metrics["minimality_drop"],
        "random_plus_one_p": float(plus_one_p),
        "random_control_sufficient_count": int(random_sufficient),
        "counterfactual_gap_closed": float(gaps["full"]),
        "path_gaps": {key: float(val) for key, val in gaps.items()},
        "path_class": path_class,
        "information_r2": r2,
        "information_msrs_mean_r2": msrs_r2,
        "probe_topk": list(probe_topk),
        "probe_topk_mean_r2": probe_r2,
        "probe_topk_sufficiency_dvx": probe_eval["sufficiency"][parent.PRIMARY],
        "probe_topk_necessity_dvx": probe_eval["necessity"][parent.PRIMARY],
        "probe_topk_counterfactual": float(probe_gap),
        "gauge_function_mse": gauge_fn,
        "gauge_msrs": list(g_coal),
        "gauge_sufficiency_dvx": float(g_suff),
        "gauge_path_class": g_path,
        "gauge_path_gaps": {key: float(val) for key, val in g_gaps.items()},
        "literal_jaccard_vs_gauge": (
            len(set(coalition) & set(g_coal)) / max(len(set(coalition) | set(g_coal)), 1)
        ),
        "baselines": base_eval,
        "architecture_action_cutset": {
            "sites": list(parent.ACT_SITES),
            "sufficiency": act_cut["sufficiency"],
            "necessity": act_cut["necessity"],
            "specificity_dvx_over_dvy": act_cut["necessity"][parent.PRIMARY]
            / max(act_cut["necessity"][parent.CONTROL], 1e-6),
        },
        "cancellation": cancel,
        "status": status,
        "hypothesis": _hypothesis(status, path_class),
        "action_embedding_only": set(coalition) <= set(parent.ACT_SITES),
        "architecture_cutset_automatic_fail": False,
        "searchable_sites": list(parent.SITE_NAMES),
        "encoder_sites_excluded": list(parent.ENCODER_SITES_EXCLUDED),
        "physics_dependency": PHYSICS_DEPENDENCY,
        "independent_controls": list(INDEPENDENT_CONTROLS),
        "downstream_of_ax": list(DOWNSTREAM_OF_AX),
        "intervention_support": "coordinatewise_mean_fill",
        "counterfactual_support": "hybrid_activation_patch_plus_path_holds",
        "levels": {
            "information_carrier": bool(coalition) and msrs_r2 > 0.0,
            "causal_mediator": status == "PATH_MECHANISM_RECOVERY_PASSED"
            or (
                bool(coalition)
                and metrics["necessity"][parent.PRIMARY] >= FROZEN_THRESHOLDS["necessity_nmse_min"]
                and gaps["full"] >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
            ),
            "computational_mechanism": status == "PATH_MECHANISM_RECOVERY_PASSED" and path_class is not None,
        },
    }


def _fit_seed(
    seed: int, stage: str
) -> tuple[PathAwareActionDeltaPredictor, dict[str, Any]]:
    if int(seed) in FORBIDDEN_SEEDS:
        raise ValueError(f"forbidden seed {seed}")
    if stage == "development" and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("development seed mismatch")
    if stage == "confirmation" and seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed mismatch")
    train_s, train_a, train_d = parent._transitions(seed * 1000 + 61, 256)
    dev_s, dev_a, dev_d = parent._transitions(seed * 1000 + 67, 64)
    model, curve, ckpt = train_model(seed, train_s, train_a, train_d, TRAIN_STEPS)
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
        "train_steps": TRAIN_STEPS,
        "checkpoint_sha256": ckpt,
        "train_loss_final": curve[-1] if curve else train_loss,
        "train_loss_fullsplit": train_loss,
        "train_loss_curve": curve,
        "train_channels": train_channels,
        "competence": bundle,
        "circuit_search_ran": False,
        "ibd002_executed": False,
        "parent_002_rerun": False,
        "hard002_primary_seeds_reused": False,
        "claim_boundary": _claim_boundary(),
        "substrate": "supervised_residual_mlp_not_jepa_objective",
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


def _functional_equivalence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passed = [row for row in rows if row.get("status") == "PATH_MECHANISM_RECOVERY_PASSED"]
    classes = {row.get("path_class") for row in passed}
    jaccard = _literal_jaccard(passed)
    same_path = len(classes) == 1 and None not in classes
    different_literal = any(value < 1.0 for value in jaccard.values()) if jaccard else False
    return {
        "n_passed": len(passed),
        "shared_path_class": next(iter(classes)) if same_path else None,
        "literal_jaccard": jaccard,
        "h_equivalent": bool(len(passed) >= 2 and same_path and different_literal),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    incompetent = [row for row in rows if row["status"] == "MODEL_INCOMPETENT"]
    passed = all(row.get("status") == "PATH_MECHANISM_RECOVERY_PASSED" for row in rows)
    if stage == "confirmation" and incompetent:
        status = "MODEL_INCOMPETENT_CONFIRMATION"
    elif incompetent:
        status = "MODEL_INCOMPETENT"
    elif passed:
        status = "PATH_MECHANISM_RECOVERY_PASSED"
    else:
        unique = {row["status"] for row in rows}
        status = next(iter(unique)) if len(unique) == 1 else "INCONCLUSIVE"
    return {
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_ID,
        "stage": stage,
        "train_steps": TRAIN_STEPS,
        "threshold_digest": threshold_digest(),
        "source_digest": source_digest(),
        "seeds": [row["seed"] for row in rows],
        "all_seeds_passed": passed,
        "all_seeds_competent": not bool(incompetent),
        "status": status,
        "evidence_level": "Causal effect" if passed else "None",
        "rows": list(rows),
        "literal_overlap_jaccard": _literal_jaccard(rows) if not incompetent else {},
        "functional_equivalence": _functional_equivalence(rows),
        "physics_dependency": PHYSICS_DEPENDENCY,
        "claim_boundary": _claim_boundary(),
        "substrate": "supervised_residual_mlp_not_jepa_objective",
        "parent_002_status_preserved": "INCONCLUSIVE",
        "parent_001_status_preserved": "MODEL_INCOMPETENT",
        "hard002_status_preserved": "NEGATIVE_RESULT",
        "ibd002_executed": False,
        "ibd003_status_preserved": "MECHANISM_RECOVERY_PASSED",
        "nomenclature": "learned WM, not JEPA objective",
        "seed_59_retrospective_pass": False,
    }


def run_development() -> dict[str, Any]:
    fitted = [_fit_seed(seed, "development") for seed in DEVELOPMENT_SEEDS]
    rows = [row for _, row in fitted]
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        return _aggregate(rows, "development")
    interpreted: list[dict[str, Any]] = []
    for seed, (model, row) in zip(DEVELOPMENT_SEEDS, fitted, strict=True):
        train_s, train_a, _ = parent._transitions(seed * 1000 + 61, 256)
        dev_s, dev_a, _ = parent._transitions(seed * 1000 + 67, 64)
        mechanism = _run_mechanism(model, seed, "development", train_s, train_a, dev_s, dev_a)
        merged = dict(row)
        merged.update(mechanism)
        merged["circuit_search_ran"] = True
        merged["evidence_level"] = (
            "Causal effect" if mechanism["status"] == "PATH_MECHANISM_RECOVERY_PASSED" else "None"
        )
        interpreted.append(merged)
    return _aggregate(interpreted, "development")


def _authorize_confirmation(dev_path: str) -> None:
    payload = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("development artifact is not LEARNED-WM-ACTION-DELTA-003")
    if payload.get("stage") != "development":
        raise ValueError("require-development file is not development")
    if payload.get("status") != "PATH_MECHANISM_RECOVERY_PASSED":
        raise ValueError("confirmation closed: development did not pass path mechanism recovery")
    if payload.get("all_seeds_competent") is not True:
        raise ValueError("confirmation closed: development was not fully competent")
    if payload.get("all_seeds_passed") is not True:
        raise ValueError("confirmation closed: development all_seeds_passed is not true")
    if payload.get("seeds") != list(DEVELOPMENT_SEEDS):
        raise ValueError("development seeds do not match freeze")
    if payload.get("threshold_digest") != threshold_digest():
        raise ValueError("development threshold digest mismatch")
    if payload.get("source_digest") != source_digest():
        raise ValueError("development source digest mismatch")
    if int(payload.get("train_steps", 0)) != TRAIN_STEPS:
        raise ValueError("development train_steps is not the frozen 800")
    sidecar = Path(dev_path).with_suffix(".provenance.json")
    if sidecar.exists():
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if prov.get("stage") != "development":
            raise ValueError("development provenance stage mismatch")
        if prov.get("seed") != DEVELOPMENT_SEEDS[0]:
            raise ValueError("development provenance seed must be 79")
        if "--stage confirmation" in str(prov.get("command", "")):
            raise ValueError("development provenance fuses confirmation")


def run_confirmation(development_path: str) -> dict[str, Any]:
    _authorize_confirmation(development_path)
    fitted = [_fit_seed(seed, "confirmation") for seed in CONFIRMATION_SEEDS]
    rows = [row for _, row in fitted]
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        return _aggregate(rows, "confirmation")
    interpreted: list[dict[str, Any]] = []
    for seed, (model, row) in zip(CONFIRMATION_SEEDS, fitted, strict=True):
        train_s, train_a, _ = parent._transitions(seed * 1000 + 61, 256)
        dev_s, dev_a, _ = parent._transitions(seed * 1000 + 67, 64)
        mechanism = _run_mechanism(model, seed, "confirmation", train_s, train_a, dev_s, dev_a)
        merged = dict(row)
        merged.update(mechanism)
        merged["circuit_search_ran"] = True
        merged["evidence_level"] = (
            "Causal effect" if mechanism["status"] == "PATH_MECHANISM_RECOVERY_PASSED" else "None"
        )
        interpreted.append(merged)
    return _aggregate(interpreted, "confirmation")


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
    parser.add_argument("--output", default="")
    parser.add_argument("--require-development", default="")
    args = parser.parse_args()
    if args.stage == "development":
        require_dev = None
        payload = run_development()
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
