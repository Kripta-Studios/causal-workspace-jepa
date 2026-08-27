"""CRCT-LEARNED-WM-ACTION-DELTA-004: Level-3 path recovery after 003 competence stop.

Does not mutate 001/002/003. Seed 59 is not a pass. Seeds 83/89 are not interpreted.
Not a JEPA-objective experiment. INTERACTING is not a Level-3 pass.
Action-stem MSRS cannot be Level 3. REDUNDANT_ROUTES is not a Level-3 pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from causal_workspace_jepa.experiments.cross_domain import crct_jepa_action_delta as parent
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta as wm002
from causal_workspace_jepa.experiments.cross_domain import crct_learned_wm_action_delta_003 as wm003

EXPERIMENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-004"
MODULE = "causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_004"
PARENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-003"
CONFIG_PATH = Path("configs/experiments/crct_learned_wm_action_delta_v4.json")

LADDER_RUNGS = (800, 2000, 5000)
DEVELOPMENT_SEEDS = (97, 101, 107)
CONFIRMATION_SEEDS = (1063, 1069, 1087)

FORBIDDEN_SEEDS = frozenset(wm003.FORBIDDEN_SEEDS) | {79, 83, 89, 1049, 1051, 1061}

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


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_digest() -> str:
    parts = [
        Path(parent.__file__).read_bytes(),
        Path(wm002.__file__).read_bytes(),
        Path(wm003.__file__).read_bytes(),
        Path(__file__).read_bytes(),
    ]
    return hashlib.sha256(b"\n".join(parts)).hexdigest()


def _require_execution_authorized() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("config experiment_id mismatch")
    if config.get("execution_authorized") is not True:
        raise ValueError("CRCT-LEARNED-WM-ACTION-DELTA-004 execution is not authorized")
    if config.get("status") == "DRAFT_NOT_PREREGISTERED":
        raise ValueError("CRCT-LEARNED-WM-ACTION-DELTA-004 is not frozen")


def _claim_boundary() -> str:
    return (
        "supervised residual-MLP PointMass world model only; not a JEPA objective; "
        "does not reinterpret 001/002/003; seed 59 is not a retrospective pass; "
        "INTERACTING is not Level 3; action-stem MSRS cannot be Level 3; "
        "REDUNDANT_ROUTES is not Level 3; does not alter HARD-002, IBD-002, or IBD-003"
    )


class PathAwareActionDeltaPredictor(wm003.PathAwareActionDeltaPredictor):
    """Same residual topology as 001–003. Path-hold equations are in the 004 protocol."""


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


LEVEL3_STATUSES = frozenset(
    {"DIRECT_PATH_MECHANISM_PASSED", "DISTRIBUTED_PATH_MECHANISM_PASSED"}
)
LEVEL2_STATUSES = frozenset(
    {"MEDIATOR_FOUND_PATH_UNRESOLVED", "INFORMATION_GATEWAY_ONLY", "REDUNDANT_ROUTES"}
)


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


def _level_for(status: str) -> int:
    if status in LEVEL3_STATUSES:
        return 3
    if status in LEVEL2_STATUSES:
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
    g_full: float,
    gauge_fn: float,
    g_suff: float,
    g_nec: float,
    g_path: str | None,
    path_class: str | None,
    action_only: bool,
    probe_unique_fail: bool,
) -> tuple[str, int]:
    """Label-blind status machine. Action-stem MSRS cannot be Level 3."""

    split = path_class in {"DIRECT", "DISTRIBUTED"}
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
    elif g_full < FROZEN_THRESHOLDS["counterfactual_gap_min"]:
        status = "INCONCLUSIVE"
    elif gauge_fn > FROZEN_THRESHOLDS["gauge_function_mse_max"]:
        status = "GAUGE_FAILED"
    elif (
        g_suff > FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        or g_nec < FROZEN_THRESHOLDS["necessity_nmse_min"]
    ):
        status = "GAUGE_FAILED"
    elif g_path != path_class:
        status = "PATH_CLASS_GAUGE_UNSTABLE"
    elif action_only:
        status = "INFORMATION_GATEWAY_ONLY"
    elif probe_unique_fail and not split:
        status = "INFORMATION_GATEWAY_ONLY"
    elif path_class == "INTERACTING":
        status = "MEDIATOR_FOUND_PATH_UNRESOLVED"
    elif path_class == "REDUNDANT_ROUTES":
        status = "REDUNDANT_ROUTES"
    elif path_class == "DIRECT":
        status = "DIRECT_PATH_MECHANISM_PASSED"
    elif path_class == "DISTRIBUTED":
        status = "DISTRIBUTED_PATH_MECHANISM_PASSED"
    else:
        status = "INCONCLUSIVE"
    return status, _level_for(status)


def _causal_conjunction(suff: float, nec: float, gap: float) -> bool:
    return (
        suff <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        and nec >= FROZEN_THRESHOLDS["necessity_nmse_min"]
        and gap >= FROZEN_THRESHOLDS["counterfactual_gap_min"]
    )


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
        y0, original = model.forward_intervene(eval_s, eval_a, None)
    metrics = parent._evaluate_coalition(model, eval_s, eval_a, means, coalition, y0)
    spec = {
        name: metrics["necessity"][parent.PRIMARY] / max(metrics["necessity"][name], 1e-6)
        for name in INDEPENDENT_CONTROLS
    }
    rng = random.Random(int(seed) * 8191 + 3)
    plus_one_p, control_sets, random_sufficient = parent._random_controls(
        model, eval_s, eval_a, means, coalition, y0, rng
    )
    rms_rows = parent._rms_controls(original, coalition, control_sets)
    rms_sufficient = [
        parent._predict_nmse(
            model, eval_s, eval_a, parent._mean_except(means, row, eval_s.shape[0]), parent.PRIMARY, y0
        )
        <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        for row in rms_rows
    ]
    action_only = set(coalition) <= set(parent.ACT_SITES) and bool(coalition)
    act_random_sufficient = 0
    if action_only:
        act_random_sufficient = _act_random_sufficient(
            model, eval_s, eval_a, means, coalition, y0, rng
        )
    r2 = wm003.information_r2(model, train_s, train_a)
    k = max(len(coalition), 1)
    probe_topk = sorted(parent.SITE_NAMES, key=lambda name: r2[name], reverse=True)[:k]
    probe_eval = parent._evaluate_coalition(model, eval_s, eval_a, means, probe_topk, y0)
    probe_gap = parent.counterfactual_gap(
        model, means, probe_topk, seed=seed, vary="ax", target=parent.PRIMARY
    )
    gaps = (
        wm003.counterfactual_paths(model, coalition, seed)
        if coalition
        else {"full": 0.0, "skip": 0.0, "residual": 0.0}
    )
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
        suff = parent._predict_nmse(model, eval_s, eval_a, fill, parent.PRIMARY, y0)
        nec = parent._predict_nmse(model, eval_s, eval_a, abl, parent.PRIMARY, y0)
        gap = parent.counterfactual_gap(model, means, sites, seed=seed, vary="ax", target=parent.PRIMARY)
        base_eval[label] = {
            "sites": sites,
            "sufficiency_dvx": suff,
            "necessity_dvx": nec,
            "counterfactual": float(gap),
            "causal_conjunction": _causal_conjunction(suff, nec, gap),
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
    g_metrics = parent._evaluate_coalition(gauged, eval_s, eval_a, g_means, g_coal, g_y0)
    g_suff = g_metrics["sufficiency"][parent.PRIMARY]
    g_nec = g_metrics["necessity"][parent.PRIMARY]
    g_gaps = (
        wm003.counterfactual_paths(gauged, g_coal, seed)
        if g_coal
        else {"full": 0.0, "skip": 0.0, "residual": 0.0}
    )
    g_path = classify_path(g_gaps) if g_coal else None
    cancel = parent._cancellation(model, eval_s, eval_a, means, coalition, y0)
    drop_still_sufficient = any(
        err <= FROZEN_THRESHOLDS["sufficiency_nmse_max"] for err in metrics["minimality_drop"].values()
    )
    spec_failed = any(spec[name] < FROZEN_THRESHOLDS["specificity_ratio_min"] for name in INDEPENDENT_CONTROLS)
    probe_unique_fail = _causal_conjunction(
        probe_eval["sufficiency"][parent.PRIMARY],
        probe_eval["necessity"][parent.PRIMARY],
        probe_gap,
    )
    status, level = adjudicate_seed(
        coalition=coalition,
        sufficiency_dvx=metrics["sufficiency"][parent.PRIMARY],
        drop_still_sufficient=drop_still_sufficient,
        necessity_dvx=metrics["necessity"][parent.PRIMARY],
        spec_failed=spec_failed,
        random_sufficient=int(random_sufficient),
        act_random_sufficient=int(act_random_sufficient),
        g_full=float(gaps["full"]),
        gauge_fn=gauge_fn,
        g_suff=float(g_suff),
        g_nec=float(g_nec),
        g_path=g_path,
        path_class=path_class,
        action_only=action_only,
        probe_unique_fail=bool(probe_unique_fail),
    )
    return {
        "msrs": list(coalition),
        "mcp": {"msrs": list(coalition), "path_class": path_class},
        "action_carrier_set": [name for name in coalition if name in parent.ACT_SITES],
        "downstream_msrs": [name for name in coalition if name not in parent.ACT_SITES],
        "pre_prune_circuit": list(raw_coalition),
        "sufficiency": metrics["sufficiency"],
        "necessity": metrics["necessity"],
        "specificity_independent": spec,
        "specificity_vs_dx_diagnostic": metrics["necessity"][parent.PRIMARY]
        / max(metrics["necessity"]["dx"], 1e-6),
        "minimality_drop": metrics["minimality_drop"],
        "random_plus_one_p": float(plus_one_p),
        "random_control_sufficient_count": int(random_sufficient),
        "act_random_control_sufficient_count": int(act_random_sufficient),
        "rms_matched_sufficient_count": int(sum(rms_sufficient)),
        "g_full": float(gaps["full"]),
        "g_skip": float(gaps["skip"]),
        "g_direct": float(gaps["skip"]),
        "g_residual": float(gaps["residual"]),
        "g_distributed": float(gaps["residual"]),
        "g_skip_semantics": "architecture_route_test_holds_hid1_hid2_at_A_overwrites_residual_msrs",
        "path_class": path_class,
        "information_r2": r2,
        "probe_topk": list(probe_topk),
        "probe_topk_sufficiency_dvx": probe_eval["sufficiency"][parent.PRIMARY],
        "probe_topk_necessity_dvx": probe_eval["necessity"][parent.PRIMARY],
        "probe_topk_counterfactual": float(probe_gap),
        "probe_uniqueness_failed": bool(probe_unique_fail),
        "gauge_function_mse": gauge_fn,
        "gauge_msrs": list(g_coal),
        "gauge_sufficiency_dvx": float(g_suff),
        "gauge_necessity_dvx": float(g_nec),
        "gauge_path_class": g_path,
        "literal_jaccard_vs_gauge": (
            len(set(coalition) & set(g_coal)) / max(len(set(coalition) | set(g_coal)), 1)
        ),
        "baselines": base_eval,
        "cancellation": cancel,
        "status": status,
        "level": level,
        "action_embedding_only": action_only,
        "architecture_cutset_automatic_fail": False,
        "action_stem_msrs_cannot_be_level3": True,
        "physics_dependency": PHYSICS_DEPENDENCY,
        "independent_controls": list(INDEPENDENT_CONTROLS),
        "downstream_of_ax": list(DOWNSTREAM_OF_AX),
        "searchable_sites": list(parent.SITE_NAMES),
        "encoder_sites_excluded": list(parent.ENCODER_SITES_EXCLUDED),
        "intervention_support": "coordinatewise_mean_fill",
        "counterfactual_support": "hybrid_activation_patch_plus_path_holds",
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
        "parent_003_rerun": False,
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


def _shared_path_class(rows: Sequence[Mapping[str, Any]]) -> str | None:
    classes = {row.get("path_class") for row in rows}
    if len(classes) == 1:
        only = next(iter(classes))
        if only in {"DIRECT", "DISTRIBUTED", "REDUNDANT_ROUTES"}:
            return str(only)
    return None


def _pass_path_class(rows: Sequence[Mapping[str, Any]]) -> str | None:
    shared = _shared_path_class(rows)
    if shared in {"DIRECT", "DISTRIBUTED"}:
        return shared
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
    observed = _shared_path_class(rows) if not incompetent else None
    shared = _pass_path_class(rows) if not incompetent else None
    if stage == "confirmation" and incompetent:
        status = "MODEL_INCOMPETENT_CONFIRMATION"
    elif incompetent:
        status = "MODEL_INCOMPETENT"
    elif statuses == {"DIRECT_PATH_MECHANISM_PASSED"}:
        status = "PATH_MECHANISM_RECOVERY_PASSED"
    elif statuses == {"DISTRIBUTED_PATH_MECHANISM_PASSED"}:
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
        "observed_shared_path_class": observed,
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
        "parent_003_status_preserved": "MODEL_INCOMPETENT",
        "parent_002_status_preserved": "INCONCLUSIVE",
        "parent_001_status_preserved": "MODEL_INCOMPETENT",
        "hard002_status_preserved": "NEGATIVE_RESULT",
        "ibd002_executed": False,
        "seed_59_retrospective_pass": False,
        "nomenclature": "learned WM, not JEPA objective",
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
        raise ValueError("previous artifact is not LEARNED-WM-ACTION-DELTA-004")
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
        merged["evidence_level"] = (
            "Causal effect" if mechanism["status"] in LEVEL3_STATUSES else "None"
        )
        interpreted.append(merged)
    return _aggregate(interpreted, "development", train_steps)


def _authorize_confirmation(dev_path: str) -> int:
    payload = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("development artifact is not LEARNED-WM-ACTION-DELTA-004")
    if payload.get("stage") != "development":
        raise ValueError("require-development file is not development")
    if payload.get("status") != "PATH_MECHANISM_RECOVERY_PASSED":
        raise ValueError("confirmation closed: development did not pass shared split path class")
    if payload.get("all_seeds_competent") is not True:
        raise ValueError("confirmation closed: development was not fully competent")
    if payload.get("all_seeds_passed") is not True:
        raise ValueError("confirmation closed: development all_seeds_passed is not true")
    if payload.get("shared_path_class") not in {"DIRECT", "DISTRIBUTED"}:
        raise ValueError("confirmation closed: no shared split path class")
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
            raise ValueError("development provenance seed must be 97")
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
        return _aggregate(
            rows,
            "confirmation",
            train_steps,
            required_shared_path_class=required_class,
        )
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
        merged["evidence_level"] = (
            "Causal effect" if mechanism["status"] in LEVEL3_STATUSES else "None"
        )
        interpreted.append(merged)
    return _aggregate(
        interpreted,
        "confirmation",
        train_steps,
        required_shared_path_class=required_class,
    )


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
