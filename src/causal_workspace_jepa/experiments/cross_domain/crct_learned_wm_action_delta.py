"""CRCT-LEARNED-WM-ACTION-DELTA-002: competent supervised WM, then CRCT.

Does not mutate or rerun 001. Not a JEPA-objective experiment.
IBD-002 is not executed. HARD-002 stays negative. IBD-003 is not rerun.
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

EXPERIMENT_ID = "CRCT-LEARNED-WM-ACTION-DELTA-002"
MODULE = "causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta"
PARENT_ID = "CRCT-JEPA-ACTION-DELTA-001"

LADDER_RUNGS = (200, 800, 2000)
DEVELOPMENT_SEEDS = (59, 71, 73)
CONFIRMATION_SEEDS = (1031, 1033, 1039)

FORBIDDEN_SEEDS = frozenset(parent.FORBIDDEN_SEEDS) | {
    43,
    47,
    53,
    1013,
    1019,
    1021,
    701,
    901,
    131,
    137,
    139,
    151,
    157,
    163,
    251,
    257,
    263,
}

MECHANISTIC_THRESHOLDS = dict(parent.FROZEN_THRESHOLDS)
FROZEN_THRESHOLDS = {
    **MECHANISTIC_THRESHOLDS,
    "ladder_rungs": list(LADDER_RUNGS),
}


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_digest() -> str:
    parent_src = Path(parent.__file__).read_bytes()
    self_src = Path(__file__).read_bytes()
    return hashlib.sha256(parent_src + b"\n" + self_src).hexdigest()


def _claim_boundary() -> str:
    return (
        "supervised residual-MLP PointMass world model only; not a JEPA objective; "
        "does not interpret 001's incompetent models; does not alter HARD-002, "
        "IBD-002, or IBD-003"
    )


def train_model(
    seed: int,
    state: Tensor,
    action: Tensor,
    delta: Tensor,
    steps: int,
) -> tuple[parent.ActionDeltaPredictor, list[float], str]:
    if int(seed) in FORBIDDEN_SEEDS:
        raise ValueError(f"forbidden seed {seed}")
    model = parent.ActionDeltaPredictor(seed)
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


def _channel_report(pred: Tensor, target: Tensor) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for name in parent.CHANNELS:
        index = parent.CHANNELS.index(name)
        p = pred[:, index]
        t = target[:, index]
        energy = float(torch.mean(t.square()).item())
        mse = float(torch.mean((p - t).square()).item())
        report[name] = {
            "variance": float(torch.var(t).item()),
            "energy": energy,
            "mse": mse,
            "nmse": mse / max(energy, 1e-12),
            "pred_variance": float(torch.var(p).item()),
        }
    full_energy = float(torch.mean(target.square()).item())
    full_mse = float(torch.mean((pred - target).square()).item())
    report["full"] = {
        "variance": float(torch.var(target).item()),
        "energy": full_energy,
        "mse": full_mse,
        "nmse": full_mse / max(full_energy, 1e-12),
        "pred_variance": float(torch.var(pred).item()),
    }
    return report


def competence_bundle(
    model: parent.ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    delta: Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        pred = model(state, action)
        loss = float(torch.mean((pred - delta).square()).item())
    channels = _channel_report(pred, delta)
    nmse = {name: channels[name]["nmse"] for name in parent.CHANNELS}
    passed = all(nmse[name] <= MECHANISTIC_THRESHOLDS["competence_nmse_max"] for name in parent.CHANNELS)
    return {"channels": channels, "nmse": nmse, "dev_loss_mse": loss, "passed": passed}


def _run_mechanism(
    model: parent.ActionDeltaPredictor,
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
    alternate, _ = parent.greedy_restore(
        model, dev_s, dev_a, means, target=parent.PRIMARY, forbidden=coalition
    )
    alternate = parent.prune_inclusion_minimal(
        model, dev_s, dev_a, means, alternate, target=parent.PRIMARY
    )
    secondary = {
        name: parent.prune_inclusion_minimal(
            model,
            dev_s,
            dev_a,
            means,
            parent.greedy_restore(model, dev_s, dev_a, means, target=name)[0],
            target=name,
        )
        for name in parent.SECONDARY_TARGETS
    }
    eval_s, eval_a = dev_s, dev_a
    if stage == "confirmation":
        eval_s, eval_a, _ = parent._transitions(seed * 1000 + 71, 64)
    with torch.no_grad():
        y0, original = model.forward_intervene(eval_s, eval_a, None)
    metrics = parent._evaluate_coalition(model, eval_s, eval_a, means, coalition, y0)
    spec_ratio = metrics["necessity"][parent.PRIMARY] / max(metrics["necessity"][parent.CONTROL], 1e-6)
    spec_ratio_dy = metrics["necessity"][parent.PRIMARY] / max(metrics["necessity"]["dy"], 1e-6)
    rng = random.Random(int(seed) * 8191 + 3)
    plus_one_p, control_sets, random_sufficient = parent._random_controls(
        model, eval_s, eval_a, means, coalition, y0, rng
    )
    rms_rows = parent._rms_controls(original, coalition, control_sets)
    rms_sufficient = [
        parent._predict_nmse(
            model, eval_s, eval_a, parent._mean_except(means, row, eval_s.shape[0]), parent.PRIMARY, y0
        )
        <= MECHANISTIC_THRESHOLDS["sufficiency_nmse_max"]
        for row in rms_rows
    ]
    gap = parent.counterfactual_gap(model, means, coalition, seed=seed, vary="ax", target=parent.PRIMARY)
    gap_ay = parent.counterfactual_gap(
        model, means, secondary["dvy"], seed=seed, vary="ay", target="dvy"
    )
    grads = parent._site_grads(
        model, eval_s[: min(64, eval_s.shape[0])], eval_a[: min(64, eval_a.shape[0])]
    )
    orig_small = {key: original[key][: min(64, original[key].shape[0])] for key in parent.SITE_NAMES}
    base = parent._baselines(orig_small, grads, max(len(coalition), 1))
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
    act_cut = parent._evaluate_coalition(model, eval_s, eval_a, means, list(parent.ACT_SITES), y0)
    secondary_eval = {}
    for name, sites in secondary.items():
        row = parent._evaluate_coalition(model, eval_s, eval_a, means, sites, y0)
        control = "dvx" if name == "dvy" else ("dy" if name == "dx" else "dx")
        secondary_eval[name] = {
            "recovered_circuit": sites,
            "sufficiency": row["sufficiency"],
            "necessity": row["necessity"],
            "specificity_target_over_control": (
                row["necessity"][name] / max(row["necessity"][control], 1e-6)
            ),
        }
    cancel = parent._cancellation(model, eval_s, eval_a, means, coalition, y0)
    alt_eval = parent._evaluate_coalition(model, eval_s, eval_a, means, alternate, y0)
    drop_still_sufficient = any(
        err <= MECHANISTIC_THRESHOLDS["sufficiency_nmse_max"]
        for err in metrics["minimality_drop"].values()
    )
    if not coalition:
        status = "LOCALIZATION_FAILED"
    elif set(coalition) <= set(parent.ACT_SITES):
        status = "ARCHITECTURE_CUTSET"
    elif metrics["sufficiency"][parent.PRIMARY] > MECHANISTIC_THRESHOLDS["sufficiency_nmse_max"]:
        status = "SUFFICIENCY_FAILED"
    elif drop_still_sufficient:
        status = "MINIMALITY_FAILED"
    elif metrics["necessity"][parent.PRIMARY] < MECHANISTIC_THRESHOLDS["necessity_nmse_min"]:
        status = "NECESSITY_FAILED"
    elif spec_ratio < MECHANISTIC_THRESHOLDS["specificity_ratio_min"]:
        status = "SPECIFICITY_FAILED"
    elif spec_ratio_dy < MECHANISTIC_THRESHOLDS["specificity_ratio_min"]:
        status = "SPECIFICITY_FAILED"
    elif random_sufficient > MECHANISTIC_THRESHOLDS["random_control_sufficient_max"]:
        status = "INCONCLUSIVE"
    elif gap < MECHANISTIC_THRESHOLDS["counterfactual_gap_min"]:
        status = "INCONCLUSIVE"
    elif gauge_fn > MECHANISTIC_THRESHOLDS["gauge_function_mse_max"]:
        status = "INCONCLUSIVE"
    elif g_suff > MECHANISTIC_THRESHOLDS["sufficiency_nmse_max"]:
        status = "INCONCLUSIVE"
    else:
        status = "MECHANISM_RECOVERY_PASSED"
    return {
        "recovered_circuit": list(coalition),
        "alternate_circuit": list(alternate),
        "alternate_sufficiency_dvx": alt_eval["sufficiency"][parent.PRIMARY],
        "sufficiency": metrics["sufficiency"],
        "necessity": metrics["necessity"],
        "specificity_dvx_over_dvy": float(spec_ratio),
        "specificity_dvx_over_dy": float(spec_ratio_dy),
        "specificity_matrix_necessity": {
            "C_dvx": metrics["necessity"],
            **{f"C_{name}": secondary_eval[name]["necessity"] for name in parent.SECONDARY_TARGETS},
        },
        "minimality_drop": metrics["minimality_drop"],
        "random_plus_one_p": float(plus_one_p),
        "random_control_sufficient_count": int(random_sufficient),
        "pre_prune_circuit": list(raw_coalition),
        "rms_matched_sufficient_count": int(sum(rms_sufficient)),
        "counterfactual_gap_closed": float(gap),
        "counterfactual_gap_closed_ay": float(gap_ay),
        "gauge_function_mse": gauge_fn,
        "gauge_recovered": list(g_coal),
        "gauge_sufficiency_dvx": float(g_suff),
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
        "secondary_targets": secondary_eval,
        "cancellation": cancel,
        "status": status,
        "searchable_sites": list(parent.SITE_NAMES),
        "encoder_sites_excluded": list(parent.ENCODER_SITES_EXCLUDED),
        "action_embedding_only": set(coalition) <= set(parent.ACT_SITES),
        "intervention_support": "coordinatewise_mean_fill",
        "counterfactual_support": "hybrid_activation_patch",
    }


def _fit_seed(
    seed: int, stage: str, train_steps: int
) -> tuple[parent.ActionDeltaPredictor, dict[str, Any]]:
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
        train_channels = _channel_report(train_pred, train_d)
    bundle = competence_bundle(model, dev_s, dev_a, dev_d)
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
        "ibd002_executed": False,
        "ibd003_rerun": False,
        "hard002_primary_seeds_reused": False,
        "parent_001_rerun": False,
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


def run_seed(seed: int, stage: str, train_steps: int, *, interpret: bool) -> dict[str, Any]:
    model, row = _fit_seed(seed, stage, train_steps)
    if row["status"] == "MODEL_INCOMPETENT" or not interpret:
        return row
    train_s, train_a, _ = parent._transitions(seed * 1000 + 61, 256)
    dev_s, dev_a, _ = parent._transitions(seed * 1000 + 67, 64)
    mechanism = _run_mechanism(model, seed, stage, train_s, train_a, dev_s, dev_a)
    row.update(mechanism)
    row["circuit_search_ran"] = True
    row["evidence_level"] = "Causal effect" if mechanism["status"] == "MECHANISM_RECOVERY_PASSED" else "None"
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]], stage: str, train_steps: int) -> dict[str, Any]:
    incompetent = [row for row in rows if row["status"] == "MODEL_INCOMPETENT"]
    passed = all(row.get("status") == "MECHANISM_RECOVERY_PASSED" for row in rows)
    if stage == "confirmation" and incompetent:
        status = "MODEL_INCOMPETENT_CONFIRMATION"
    elif incompetent:
        status = "MODEL_INCOMPETENT"
    elif passed:
        status = "MECHANISM_RECOVERY_PASSED"
    else:
        unique = {row["status"] for row in rows}
        status = next(iter(unique)) if len(unique) == 1 else "INCONCLUSIVE"
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
        "status": status,
        "evidence_level": "Causal effect" if passed else "None",
        "rows": list(rows),
        "literal_overlap_jaccard": parent._literal_jaccard(rows) if passed or not incompetent else {},
        "claim_boundary": _claim_boundary(),
        "substrate": "supervised_residual_mlp_not_jepa_objective",
        "ibd002_executed": False,
        "ibd003_rerun": False,
        "parent_001_status_preserved": "MODEL_INCOMPETENT",
        "hard002_status_preserved": "NEGATIVE_RESULT",
        "ibd003_status_preserved": "MECHANISM_RECOVERY_PASSED",
        "nomenclature": "learned WM, not JEPA objective",
    }


def _trained_row(
    seed: int, stage: str, train_steps: int
) -> tuple[parent.ActionDeltaPredictor | None, dict[str, Any]]:
    model, row = _fit_seed(seed, stage, train_steps)
    if row["status"] == "MODEL_INCOMPETENT":
        return None, row
    return model, row


def run_development_rung(train_steps: int, *, previous_path: str | None = None) -> dict[str, Any]:
    if int(train_steps) not in LADDER_RUNGS:
        raise ValueError(f"rung {train_steps} is not on the frozen ladder")
    if int(train_steps) == LADDER_RUNGS[0]:
        if previous_path:
            raise ValueError("rung 200 must not receive --require-previous")
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
            "Causal effect" if mechanism["status"] == "MECHANISM_RECOVERY_PASSED" else "None"
        )
        interpreted.append(merged)
    return _aggregate(interpreted, "development", train_steps)


def run_confirmation(development_path: str) -> dict[str, Any]:
    train_steps = _authorize_confirmation(development_path)
    fitted = [_trained_row(seed, "confirmation", train_steps) for seed in CONFIRMATION_SEEDS]
    rows = [row for _, row in fitted]
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        return _aggregate(rows, "confirmation", train_steps)
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
            "Causal effect" if mechanism["status"] == "MECHANISM_RECOVERY_PASSED" else "None"
        )
        interpreted.append(merged)
    return _aggregate(interpreted, "confirmation", train_steps)


def _authorize_previous(path: str, current_rung: int) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("previous artifact is not LEARNED-WM-ACTION-DELTA-002")
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


def _authorize_confirmation(dev_path: str) -> int:
    payload = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("development artifact is not LEARNED-WM-ACTION-DELTA-002")
    if payload.get("stage") != "development":
        raise ValueError("require-development file is not development")
    if payload.get("status") != "MECHANISM_RECOVERY_PASSED":
        raise ValueError("confirmation closed: development did not pass mechanism recovery")
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
    selected = int(payload["selected_rung"])
    if selected not in LADDER_RUNGS:
        raise ValueError("selected rung is not on the frozen ladder")
    if int(payload["train_steps"]) != selected:
        raise ValueError("train_steps does not match selected_rung")
    sidecar = Path(dev_path).with_suffix(".provenance.json")
    if sidecar.exists():
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if prov.get("stage") != "development":
            raise ValueError("development provenance stage mismatch")
        if prov.get("seed") != DEVELOPMENT_SEEDS[0]:
            raise ValueError("development provenance seed must be 59")
        if "--stage confirmation" in str(prov.get("command", "")):
            raise ValueError("development provenance fuses confirmation")
    return selected


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
        if args.require_previous:
            _authorize_previous(args.require_previous, int(args.rung))
        elif int(args.rung) != LADDER_RUNGS[0]:
            raise ValueError("first development rung must be 200")
        extra = f"--rung {int(args.rung)}"
        if args.require_previous:
            extra += f" --require-previous {_relative_posix(Path(args.require_previous))}"
        payload = run_development_rung(
            int(args.rung),
            previous_path=args.require_previous or None,
        )
        require_dev = None
    else:
        if not args.require_development:
            raise ValueError("confirmation requires --require-development")
        extra = ""
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
