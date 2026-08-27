"""CRCT-JEPA-ACTION-DELTA-001: learned PointMass action-delta mechanism recovery.

TinyJEPA ridge/identity is not used. IBD-002 is not executed. HARD-002 stays negative.
IBD-003 is not rerun. Interventions recompute downstream sites after an override.
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
from torch import Tensor, nn

from causal_workspace_jepa.data.synthetic.pointmass import generate_pointmass2d
from causal_workspace_jepa.interpretability.crct_coalition import (
    is_epsilon_sufficient,
    restoration_error,
)

EXPERIMENT_ID = "CRCT-JEPA-ACTION-DELTA-001"
MODULE = "causal_workspace_jepa.experiments.cross_domain.crct_jepa_action_delta"
HIDDEN = 6
TRAIN_STEPS = 200
BATCH = 64
LR = 3e-3

DEVELOPMENT_SEEDS = (43, 47, 53)
CONFIRMATION_SEEDS = (1013, 1019, 1021)
FORBIDDEN_SEEDS = frozenset(
    {
        1009,
        2027,
        4093,
        11,
        13,
        17,
        811,
        823,
        829,
        21,
        23,
        29,
        941,
        947,
        953,
        31,
        37,
        41,
        971,
        977,
        983,
    }
)

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
}

CHANNELS = ("dx", "dy", "dvx", "dvy")
PRIMARY = "dvx"
CONTROL = "dvy"
SECONDARY_TARGETS = ("dvy", "dx", "dy")
ACT_SITES = tuple(f"act_{i}" for i in range(HIDDEN))
B1_SITES = tuple(f"b1_{i}" for i in range(HIDDEN))
B2_SITES = tuple(f"b2_{i}" for i in range(HIDDEN))
SITE_NAMES = ACT_SITES + B1_SITES + B2_SITES
ENCODER_SITES_EXCLUDED = ("enc_0", "enc_1", "enc_2", "enc_3", "enc_4", "enc_5")


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _channel(delta: Tensor, name: str) -> Tensor:
    index = CHANNELS.index(name)
    return delta[:, index : index + 1]


def _as_batch(value: Tensor, batch: int) -> Tensor:
    if value.ndim == 0:
        return value.expand(batch)
    return value


class ActionDeltaPredictor(nn.Module):
    """Learned residual MLP with explicit intervenable hidden units.

    Searchable sites are post-nonlinearity coordinates of the action embedding
    and the two residual-block MLPs. Encoder coordinates are trained but excluded
    from the search set (frozen omission). Residual-stream coordinates are not
    independently searchable; they are affected by re-forward after site overrides.
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        if int(seed) in FORBIDDEN_SEEDS:
            raise ValueError(f"forbidden seed {seed}")
        torch.manual_seed(int(seed))
        self.enc1 = nn.Linear(4, HIDDEN)
        self.enc2 = nn.Linear(HIDDEN, HIDDEN)
        self.act1 = nn.Linear(2, HIDDEN)
        self.mix = nn.Linear(2 * HIDDEN, HIDDEN)
        self.b1_w1 = nn.Linear(HIDDEN, HIDDEN)
        self.b1_w2 = nn.Linear(HIDDEN, HIDDEN)
        self.b2_w1 = nn.Linear(HIDDEN, HIDDEN)
        self.b2_w2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, 4)
        self.register_buffer("q_act", torch.eye(HIDDEN))
        self.register_buffer("q_b1", torch.eye(HIDDEN))
        self.register_buffer("q_b2", torch.eye(HIDDEN))

    def forward_intervene(
        self,
        state: Tensor,
        action: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Causal re-forward: overridden sites are replaced, later sites recompute."""

        over = overrides or {}
        z = torch.tanh(self.enc2(torch.tanh(self.enc1(state))))
        e = torch.tanh(self.act1(action)) @ self.q_act
        e = _override_units(e, "act", over)
        h = torch.tanh(self.mix(torch.cat([z, e], dim=-1)))
        hid1 = torch.tanh(self.b1_w1(h)) @ self.q_b1
        hid1 = _override_units(hid1, "b1", over)
        h = h + self.b1_w2(hid1)
        hid2 = torch.tanh(self.b2_w1(h)) @ self.q_b2
        hid2 = _override_units(hid2, "b2", over)
        h = h + self.b2_w2(hid2)
        sites = {f"act_{i}": e[:, i] for i in range(HIDDEN)}
        sites.update({f"b1_{i}": hid1[:, i] for i in range(HIDDEN)})
        sites.update({f"b2_{i}": hid2[:, i] for i in range(HIDDEN)})
        return self.out(h), sites

    def collect_sites(self, state: Tensor, action: Tensor) -> dict[str, Tensor]:
        _, sites = self.forward_intervene(state, action, None)
        return sites

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        delta, _ = self.forward_intervene(state, action, None)
        return delta

    def apply_hidden_gauge(self, q_act: Tensor, q_b1: Tensor, q_b2: Tensor) -> None:
        """Mix searchable coordinates; compensate the next linear. Function preserved."""

        with torch.no_grad():
            eye = torch.eye(HIDDEN, device=self.mix.weight.device, dtype=self.mix.weight.dtype)
            blk = torch.block_diag(eye, q_act.to(dtype=self.mix.weight.dtype))
            self.mix.weight.copy_(self.mix.weight @ blk)
            self.b1_w2.weight.copy_(self.b1_w2.weight @ q_b1.to(dtype=self.b1_w2.weight.dtype))
            self.b2_w2.weight.copy_(self.b2_w2.weight @ q_b2.to(dtype=self.b2_w2.weight.dtype))
            self.q_act.copy_(q_act)
            self.q_b1.copy_(q_b1)
            self.q_b2.copy_(q_b2)


def _override_units(tensor: Tensor, prefix: str, overrides: Mapping[str, Tensor]) -> Tensor:
    out = tensor.clone()
    batch = out.shape[0]
    for index in range(HIDDEN):
        key = f"{prefix}_{index}"
        if key in overrides:
            out[:, index] = _as_batch(overrides[key], batch)
    return out


def _transitions(seed: int, trajectories: int) -> tuple[Tensor, Tensor, Tensor]:
    data = generate_pointmass2d(trajectories=trajectories, steps=6, seed=int(seed))
    state = torch.tensor(data.states[:, :-1, :].reshape(-1, 4))
    action = torch.tensor(data.actions.reshape(-1, 2))
    delta = torch.tensor((data.states[:, 1:, :] - data.states[:, :-1, :]).reshape(-1, 4))
    return state, action, delta


def _site_means(model: ActionDeltaPredictor, state: Tensor, action: Tensor) -> dict[str, Tensor]:
    sites = model.collect_sites(state, action)
    return {name: tensor.mean().detach() for name, tensor in sites.items()}


def _mean_except(
    means: Mapping[str, Tensor],
    keep: Sequence[str],
    batch: int,
) -> dict[str, Tensor]:
    kept = set(keep)
    return {name: _as_batch(means[name], batch) for name in SITE_NAMES if name not in kept}


def _mean_on(
    means: Mapping[str, Tensor],
    names: Sequence[str],
    batch: int,
) -> dict[str, Tensor]:
    return {name: _as_batch(means[name], batch) for name in names}


def _orthogonal(seed: int, offset: int) -> Tensor:
    gen = torch.Generator().manual_seed(int(seed) + offset)
    raw = torch.randn(HIDDEN, HIDDEN, generator=gen)
    q, _ = torch.linalg.qr(raw)
    return q


def train_model(seed: int, state: Tensor, action: Tensor, delta: Tensor) -> ActionDeltaPredictor:
    model = ActionDeltaPredictor(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = state.shape[0]
    gen = torch.Generator().manual_seed(int(seed) + 13)
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, n, (min(BATCH, n),), generator=gen)
        pred = model(state[idx], action[idx])
        loss = torch.mean((pred - delta[idx]).square())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def competence(
    model: ActionDeltaPredictor, state: Tensor, action: Tensor, delta: Tensor
) -> dict[str, float]:
    with torch.no_grad():
        pred = model(state, action)
    return {name: restoration_error(_channel(pred, name), _channel(delta, name)) for name in CHANNELS}


def _predict_nmse(
    model: ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    overrides: Mapping[str, Tensor],
    target: str,
    original_delta: Tensor,
) -> float:
    with torch.no_grad():
        pred, _ = model.forward_intervene(state, action, overrides)
    return restoration_error(_channel(pred, target), _channel(original_delta, target))


def greedy_restore(
    model: ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    means: Mapping[str, Tensor],
    *,
    target: str,
    forbidden: Sequence[str] = (),
) -> tuple[list[str], float]:
    """Label-blind restore-only search on the network's own target channel."""

    blocked = set(forbidden)
    with torch.no_grad():
        baseline, _ = model.forward_intervene(state, action, None)
    goal = _channel(baseline, target)
    batch = state.shape[0]
    chosen: list[str] = []
    remaining = [name for name in SITE_NAMES if name not in blocked]
    empty = _mean_except(means, (), batch)
    with torch.no_grad():
        pred, _ = model.forward_intervene(state, action, empty)
        current_error = restoration_error(_channel(pred, target), goal)
    for _ in range(int(FROZEN_THRESHOLDS["max_coalition"])):
        best_name = None
        best_error = current_error
        for name in remaining:
            trial = _mean_except(means, chosen + [name], batch)
            with torch.no_grad():
                pred, _ = model.forward_intervene(state, action, trial)
                err = restoration_error(_channel(pred, target), goal)
            if err + 1e-15 < best_error:
                best_error = err
                best_name = name
        if best_name is None:
            break
        improvement = current_error - best_error
        if improvement < FROZEN_THRESHOLDS["min_step_nmse"]:
            break
        chosen.append(best_name)
        remaining.remove(best_name)
        current_error = best_error
        if is_epsilon_sufficient(current_error, epsilon=FROZEN_THRESHOLDS["sufficiency_nmse_max"]):
            break
    return chosen, current_error


def prune_inclusion_minimal(
    model: ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    means: Mapping[str, Tensor],
    coalition: Sequence[str],
    *,
    target: str,
) -> list[str]:
    """Drop members that are not required for frozen sufficiency. Part of the selector."""

    with torch.no_grad():
        baseline, _ = model.forward_intervene(state, action, None)
    goal = _channel(baseline, target)
    chosen = list(coalition)
    changed = True
    while changed and chosen:
        changed = False
        for name in list(chosen):
            trial = [item for item in chosen if item != name]
            fill = _mean_except(means, trial, state.shape[0])
            with torch.no_grad():
                pred, _ = model.forward_intervene(state, action, fill)
                err = restoration_error(_channel(pred, target), goal)
            if is_epsilon_sufficient(err, epsilon=FROZEN_THRESHOLDS["sufficiency_nmse_max"]):
                chosen = trial
                changed = True
                break
    return chosen


def counterfactual_gap(
    model: ActionDeltaPredictor,
    means: Mapping[str, Tensor],
    coalition: Sequence[str],
    *,
    seed: int,
    vary: str = "ax",
    target: str = PRIMARY,
) -> float:
    """Median fraction of counterfactual target gap closed by patching C from a' into a."""

    gen = torch.Generator().manual_seed(int(seed) + 409)
    n = int(FROZEN_THRESHOLDS["counterfactual_pairs"])
    state = torch.empty(n, 4)
    state[:, 0:2] = (torch.rand(n, 2, generator=gen) * 2.0) - 1.0
    state[:, 2:4] = (torch.rand(n, 2, generator=gen) * 0.4) - 0.2
    ax = (torch.rand(n, 1, generator=gen) * 2.0) - 1.0
    ay = (torch.rand(n, 1, generator=gen) * 2.0) - 1.0
    if vary == "ax":
        varied = ax
        other = ay
    else:
        varied = ay
        other = ax
    varied2 = torch.clamp(varied + 0.7, -1.0, 1.0)
    same = (varied2 - varied).abs() < 1e-6
    varied2 = torch.where(same, torch.clamp(varied - 0.7, -1.0, 1.0), varied2)
    if vary == "ax":
        action_a = torch.cat([varied, other], dim=-1)
        action_b = torch.cat([varied2, other], dim=-1)
    else:
        action_a = torch.cat([other, varied], dim=-1)
        action_b = torch.cat([other, varied2], dim=-1)
    with torch.no_grad():
        y_a, sites_a = model.forward_intervene(state, action_a, None)
        y_b, sites_b = model.forward_intervene(state, action_b, None)
        patch = {name: sites_b[name] for name in coalition}
        y_p, _ = model.forward_intervene(state, action_a, patch)
    t = CHANNELS.index(target)
    base = (y_b[:, t] - y_a[:, t]).square()
    after = (y_b[:, t] - y_p[:, t]).square()
    closed = 1.0 - after / base.clamp_min(1e-12)
    return float(closed.median().item())


def _baselines(original: Mapping[str, Tensor], grads: Mapping[str, Tensor], k: int) -> dict[str, list[str]]:
    mag = sorted(SITE_NAMES, key=lambda n: float(original[n].abs().mean()), reverse=True)
    grd = sorted(SITE_NAMES, key=lambda n: float(grads[n].abs().mean()), reverse=True)
    axg = sorted(
        SITE_NAMES,
        key=lambda n: float((original[n].abs() * grads[n].abs()).mean()),
        reverse=True,
    )
    return {
        "magnitude_topk": mag[:k],
        "gradient_topk": grd[:k],
        "actgrad_topk": axg[:k],
    }


def _site_grads(model: ActionDeltaPredictor, state: Tensor, action: Tensor) -> dict[str, Tensor]:
    sites = {key: value.detach().requires_grad_(True) for key, value in model.collect_sites(state, action).items()}
    pred, _ = model.forward_intervene(state, action, sites)
    pred[:, 2].sum().backward()
    return {key: (value.grad if value.grad is not None else torch.zeros_like(value)) for key, value in sites.items()}


def _evaluate_coalition(
    model: ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    means: Mapping[str, Tensor],
    coalition: Sequence[str],
    original_delta: Tensor,
) -> dict[str, Any]:
    batch = state.shape[0]
    restored = _mean_except(means, coalition, batch)
    ablated = _mean_on(means, coalition, batch)
    suff = {name: _predict_nmse(model, state, action, restored, name, original_delta) for name in CHANNELS}
    nec = {name: _predict_nmse(model, state, action, ablated, name, original_delta) for name in CHANNELS}
    drop = {
        name: _predict_nmse(
            model,
            state,
            action,
            _mean_except(means, [item for item in coalition if item != name], batch),
            PRIMARY,
            original_delta,
        )
        for name in coalition
    }
    return {"sufficiency": suff, "necessity": nec, "minimality_drop": drop}


def _random_controls(
    model: ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    means: Mapping[str, Tensor],
    coalition: Sequence[str],
    original_delta: Tensor,
    rng: random.Random,
) -> tuple[float, list[list[str]]]:
    size = max(len(coalition), 1)
    pool = [list(item) for item in combinations(SITE_NAMES, size) if set(item) != set(coalition)]
    rng.shuffle(pool)
    control_sets = pool[: int(FROZEN_THRESHOLDS["random_control_count"])]
    sufficient = [
        _predict_nmse(
            model,
            state,
            action,
            _mean_except(means, row, state.shape[0]),
            PRIMARY,
            original_delta,
        )
        <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        for row in control_sets
    ]
    plus_one_p = (sum(sufficient) + 1) / (len(control_sets) + 1) if control_sets else 1.0
    return float(plus_one_p), control_sets, int(sum(sufficient))


def _rms_controls(
    original: Mapping[str, Tensor],
    coalition: Sequence[str],
    control_sets: Sequence[Sequence[str]],
    count: int = 8,
) -> list[list[str]]:
    if not coalition or not control_sets:
        return []
    target_rms = float(
        sum(float(original[name].square().mean().sqrt().item()) for name in coalition) / len(coalition)
    )

    def rms(row: Sequence[str]) -> float:
        return float(sum(float(original[name].square().mean().sqrt().item()) for name in row) / max(len(row), 1))

    ranked = sorted(control_sets, key=lambda row: abs(rms(row) - target_rms))
    return [list(row) for row in ranked[:count]]


def _cancellation(
    model: ActionDeltaPredictor,
    state: Tensor,
    action: Tensor,
    means: Mapping[str, Tensor],
    coalition: Sequence[str],
    original_delta: Tensor,
) -> dict[str, Any]:
    batch = state.shape[0]
    pairs: list[dict[str, Any]] = []
    if len(coalition) < 2:
        return {"status": "NO_MEANINGFUL_CANCELLATION_DETECTED", "pairs": pairs}
    with torch.no_grad():
        base_pred, _ = model.forward_intervene(state, action, None)
    base_mean = float(base_pred[:, 2].mean().item())
    min_member = FROZEN_THRESHOLDS["cancellation_member_nmse_min"]
    for left, right in combinations(coalition, 2):
        nec_l = _predict_nmse(model, state, action, _mean_on(means, [left], batch), PRIMARY, original_delta)
        nec_r = _predict_nmse(model, state, action, _mean_on(means, [right], batch), PRIMARY, original_delta)
        nec_j = _predict_nmse(
            model, state, action, _mean_on(means, [left, right], batch), PRIMARY, original_delta
        )
        with torch.no_grad():
            pred_l, _ = model.forward_intervene(state, action, _mean_on(means, [left], batch))
            pred_r, _ = model.forward_intervene(state, action, _mean_on(means, [right], batch))
        signed_l = float(pred_l[:, 2].mean().item() - base_mean)
        signed_r = float(pred_r[:, 2].mean().item() - base_mean)
        opposing = signed_l * signed_r < 0
        joint_small = nec_j <= 0.5 * min(nec_l, nec_r)
        if opposing and nec_l >= min_member and nec_r >= min_member and joint_small:
            pairs.append(
                {
                    "pair": [left, right],
                    "member_nmse": [nec_l, nec_r],
                    "joint_nmse": nec_j,
                    "signed": [signed_l, signed_r],
                }
            )
    return {
        "status": "DETECTED" if pairs else "NO_MEANINGFUL_CANCELLATION_DETECTED",
        "pairs": pairs,
    }


def run_seed(seed: int, stage: str) -> dict[str, Any]:
    if int(seed) in FORBIDDEN_SEEDS:
        raise ValueError(f"forbidden seed {seed}")
    if stage == "development" and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("development seed mismatch")
    if stage == "confirmation" and seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed mismatch")
    train_s, train_a, train_d = _transitions(seed * 1000 + 61, 256)
    dev_s, dev_a, dev_d = _transitions(seed * 1000 + 67, 64)
    model = train_model(seed, train_s, train_a, train_d)
    comp = competence(model, dev_s, dev_a, dev_d)
    if any(comp[name] > FROZEN_THRESHOLDS["competence_nmse_max"] for name in CHANNELS):
        return {
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "seed": seed,
            "status": "MODEL_INCOMPETENT",
            "evidence_level": "None",
            "competence": {key: float(value) for key, value in comp.items()},
            "ibd002_executed": False,
            "ibd003_rerun": False,
            "hard002_primary_seeds_reused": False,
            "claim_boundary": _claim_boundary(),
        }
    means = _site_means(model, train_s, train_a)
    raw_coalition, _ = greedy_restore(model, dev_s, dev_a, means, target=PRIMARY)
    coalition = prune_inclusion_minimal(
        model, dev_s, dev_a, means, raw_coalition, target=PRIMARY
    )
    alternate, _ = greedy_restore(
        model, dev_s, dev_a, means, target=PRIMARY, forbidden=coalition
    )
    alternate = prune_inclusion_minimal(
        model, dev_s, dev_a, means, alternate, target=PRIMARY
    )
    secondary = {
        name: prune_inclusion_minimal(
            model,
            dev_s,
            dev_a,
            means,
            greedy_restore(model, dev_s, dev_a, means, target=name)[0],
            target=name,
        )
        for name in SECONDARY_TARGETS
    }
    eval_s, eval_a = dev_s, dev_a
    if stage == "confirmation":
        eval_s, eval_a, _ = _transitions(seed * 1000 + 71, 64)
    with torch.no_grad():
        y0, original = model.forward_intervene(eval_s, eval_a, None)
    metrics = _evaluate_coalition(model, eval_s, eval_a, means, coalition, y0)
    spec_ratio = metrics["necessity"][PRIMARY] / max(metrics["necessity"][CONTROL], 1e-6)
    spec_ratio_dy = metrics["necessity"][PRIMARY] / max(metrics["necessity"]["dy"], 1e-6)
    rng = random.Random(int(seed) * 8191 + 3)
    plus_one_p, control_sets, random_sufficient = _random_controls(
        model, eval_s, eval_a, means, coalition, y0, rng
    )
    rms_rows = _rms_controls(original, coalition, control_sets)
    rms_sufficient = [
        _predict_nmse(
            model, eval_s, eval_a, _mean_except(means, row, eval_s.shape[0]), PRIMARY, y0
        )
        <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        for row in rms_rows
    ]
    gap = counterfactual_gap(model, means, coalition, seed=seed, vary="ax", target=PRIMARY)
    gap_ay = counterfactual_gap(model, means, secondary["dvy"], seed=seed, vary="ay", target="dvy")
    grads = _site_grads(model, eval_s[: min(64, eval_s.shape[0])], eval_a[: min(64, eval_a.shape[0])])
    orig_small = {
        key: original[key][: min(64, original[key].shape[0])] for key in SITE_NAMES
    }
    base = _baselines(orig_small, grads, max(len(coalition), 1))
    base_eval = {}
    for label, sites in base.items():
        fill = _mean_except(means, sites, eval_s.shape[0])
        abl = _mean_on(means, sites, eval_s.shape[0])
        base_eval[label] = {
            "sites": sites,
            "sufficiency_dvx": _predict_nmse(model, eval_s, eval_a, fill, PRIMARY, y0),
            "necessity_dvx": _predict_nmse(model, eval_s, eval_a, abl, PRIMARY, y0),
        }
    gauged = copy.deepcopy(model)
    gauged.apply_hidden_gauge(_orthogonal(seed, 7), _orthogonal(seed, 11), _orthogonal(seed, 13))
    with torch.no_grad():
        gauge_fn = float(torch.mean((gauged(eval_s, eval_a) - y0).square()).item())
    g_means = _site_means(gauged, train_s, train_a)
    g_coal, _ = greedy_restore(gauged, dev_s, dev_a, g_means, target=PRIMARY)
    g_coal = prune_inclusion_minimal(
        gauged, dev_s, dev_a, g_means, g_coal, target=PRIMARY
    )
    with torch.no_grad():
        g_y0, _ = gauged.forward_intervene(eval_s, eval_a, None)
    g_rest = _mean_except(g_means, g_coal, eval_s.shape[0])
    g_suff = _predict_nmse(gauged, eval_s, eval_a, g_rest, PRIMARY, g_y0)
    act_cut = _evaluate_coalition(model, eval_s, eval_a, means, list(ACT_SITES), y0)
    architecture_action_cutset = {
        "sites": list(ACT_SITES),
        "sufficiency": act_cut["sufficiency"],
        "necessity": act_cut["necessity"],
        "specificity_dvx_over_dvy": act_cut["necessity"][PRIMARY]
        / max(act_cut["necessity"][CONTROL], 1e-6),
    }
    secondary_eval = {}
    for name, sites in secondary.items():
        row = _evaluate_coalition(model, eval_s, eval_a, means, sites, y0)
        control = "dvx" if name == "dvy" else ("dy" if name == "dx" else "dx")
        secondary_eval[name] = {
            "recovered_circuit": sites,
            "sufficiency": row["sufficiency"],
            "necessity": row["necessity"],
            "specificity_target_over_control": (
                row["necessity"][name] / max(row["necessity"][control], 1e-6)
            ),
        }
    signed = {
        name: {
            "mean": float(original[name].mean().item()),
            "abs_mean": float(original[name].abs().mean().item()),
            "sign": float(torch.sign(original[name].mean()).item()),
        }
        for name in SITE_NAMES
    }
    cancel = _cancellation(model, eval_s, eval_a, means, coalition, y0)
    alt_eval = _evaluate_coalition(model, eval_s, eval_a, means, alternate, y0)
    drop_still_sufficient = any(
        err <= FROZEN_THRESHOLDS["sufficiency_nmse_max"]
        for err in metrics["minimality_drop"].values()
    )
    if not coalition:
        status = "LOCALIZATION_FAILED"
    elif set(coalition) == set(ACT_SITES):
        status = "INCONCLUSIVE"
    elif metrics["sufficiency"][PRIMARY] > FROZEN_THRESHOLDS["sufficiency_nmse_max"]:
        status = "SUFFICIENCY_FAILED"
    elif drop_still_sufficient:
        status = "MINIMALITY_FAILED"
    elif metrics["necessity"][PRIMARY] < FROZEN_THRESHOLDS["necessity_nmse_min"]:
        status = "NECESSITY_FAILED"
    elif spec_ratio < FROZEN_THRESHOLDS["specificity_ratio_min"]:
        status = "SPECIFICITY_FAILED"
    elif spec_ratio_dy < FROZEN_THRESHOLDS["specificity_ratio_min"]:
        status = "SPECIFICITY_FAILED"
    elif random_sufficient > FROZEN_THRESHOLDS["random_control_sufficient_max"]:
        status = "INCONCLUSIVE"
    elif gap < FROZEN_THRESHOLDS["counterfactual_gap_min"]:
        status = "INCONCLUSIVE"
    elif gauge_fn > FROZEN_THRESHOLDS["gauge_function_mse_max"]:
        status = "INCONCLUSIVE"
    elif g_suff > FROZEN_THRESHOLDS["sufficiency_nmse_max"]:
        status = "INCONCLUSIVE"
    else:
        status = "MECHANISM_RECOVERY_PASSED"
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "seed": seed,
        "threshold_digest": threshold_digest(),
        "competence": {key: float(value) for key, value in comp.items()},
        "recovered_circuit": list(coalition),
        "alternate_circuit": list(alternate),
        "alternate_sufficiency_dvx": alt_eval["sufficiency"][PRIMARY],
        "sufficiency": metrics["sufficiency"],
        "necessity": metrics["necessity"],
        "specificity_dvx_over_dvy": float(spec_ratio),
        "specificity_dvx_over_dy": float(spec_ratio_dy),
        "specificity_matrix_necessity": {
            "C_dvx": metrics["necessity"],
            **{f"C_{name}": secondary_eval[name]["necessity"] for name in SECONDARY_TARGETS},
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
        "architecture_action_cutset": architecture_action_cutset,
        "secondary_targets": secondary_eval,
        "signed": signed,
        "cancellation": cancel,
        "status": status,
        "evidence_level": "Causal effect" if status == "MECHANISM_RECOVERY_PASSED" else "None",
        "claim_boundary": _claim_boundary(),
        "hard002_primary_seeds_reused": False,
        "ibd002_executed": False,
        "ibd003_rerun": False,
        "in_support_mean_ablate": False,
        "intervention_support": "coordinatewise_mean_fill",
        "counterfactual_support": "hybrid_activation_patch",
        "searchable_sites": list(SITE_NAMES),
        "encoder_sites_excluded": list(ENCODER_SITES_EXCLUDED),
    }


def _claim_boundary() -> str:
    return (
        "learned tiny supervised PointMass residual-MLP predictor only; not a JEPA "
        "objective, not Qwen, workspace, Platonic, or planning; "
        "does not alter HARD-002, IBD-002, or IBD-003"
    )


def _literal_jaccard(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    circuits = [set(row.get("recovered_circuit") or ()) for row in rows]
    pairs: dict[str, float] = {}
    for i, left in enumerate(circuits):
        for j, right in enumerate(circuits):
            if j <= i:
                continue
            denom = max(len(left | right), 1)
            pairs[f"{rows[i]['seed']}_{rows[j]['seed']}"] = len(left & right) / denom
    return pairs


def run_stage(stage: str) -> dict[str, Any]:
    seeds = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    rows = [run_seed(seed, stage) for seed in seeds]
    passed = all(row["status"] == "MECHANISM_RECOVERY_PASSED" for row in rows)
    if any(row["status"] == "MODEL_INCOMPETENT" for row in rows):
        status = "MODEL_INCOMPETENT"
    elif passed:
        status = "MECHANISM_RECOVERY_PASSED"
    else:
        unique = {row["status"] for row in rows}
        status = next(iter(unique)) if len(unique) == 1 else "INCONCLUSIVE"
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "threshold_digest": threshold_digest(),
        "source_digest": source_digest(),
        "seeds": list(seeds),
        "all_seeds_passed": passed,
        "status": status,
        "evidence_level": "Causal effect" if passed else "None",
        "rows": rows,
        "literal_overlap_jaccard": _literal_jaccard(rows),
        "claim_boundary": _claim_boundary(),
        "ibd002_executed": False,
        "ibd003_rerun": False,
        "hard002_status_preserved": "NEGATIVE_RESULT",
        "ibd003_status_preserved": "MECHANISM_RECOVERY_PASSED",
    }


def _authorize_confirmation(dev_path: str) -> None:
    payload = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("development artifact is not JEPA-ACTION-DELTA-001")
    if payload.get("stage") != "development":
        raise ValueError("require-development file is not a development stage artifact")
    if payload.get("status") != "MECHANISM_RECOVERY_PASSED":
        raise ValueError("confirmation closed: development did not pass")
    if payload.get("all_seeds_passed") is not True:
        raise ValueError("confirmation closed: development all_seeds_passed is not true")
    if payload.get("seeds") != list(DEVELOPMENT_SEEDS):
        raise ValueError("development seeds do not match freeze")
    if payload.get("threshold_digest") != threshold_digest():
        raise ValueError("development threshold digest does not match this source")
    if payload.get("source_digest") != source_digest():
        raise ValueError("development source digest does not match this source")
    sidecar = Path(dev_path).with_suffix(".provenance.json")
    if sidecar.exists():
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if prov.get("stage") != "development":
            raise ValueError("development provenance stage mismatch")
        if prov.get("seed") != DEVELOPMENT_SEEDS[0]:
            raise ValueError("development provenance seed must be 43")
        if "--stage development" not in str(prov.get("command", "")):
            raise ValueError("development provenance command is not a development stage")
        if "--stage confirmation" in str(prov.get("command", "")):
            raise ValueError("development provenance fuses confirmation")


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
    if args.stage == "confirmation":
        if not args.require_development:
            raise ValueError("confirmation requires --require-development")
        _authorize_confirmation(args.require_development)
    payload = run_stage(args.stage)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        rel = _relative_posix(path)
        command = stage_cli_command(
            MODULE,
            args.stage,
            rel,
            require_development=(
                _relative_posix(Path(args.require_development)) if args.require_development else None
            ),
        )
        provenance = collect_provenance(
            command,
            "configs/resource/cpu_vps.yaml",
            seed=int(payload["seeds"][0]),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        extra = {
            "experiment_id": EXPERIMENT_ID,
            "stage": args.stage,
            "command_stage": args.stage,
            "seeds": list(payload["seeds"]),
            "metrics": rel,
            "threshold_digest": payload["threshold_digest"],
            "source_digest": payload.get("source_digest"),
        }
        write_provenance(path.with_suffix(".provenance.json"), provenance, extra=extra)
    print(text)
    return 0 if payload["all_seeds_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
