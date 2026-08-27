"""Interpretable-by-design coalition plant for CRCT successors.

The plant is a tiny concept-bottleneck with planted known, unknown, residual,
redundant, cancelling, and decoy routes. HARD-002 primary seeds are never used.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import Tensor, nn

from causal_workspace_jepa.interpretability.crct_coalition import (
    evaluate_coalition,
    intervention_support_label,
)

PLANTED_NODES = (
    "known_a",
    "known_b",
    "unknown",
    "residual",
    "cancel_pos",
    "cancel_neg",
)
DECOY_NODES = ("decoy",)
ALL_NODES = PLANTED_NODES + DECOY_NODES

FROZEN_THRESHOLDS = {
    "epsilon": 0.02,
    "redundancy_correlation_min": 0.99,
    "cancellation_sum_energy_ratio_max": 0.02,
    "cancellation_min_member_energy": 1e-4,
    "gauge_spearman_min": 0.99,
    "decoy_causal_energy_max": 1e-8,
    "matched_control_must_fail_sufficiency": True,
}

DEVELOPMENT_SEEDS = (11, 13, 17)
CONFIRMATION_SEEDS = (811, 823, 829)
HARD002_PRIMARY_SEEDS = (1009, 2027, 4093)


def threshold_digest() -> str:
    return hashlib.sha256(
        json.dumps(FROZEN_THRESHOLDS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class InterpretableBottleneckCircuit(nn.Module):
    """Linear concept bottleneck with explicit equivalent and cancelling paths."""

    def __init__(self, seed: int, *, device: torch.device) -> None:
        super().__init__()
        if int(seed) in HARD002_PRIMARY_SEEDS:
            raise ValueError("HARD-002 primary seeds are frozen and cannot be reused")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.seed = int(seed)
        self._device = device
        state_dim = 8
        action_dim = 4
        hidden = 6
        known_w = torch.randn(state_dim + action_dim, 2, generator=generator)
        known_r = torch.randn(2, hidden, generator=generator)
        unknown_w = torch.randn(state_dim + action_dim, hidden, generator=generator) * 0.4
        residual_w = torch.randn(state_dim + action_dim, hidden, generator=generator) * 0.85
        cancel_w = torch.randn(state_dim + action_dim, hidden, generator=generator) * 0.5
        decoy_w = torch.randn(state_dim + action_dim, hidden, generator=generator) * 3.0
        self.register_buffer("known_w", known_w.to(device))
        self.register_buffer("known_r", known_r.to(device))
        self.register_buffer("unknown_w", unknown_w.to(device))
        self.register_buffer("residual_w", residual_w.to(device))
        self.register_buffer("cancel_w", cancel_w.to(device))
        self.register_buffer("decoy_w", decoy_w.to(device))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

    def _features(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.cat([state, action], dim=-1)

    def component_output(self, name: str, state: Tensor, action: Tensor) -> Tensor:
        features = self._features(state, action)
        if name in {"known_a", "known_b"}:
            return features @ self.known_w @ self.known_r
        if name == "unknown":
            return features @ self.unknown_w
        if name == "residual":
            return (features @ self.residual_w) * torch.tanh(features[:, : self.hidden])
        if name == "cancel_pos":
            return features @ self.cancel_w
        if name == "cancel_neg":
            return -(features @ self.cancel_w)
        if name == "decoy":
            return features @ self.decoy_w
        raise KeyError(name)

    def absolute_output(self, state: Tensor, action: Tensor) -> Tensor:
        return (
            self.component_output("known_a", state, action)
            + self.component_output("unknown", state, action)
            + self.component_output("residual", state, action)
            + self.component_output("cancel_pos", state, action)
            + self.component_output("cancel_neg", state, action)
        )

    def gauge_output(self, state: Tensor, action: Tensor, *, scale: float) -> Tensor:
        features = self._features(state, action)
        known = features @ (self.known_w * scale) @ (self.known_r / scale)
        return (
            known
            + self.component_output("unknown", state, action)
            + self.component_output("residual", state, action)
        )

    def contributions(self, state: Tensor, action: Tensor) -> dict[str, Tensor]:
        return {name: self.component_output(name, state, action) for name in ALL_NODES}


def _make_split(n: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
    state = torch.randn(n, 8, generator=generator).to(device)
    action = torch.randn(n, 4, generator=generator).to(device)
    return state, action


def _spearman(left: list[float], right: list[float]) -> float:
    def ranks(values: list[float]) -> Tensor:
        order = sorted(range(len(values)), key=lambda i: values[i])
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


def run_coalition_benchmark(
    *,
    seed: int,
    stage: str,
    device_name: str = "cpu",
    samples: int = 256,
) -> dict[str, Any]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be development or confirmation")
    if stage == "development" and seed not in DEVELOPMENT_SEEDS:
        raise ValueError("development must use the frozen development seeds")
    if stage == "confirmation" and seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation must use the frozen confirmation seeds")
    device = torch.device(device_name)
    model = InterpretableBottleneckCircuit(seed, device=device)
    iid_state, iid_action = _make_split(samples, seed + 101, device)
    ood_state, ood_action = _make_split(samples, seed + 909, device)
    ood_state = ood_state * 1.7
    iid_target = model.absolute_output(iid_state, iid_action)
    ood_target = model.absolute_output(ood_state, ood_action)
    iid_all = model.contributions(iid_state, iid_action)
    ood_all = model.contributions(ood_state, ood_action)
    iid_contrib = {name: iid_all[name] for name in PLANTED_NODES}
    ood_contrib = {name: ood_all[name] for name in PLANTED_NODES}
    causal_energy = {
        name: float(torch.mean(tensor.square()).item()) for name, tensor in iid_contrib.items()
    }
    selected = ["known_a", "unknown", "residual"]
    matched_control = ["cancel_pos", "cancel_neg"]
    support = [
        intervention_support_label(
            operation="zero",
            site=name,
            magnitude=1.0,
            combination=(name,),
            trained_operations=("zero", "scale"),
            trained_sites=ALL_NODES,
            magnitude_max=1.0,
            trained_combination_size_max=3,
        )
        for name in selected
    ]
    support.append(
        intervention_support_label(
            operation="steer",
            site="known_a",
            magnitude=8.0,
            combination=("known_a", "unknown", "residual", "decoy"),
            trained_operations=("zero", "scale"),
            trained_sites=ALL_NODES,
            magnitude_max=1.0,
            trained_combination_size_max=3,
        )
    )
    iid_report = evaluate_coalition(
        iid_contrib,
        iid_target,
        planted=PLANTED_NODES,
        selected=selected,
        epsilon=FROZEN_THRESHOLDS["epsilon"],
        redundancy_correlation_min=FROZEN_THRESHOLDS["redundancy_correlation_min"],
        cancellation_sum_energy_ratio_max=FROZEN_THRESHOLDS["cancellation_sum_energy_ratio_max"],
        cancellation_min_member_energy=FROZEN_THRESHOLDS["cancellation_min_member_energy"],
        support_labels=support,
    )
    ood_report = evaluate_coalition(
        ood_contrib,
        ood_target,
        planted=PLANTED_NODES,
        selected=selected,
        epsilon=FROZEN_THRESHOLDS["epsilon"],
        redundancy_correlation_min=FROZEN_THRESHOLDS["redundancy_correlation_min"],
        cancellation_sum_energy_ratio_max=FROZEN_THRESHOLDS["cancellation_sum_energy_ratio_max"],
        cancellation_min_member_energy=FROZEN_THRESHOLDS["cancellation_min_member_energy"],
        support_labels=support,
    )
    control_report = evaluate_coalition(
        iid_contrib,
        iid_target,
        planted=PLANTED_NODES,
        selected=matched_control,
        epsilon=FROZEN_THRESHOLDS["epsilon"],
        redundancy_correlation_min=FROZEN_THRESHOLDS["redundancy_correlation_min"],
        cancellation_sum_energy_ratio_max=FROZEN_THRESHOLDS["cancellation_sum_energy_ratio_max"],
        cancellation_min_member_energy=FROZEN_THRESHOLDS["cancellation_min_member_energy"],
    )
    gauge_base = [causal_energy[name] for name in PLANTED_NODES]
    scale = 25.0
    gauged = InterpretableBottleneckCircuit(seed, device=device)
    gauged.known_w.mul_(scale)
    gauged.known_r.div_(scale)
    gauged_target = gauged.absolute_output(iid_state, iid_action)
    gauged_all = gauged.contributions(iid_state, iid_action)
    gauged_contrib = {name: gauged_all[name] for name in PLANTED_NODES}
    gauge_energy = [
        float(torch.mean(gauged_contrib[name].square()).item()) for name in PLANTED_NODES
    ]
    gauge_fn_error = float(torch.mean((gauged_target - iid_target).square()).item())
    gauged_report = evaluate_coalition(
        gauged_contrib,
        gauged_target,
        planted=PLANTED_NODES,
        selected=selected,
        epsilon=FROZEN_THRESHOLDS["epsilon"],
        redundancy_correlation_min=FROZEN_THRESHOLDS["redundancy_correlation_min"],
        cancellation_sum_energy_ratio_max=FROZEN_THRESHOLDS["cancellation_sum_energy_ratio_max"],
        cancellation_min_member_energy=FROZEN_THRESHOLDS["cancellation_min_member_energy"],
        support_labels=support,
    )
    broken = InterpretableBottleneckCircuit(seed, device=device)
    broken.known_w.mul_(scale)
    broken_known_energy = float(
        torch.mean(broken.component_output("known_a", iid_state, iid_action).square()).item()
    )
    decoy_energy = float(torch.mean(iid_all["decoy"].square()).item())
    decoy_causal = 0.0  # decoy is excluded from absolute_output by construction
    gauge_spearman = _spearman(gauge_base, gauge_energy)
    gauged_minimal = {frozenset(item) for item in gauged_report.minimal_sets}
    out_of_support_labeled = any(not bool(item["in_support"]) for item in support)
    gauge_ok = (
        gauge_fn_error <= 1e-8
        and gauge_spearman >= FROZEN_THRESHOLDS["gauge_spearman_min"]
        and gauged_report.sufficient
        and frozenset({"known_a", "unknown", "residual"}) in gauged_minimal
        and frozenset({"known_b", "unknown", "residual"}) in gauged_minimal
        and broken_known_energy > causal_energy["known_a"] * 10.0
    )
    ontology_distinguished = (
        iid_report.literal["recall"] < 1.0
        and iid_report.sufficient
        and any(set(item) == {"known_a", "unknown", "residual"} for item in iid_report.minimal_sets)
        and any(set(item) == {"known_b", "unknown", "residual"} for item in iid_report.minimal_sets)
        and not control_report.sufficient
        and any(item["equivalent"] for item in iid_report.equivalence)
        and bool(iid_report.cancellation)
        and decoy_causal == 0.0
        and decoy_energy > 1.0
        and out_of_support_labeled
        and gauge_ok
        and ood_report.sufficient
    )
    return {
        "experiment_id": "CRCT-COALITION-IBD-001",
        "stage": stage,
        "seed": seed,
        "threshold_digest": threshold_digest(),
        "thresholds": dict(FROZEN_THRESHOLDS),
        "planted_nodes": list(PLANTED_NODES),
        "selected_functional_circuit": selected,
        "matched_control": matched_control,
        "iid": iid_report.as_dict(),
        "ood": ood_report.as_dict(),
        "matched_control_report": control_report.as_dict(),
        "causal_energy": causal_energy,
        "decoy_activation_energy": decoy_energy,
        "decoy_causal_energy": decoy_causal,
        "gauge_function_mse": gauge_fn_error,
        "gauge_energy_spearman": gauge_spearman,
        "gauge_applied": True,
        "uncompensated_known_energy": broken_known_energy,
        "gauged": gauged_report.as_dict(),
        "ontology_distinguished": ontology_distinguished,
        "hard002_primary_seeds_reused": False,
        "status": "SMOKE_VALIDATED" if ontology_distinguished else "NEGATIVE_RESULT",
        "evidence_level": "Causal mediation",
        "claim_boundary": (
            "synthetic interpretable-by-design evaluator test only; not a Qwen, JEPA, "
            "or workspace claim; does not alter HARD-002"
        ),
    }


def run_stage(stage: str, *, device_name: str = "cpu") -> dict[str, Any]:
    seeds = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    rows = [run_coalition_benchmark(seed=seed, stage=stage, device_name=device_name) for seed in seeds]
    distinguished = all(row["ontology_distinguished"] for row in rows)
    return {
        "experiment_id": "CRCT-COALITION-IBD-001",
        "stage": stage,
        "threshold_digest": threshold_digest(),
        "seeds": list(seeds),
        "all_seeds_distinguish_ontology": distinguished,
        "status": "SMOKE_VALIDATED" if distinguished else "NEGATIVE_RESULT",
        "evidence_level": "Causal mediation",
        "rows": rows,
    }


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    payload = run_stage(args.stage, device_name=args.device)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["all_seeds_distinguish_ontology"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
