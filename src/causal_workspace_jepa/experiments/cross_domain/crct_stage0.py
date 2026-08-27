from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn


NodeFamily = Literal["route", "bypass", "nuisance"]


@dataclass(frozen=True)
class Profile:
    name: str
    train_samples: int
    validation_samples: int
    test_samples: int
    diagnostic_samples: int
    state_dim: int
    action_dim: int
    route_width: int
    bypass_width: int
    nuisance_width: int
    route_active: int
    bypass_active: int
    student_width: int
    student_depth: int
    student_steps: int
    batch_size: int
    learning_rate: float
    random_control_count: int


PROFILES = {
    "smoke": Profile(
        name="smoke", train_samples=4096, validation_samples=1024, test_samples=2048,
        diagnostic_samples=1024, state_dim=12, action_dim=8, route_width=12, bypass_width=6,
        nuisance_width=12, route_active=3, bypass_active=2, student_width=128,
        student_depth=3, student_steps=120, batch_size=512, learning_rate=2e-3,
        random_control_count=32,
    ),
    "full": Profile(
        name="full", train_samples=131072, validation_samples=16384, test_samples=32768,
        diagnostic_samples=8192, state_dim=16, action_dim=10, route_width=24, bypass_width=12,
        nuisance_width=24, route_active=6, bypass_active=3, student_width=512,
        student_depth=4, student_steps=1800, batch_size=4096, learning_rate=1e-3,
        random_control_count=256,
    ),
    "max": Profile(
        name="max", train_samples=393216, validation_samples=65536, test_samples=65536,
        diagnostic_samples=16384, state_dim=20, action_dim=12, route_width=32, bypass_width=16,
        nuisance_width=32, route_active=8, bypass_active=4, student_width=768,
        student_depth=5, student_steps=3200, batch_size=8192, learning_rate=8e-4,
        random_control_count=512,
    ),
}


def _sha256_json(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _randn(shape: tuple[int, ...], *, generator: torch.Generator, scale: float = 1.0) -> Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


class SyntheticCircuitPlant(nn.Module):
    """Known finite-intervention circuit with differential and genuinely nonlinear paths.

    The plant deliberately contains high-variance inactive coordinates.  Only a sparse subset of
    route/bypass coordinates reaches the output.  This lets the benchmark score circuit recovery
    against known node/edge truth instead of merely measuring predictive MSE.
    """

    def __init__(self, profile: Profile, *, seed: int) -> None:
        super().__init__()
        self.profile = profile
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        sd, ad = profile.state_dim, profile.action_dim
        rw, bw, nw = profile.route_width, profile.bypass_width, profile.nuisance_width

        self.register_buffer("w_linear", _randn((ad, sd), generator=g, scale=0.22))
        self.register_buffer("w_quad_a", _randn((ad, sd), generator=g, scale=0.28))
        self.register_buffer("w_quad_b", _randn((ad, sd), generator=g, scale=0.28))

        self.register_buffer("w_route_state", _randn((sd, rw), generator=g, scale=0.34))
        self.register_buffer("w_route_action", _randn((ad, rw), generator=g, scale=0.34))
        self.register_buffer("w_value", _randn((ad, rw), generator=g, scale=0.42))
        route_out = _randn((rw, sd), generator=g, scale=0.20)
        route_out[profile.route_active :] = 0.0
        self.register_buffer("w_route_out", route_out)

        self.register_buffer("w_bypass_action", _randn((ad, bw), generator=g, scale=0.45))
        self.register_buffer("w_bypass_state", _randn((sd, bw), generator=g, scale=0.30))
        bypass_out = _randn((bw, sd), generator=g, scale=0.18)
        bypass_out[profile.bypass_active :] = 0.0
        self.register_buffer("w_bypass_out", bypass_out)

        self.register_buffer("w_nuisance", _randn((sd, nw), generator=g, scale=0.65))

    def internals(
        self,
        x: Tensor,
        u: Tensor,
        *,
        gauge_scales: Tensor | None = None,
    ) -> dict[str, Tensor]:
        route_gate = torch.sigmoid(1.65 * (x @ self.w_route_state + u @ self.w_route_action))
        route_value = torch.tanh(2.0 * (u @ self.w_value))
        route_merge = route_gate * route_value
        route_readout_weight = self.w_route_out
        if gauge_scales is not None:
            route_merge = route_merge * gauge_scales
            route_readout_weight = route_readout_weight / gauge_scales[:, None]
        bypass_hidden = torch.sin(
            (u @ self.w_bypass_action) * (0.65 + torch.tanh(x @ self.w_bypass_state))
        )
        nuisance_hidden = 6.0 * torch.tanh(x @ self.w_nuisance)
        linear = u @ self.w_linear
        quadratic = 0.45 * ((u @ self.w_quad_a) * (u @ self.w_quad_b))
        route = 0.90 * (route_merge @ route_readout_weight)
        bypass = 0.55 * (bypass_hidden @ self.w_bypass_out)
        return {
            "linear": linear,
            "quadratic": quadratic,
            "route_gate": route_gate,
            "route_value": route_value,
            "route_merge": route_merge,
            "route": route,
            "bypass_hidden": bypass_hidden,
            "bypass": bypass,
            "nuisance_hidden": nuisance_hidden,
        }

    def delta(
        self,
        x: Tensor,
        u: Tensor,
        *,
        ablate: tuple[NodeFamily, int] | None = None,
        gauge_scales: Tensor | None = None,
    ) -> Tensor:
        values = self.internals(x, u, gauge_scales=gauge_scales)
        linear = values["linear"]
        quadratic = values["quadratic"]
        route_merge = (
            values["route_merge"].clone()
            if ablate and ablate[0] == "route"
            else values["route_merge"]
        )
        bypass_hidden = (
            values["bypass_hidden"].clone()
            if ablate and ablate[0] == "bypass"
            else values["bypass_hidden"]
        )
        if ablate is not None:
            family, index = ablate
            if family == "route":
                route_merge[..., index] = 0.0
            elif family == "bypass":
                bypass_hidden[..., index] = 0.0
            elif family == "nuisance":
                # Nuisance is deliberately disconnected from the output.
                pass
            else:
                raise ValueError(f"unknown node family: {family}")
        route_weight = self.w_route_out
        if gauge_scales is not None:
            route_weight = route_weight / gauge_scales[:, None]
        route = 0.90 * (route_merge @ route_weight)
        bypass = 0.55 * (bypass_hidden @ self.w_bypass_out)
        return linear + quadratic + route + bypass

    def ground_truth_nodes(self) -> set[str]:
        route = {f"route:{i}" for i in range(self.profile.route_active)}
        bypass = {f"bypass:{i}" for i in range(self.profile.bypass_active)}
        return route | bypass

    def all_nodes(self) -> list[str]:
        return (
            [f"route:{i}" for i in range(self.profile.route_width)]
            + [f"bypass:{i}" for i in range(self.profile.bypass_width)]
            + [f"nuisance:{i}" for i in range(self.profile.nuisance_width)]
        )


class ResidualStudent(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, width), nn.GELU()])
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        return self.net(torch.cat([x, u], dim=-1))


def _make_dataset(
    profile: Profile,
    *,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    total = profile.train_samples + profile.validation_samples + profile.test_samples
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn((total, profile.state_dim), generator=g, dtype=torch.float32)
    u = torch.rand((total, profile.action_dim), generator=g, dtype=torch.float32) * 2.4 - 1.2
    # Slightly widen the finite-intervention regime while keeping inputs bounded.
    scales = 0.75 + 0.5 * torch.rand((total, 1), generator=g, dtype=torch.float32)
    u = u * scales
    return x.to(device), u.to(device), torch.arange(total, device=device)


def _directional_taylor(
    plant: SyntheticCircuitPlant,
    x: Tensor,
    u: Tensor,
    *,
    order: int,
    ablate: tuple[NodeFamily, int] | None = None,
    gauge_scales: Tensor | None = None,
) -> Tensor:
    """Exact directional Taylor transport at u=0 using nested forward-mode AD."""
    u0 = torch.zeros_like(u)

    def f(z: Tensor) -> Tensor:
        return plant.delta(x, z, ablate=ablate, gauge_scales=gauge_scales)

    f0, j1 = torch.func.jvp(f, (u0,), (u,))
    if order == 1:
        return f0 + j1
    if order != 2:
        raise ValueError("order must be 1 or 2")

    def first_at(z: Tensor) -> Tensor:
        return torch.func.jvp(f, (z,), (u,))[1]

    _, j2 = torch.func.jvp(first_at, (u0,), (u,))
    return f0 + j1 + 0.5 * j2


def _batched_taylor(
    plant: SyntheticCircuitPlant,
    x: Tensor,
    u: Tensor,
    *,
    order: int,
    chunk: int,
    ablate: tuple[NodeFamily, int] | None = None,
    gauge_scales: Tensor | None = None,
) -> Tensor:
    pieces = []
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        pieces.append(
            _directional_taylor(
                plant,
                x[start:stop],
                u[start:stop],
                order=order,
                ablate=ablate,
                gauge_scales=gauge_scales,
            ).detach()
        )
    return torch.cat(pieces, dim=0)


def _mse(a: Tensor, b: Tensor) -> float:
    return float(torch.mean((a - b).square()).detach().cpu())


def _energy(a: Tensor) -> float:
    return float(torch.mean(a.square()).detach().cpu())


def _nmse(pred: Tensor, target: Tensor) -> float:
    denom = torch.mean(target.square()).clamp_min(1e-12)
    return float((torch.mean((pred - target).square()) / denom).detach().cpu())


def _average_precision(ranking: list[str], truth: set[str]) -> float:
    if not truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, name in enumerate(ranking, start=1):
        if name in truth:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(truth)


def _precision_recall_at_k(ranking: list[str], truth: set[str], k: int) -> tuple[float, float]:
    chosen = set(ranking[:k])
    hits = len(chosen & truth)
    return hits / max(k, 1), hits / max(len(truth), 1)


def _spearman_from_scores(left: dict[str, float], right: dict[str, float]) -> float:
    names = sorted(set(left) & set(right))
    if len(names) < 2:
        return 1.0
    def ranks(scores: dict[str, float]) -> dict[str, float]:
        ordered = sorted(names, key=lambda n: (scores[n], n))
        return {name: float(i) for i, name in enumerate(ordered)}
    a, b = ranks(left), ranks(right)
    av = statistics.fmean(a.values())
    bv = statistics.fmean(b.values())
    num = sum((a[n] - av) * (b[n] - bv) for n in names)
    da = math.sqrt(sum((a[n] - av) ** 2 for n in names))
    db = math.sqrt(sum((b[n] - bv) ** 2 for n in names))
    return num / max(da * db, 1e-12)


def _score_nodes(
    plant: SyntheticCircuitPlant,
    x: Tensor,
    u: Tensor,
    *,
    chunk: int,
    gauge_scales: Tensor | None = None,
) -> tuple[dict[str, dict[str, float]], Tensor, Tensor, Tensor]:
    full = plant.delta(x, u, gauge_scales=gauge_scales).detach()
    t1 = _batched_taylor(plant, x, u, order=1, chunk=chunk, gauge_scales=gauge_scales)
    t2 = _batched_taylor(plant, x, u, order=2, chunk=chunk, gauge_scales=gauge_scales)
    residual = full - t2
    residual_energy = torch.mean(residual.square()).clamp_min(1e-12)
    internals = plant.internals(x, u, gauge_scales=gauge_scales)

    scores: dict[str, dict[str, float]] = {
        "baseline:linear_path": {
            "activation_rms": float(torch.sqrt(torch.mean(internals["linear"].square())).cpu()),
            "finite_effect_energy": _energy(internals["linear"]),
            "first_order_effect_energy": _energy(internals["linear"]),
            "second_order_effect_energy": _energy(internals["linear"]),
            "residual_causal_fraction": 0.0,
        },
        "baseline:quadratic_path": {
            "activation_rms": float(torch.sqrt(torch.mean(internals["quadratic"].square())).cpu()),
            "finite_effect_energy": _energy(internals["quadratic"]),
            "first_order_effect_energy": 0.0,
            "second_order_effect_energy": _energy(internals["quadratic"]),
            "residual_causal_fraction": 0.0,
        },
    }
    for family, width, activation_key in (
        ("route", plant.profile.route_width, "route_merge"),
        ("bypass", plant.profile.bypass_width, "bypass_hidden"),
        ("nuisance", plant.profile.nuisance_width, "nuisance_hidden"),
    ):
        activation = internals[activation_key]
        for index in range(width):
            name = f"{family}:{index}"
            ablate = (family, index)
            if family == "nuisance":
                ablated_full = full
                ablated_t1 = t1
                ablated_t2 = t2
            else:
                ablated_full = plant.delta(x, u, ablate=ablate, gauge_scales=gauge_scales).detach()
                ablated_t1 = _batched_taylor(
                    plant, x, u, order=1, chunk=chunk, ablate=ablate, gauge_scales=gauge_scales
                )
                ablated_t2 = _batched_taylor(
                    plant, x, u, order=2, chunk=chunk, ablate=ablate, gauge_scales=gauge_scales
                )
            ablated_residual = ablated_full - ablated_t2
            finite_effect = torch.mean((full - ablated_full).square())
            first_effect = torch.mean((t1 - ablated_t1).square())
            second_effect = torch.mean((t2 - ablated_t2).square())
            residual_effect = torch.mean((residual - ablated_residual).square())
            scores[name] = {
                "activation_rms": float(
                    torch.sqrt(torch.mean(activation[:, index].square())).cpu()
                ),
                "finite_effect_energy": float(finite_effect.cpu()),
                "first_order_effect_energy": float(first_effect.cpu()),
                "second_order_effect_energy": float(second_effect.cpu()),
                "residual_causal_fraction": float((residual_effect / residual_energy).cpu()),
            }
    return scores, full, t1, t2


def _train_student(
    profile: Profile,
    plant: SyntheticCircuitPlant,
    x: Tensor,
    u: Tensor,
    *,
    device: torch.device,
    seed: int,
    chunk: int,
) -> dict[str, float | int | str]:
    n_train = profile.train_samples
    n_val = profile.validation_samples
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, x.shape[0])

    with torch.no_grad():
        full = plant.delta(x, u).detach()
    t2 = _batched_taylor(plant, x, u, order=2, chunk=chunk)
    residual = (full - t2).detach()

    _seed_all(seed + 404)
    student = ResidualStudent(
        profile.state_dim + profile.action_dim,
        profile.state_dim,
        width=profile.student_width,
        depth=profile.student_depth,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=profile.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 505)
    loss_trace: list[float] = []
    student.train()
    for step in range(profile.student_steps):
        idx = torch.randint(0, n_train, (profile.batch_size,), generator=generator, device=device)
        optimizer.zero_grad(set_to_none=True)
        pred = student(x[idx], u[idx])
        loss = torch.mean((pred - residual[idx]).square())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite student loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(profile.student_steps // 12, 1) == 0:
            loss_trace.append(float(loss.detach().cpu()))

    student.eval()
    with torch.no_grad():
        val_pred = student(x[val_slice], u[val_slice])
        test_pred = student(x[test_slice], u[test_slice])
        zero_val = torch.zeros_like(residual[val_slice])
        zero_test = torch.zeros_like(residual[test_slice])
        val_baseline = _mse(zero_val, residual[val_slice])
        test_baseline = _mse(zero_test, residual[test_slice])
        val_mse = _mse(val_pred, residual[val_slice])
        test_mse = _mse(test_pred, residual[test_slice])
        full_test = full[test_slice]
        t2_test = t2[test_slice]
        reconstructed = t2_test + test_pred
        replay_nmse = _nmse(reconstructed, full_test)
    params = sum(p.numel() for p in student.parameters())
    return {
        "parameter_count": params,
        "steps": profile.student_steps,
        "batch_size": profile.batch_size,
        "first_logged_loss": loss_trace[0],
        "last_logged_loss": loss_trace[-1],
        "validation_zero_residual_mse": val_baseline,
        "validation_student_mse": val_mse,
        "validation_improvement_fraction": 1.0 - val_mse / max(val_baseline, 1e-12),
        "test_zero_residual_mse": test_baseline,
        "test_student_mse": test_mse,
        "test_improvement_fraction": 1.0 - test_mse / max(test_baseline, 1e-12),
        "test_full_effect_replay_nmse": replay_nmse,
        "loss_trace": loss_trace,
    }


def _random_control_distribution(
    scores: dict[str, dict[str, float]],
    truth_count: int,
    *,
    seed: int,
    count: int,
) -> dict[str, float | int]:
    names = list(scores)
    truth_ranking = sorted(names, key=lambda n: scores[n]["residual_causal_fraction"], reverse=True)
    candidate_score = sum(
        scores[name]["residual_causal_fraction"] for name in truth_ranking[:truth_count]
    )
    rng = random.Random(seed)
    random_scores = []
    for _ in range(count):
        chosen = rng.sample(names, k=truth_count)
        random_scores.append(sum(scores[n]["residual_causal_fraction"] for n in chosen))
    ordered = sorted(random_scores)
    p95 = ordered[min(len(ordered) - 1, math.floor(0.95 * len(ordered)))]
    exceed = sum(value >= candidate_score for value in random_scores)
    return {
        "control_count": count,
        "candidate_topk_total_residual_causal_fraction": candidate_score,
        "random_mean": statistics.fmean(random_scores),
        "random_p95": p95,
        "empirical_p_value_plus_one": (exceed + 1) / (count + 1),
    }


def run_benchmark(
    *,
    profile_name: str,
    seed: int,
    device_name: str,
    output: Path,
) -> dict[str, object]:
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}; choose from {sorted(PROFILES)}")
    profile = PROFILES[profile_name]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    _seed_all(seed)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    plant = SyntheticCircuitPlant(profile, seed=seed + 1000).to(device)
    x, u, _ = _make_dataset(profile, seed=seed + 2000, device=device)
    truth = plant.ground_truth_nodes()
    diag_n = min(profile.diagnostic_samples, profile.test_samples)
    diag_start = profile.train_samples + profile.validation_samples
    x_diag = x[diag_start : diag_start + diag_n]
    u_diag = u[diag_start : diag_start + diag_n]
    chunk = min(2048 if device.type == "cuda" else 256, diag_n)

    node_scores, full, t1, t2 = _score_nodes(plant, x_diag, u_diag, chunk=chunk)
    residual = full - t2
    rankings: dict[str, list[str]] = {}
    for metric in (
        "activation_rms",
        "finite_effect_energy",
        "first_order_effect_energy",
        "second_order_effect_energy",
        "residual_causal_fraction",
    ):
        rankings[metric] = sorted(node_scores, key=lambda n: node_scores[n][metric], reverse=True)

    k = len(truth)
    ranking_metrics = {}
    for metric, ranking in rankings.items():
        precision, recall = _precision_recall_at_k(ranking, truth, k)
        ranking_metrics[metric] = {
            "average_precision": _average_precision(ranking, truth),
            "precision_at_truth_k": precision,
            "recall_at_truth_k": recall,
            "top_k": ranking[:k],
        }

    # Function-preserving diagonal gauge: coordinates are rescaled and readout inverted.
    gauge_generator = torch.Generator(device="cpu")
    gauge_generator.manual_seed(seed + 3000)
    log_scales = torch.linspace(-2.5, 2.5, profile.route_width)
    perm = torch.randperm(profile.route_width, generator=gauge_generator)
    gauge_scales = torch.exp(log_scales[perm]).to(device)
    gauge_full = plant.delta(x_diag, u_diag, gauge_scales=gauge_scales).detach()
    gauge_function_max_abs_error = float((gauge_full - full).abs().max().cpu())
    gauge_scores, _, _, _ = _score_nodes(
        plant, x_diag, u_diag, chunk=chunk, gauge_scales=gauge_scales
    )
    original_activation = {
        name: values["activation_rms"]
        for name, values in node_scores.items()
        if name.startswith("route:")
    }
    gauged_activation = {
        name: values["activation_rms"]
        for name, values in gauge_scores.items()
        if name.startswith("route:")
    }
    original_causal = {
        name: values["residual_causal_fraction"]
        for name, values in node_scores.items()
        if name.startswith("route:")
    }
    gauged_causal = {
        name: values["residual_causal_fraction"]
        for name, values in gauge_scores.items()
        if name.startswith("route:")
    }
    gauge_causal_max_diff = max(abs(original_causal[n] - gauged_causal[n]) for n in original_causal)

    student_metrics = _train_student(
        profile,
        plant,
        x,
        u,
        device=device,
        seed=seed,
        chunk=min(4096 if device.type == "cuda" else 256, x.shape[0]),
    )
    controls = _random_control_distribution(
        node_scores, k, seed=seed + 7000, count=profile.random_control_count
    )

    residual_power = _energy(residual) / max(_energy(full), 1e-12)
    status_gates = {
        "finite_residual_power_ge_0_05": residual_power >= 0.05,
        "t2_beats_t1": _mse(t2, full) < _mse(t1, full),
        "residual_circuit_ap_ge_0_90": (
            ranking_metrics["residual_causal_fraction"]["average_precision"] >= 0.90
        ),
        "residual_circuit_precision_at_k_ge_0_80": (
            ranking_metrics["residual_causal_fraction"]["precision_at_truth_k"] >= 0.80
        ),
        "matched_random_specificity": (
            controls["candidate_topk_total_residual_causal_fraction"] > controls["random_p95"]
        ),
        "gauge_function_invariant": gauge_function_max_abs_error <= 2e-5,
        "gauge_causal_score_invariant": gauge_causal_max_diff <= 2e-4,
        "student_validation_improvement_ge_0_50": (
            student_metrics["validation_improvement_fraction"] >= 0.50
        ),
        "student_test_improvement_ge_0_50": student_metrics["test_improvement_fraction"] >= 0.50,
        "student_full_replay_nmse_le_0_20": student_metrics["test_full_effect_replay_nmse"] <= 0.20,
    }
    status = "SMOKE_VALIDATED" if all(status_gates.values()) else "NEGATIVE_RESULT"

    elapsed = time.perf_counter() - start
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_capability": (
            torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        ),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "elapsed_seconds": elapsed,
    }
    result: dict[str, object] = {
        "schema_version": "crct_stage0_circuit_recovery_v1",
        "status": status,
        "seed": seed,
        "profile": asdict(profile),
        "runtime": runtime,
        "ground_truth": {
            "active_nodes": sorted(truth),
            "active_node_count": len(truth),
            "interpretation": (
                "sparse finite nonlinear route+bypass coordinates; linear/quadratic paths are "
                "differential baseline, nuisance coordinates are disconnected"
            ),
        },
        "differential_diagnostics": {
            "full_effect_energy": _energy(full),
            "first_order_nmse": _nmse(t1, full),
            "second_order_nmse": _nmse(t2, full),
            "residual_energy": _energy(residual),
            "residual_power_fraction": residual_power,
        },
        "node_scores": node_scores,
        "rankings": ranking_metrics,
        "random_matched_controls": controls,
        "gauge_diagnostics": {
            "function_max_abs_error": gauge_function_max_abs_error,
            "route_activation_rank_spearman": _spearman_from_scores(
                original_activation, gauged_activation
            ),
            "route_residual_causal_rank_spearman": _spearman_from_scores(
                original_causal, gauged_causal
            ),
            "route_residual_causal_max_abs_score_difference": gauge_causal_max_diff,
            "gauge_scale_min": float(gauge_scales.min().cpu()),
            "gauge_scale_max": float(gauge_scales.max().cpu()),
        },
        "residual_student": student_metrics,
        "gates": status_gates,
    }
    result["result_sha256"] = _sha256_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Causal-Residual Circuit Tracing Stage-0 benchmark"
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_benchmark(
        profile_name=args.profile,
        seed=args.seed,
        device_name=args.device,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": result["seed"],
                "profile": result["profile"]["name"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output),
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "SMOKE_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
