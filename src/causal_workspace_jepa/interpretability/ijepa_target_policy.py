"""Fair-baseline policy for future Intervention-JEPA residual claims.

HARD-002 found that a capacity-matched direct-delta MLP beat a learned-residual MLP
on every primary seed, IID and OOD. Residual learners are therefore not privileged
by architecture. This module encodes the prospective eligibility rule; it does not
rerun HARD-002 or change its frozen negative result.
"""

from __future__ import annotations

from typing import Mapping, Sequence

REQUIRED_IJEPA_BASELINES: tuple[str, ...] = (
    "no_change",
    "mean_effect",
    "jvp_first_order",
    "t2_quadratic",
    "relinearized_jvp",
    "direct_delta_capacity_matched",
    "differential_plus_learned_residual",
)

DEFAULT_RESIDUAL_POWER_FLOOR = 0.08
HARD002_STATUS = "NEGATIVE_RESULT"
HARD002_DIRECT_DELTA_BEAT_RESIDUAL = True


def missing_required_baselines(present: Sequence[str]) -> list[str]:
    have = set(present)
    return [name for name in REQUIRED_IJEPA_BASELINES if name not in have]


def residual_power_eligible(
    residual_power: float,
    *,
    floor: float = DEFAULT_RESIDUAL_POWER_FLOOR,
) -> bool:
    return float(residual_power) >= float(floor)


def learned_residual_claim_eligible(
    *,
    residual_power: float,
    residual_power_floor: float = DEFAULT_RESIDUAL_POWER_FLOOR,
    differential_plus_residual_heldout_nmse: float,
    direct_delta_heldout_nmse: float,
    present_baselines: Sequence[str],
    original_model_replay_passed: bool,
    residual_stable_across_seeds: bool,
) -> dict[str, object]:
    """Return a fail-closed eligibility record for a learned residual claim."""

    missing = missing_required_baselines(present_baselines)
    power_ok = residual_power_eligible(residual_power, floor=residual_power_floor)
    beats_direct = float(differential_plus_residual_heldout_nmse) < float(direct_delta_heldout_nmse)
    eligible = (
        not missing
        and power_ok
        and beats_direct
        and original_model_replay_passed
        and residual_stable_across_seeds
    )
    return {
        "eligible": eligible,
        "missing_baselines": missing,
        "residual_power": float(residual_power),
        "residual_power_floor": float(residual_power_floor),
        "residual_power_ok": power_ok,
        "beats_direct_delta": beats_direct,
        "original_model_replay_passed": bool(original_model_replay_passed),
        "residual_stable_across_seeds": bool(residual_stable_across_seeds),
        "hard002_status_preserved": HARD002_STATUS,
        "hard002_direct_delta_beat_residual": HARD002_DIRECT_DELTA_BEAT_RESIDUAL,
        "rule": (
            "A learned residual branch is eligible only when residual power exceeds the "
            "frozen floor, fair baselines are present, differential+residual beats "
            "direct-delta on held-out metrics, the residual is seed-stable, and original-model "
            "replay succeeds. HARD-002 remains NEGATIVE_RESULT."
        ),
    }


def assert_no_privileged_residual(metrics: Mapping[str, float]) -> None:
    """Refuse to treat residual NMSE as success if direct-delta is better."""

    residual = metrics.get("residual_mlp_nmse")
    direct = metrics.get("direct_delta_mlp_nmse")
    if residual is None or direct is None:
        raise KeyError("both residual_mlp_nmse and direct_delta_mlp_nmse are required")
    if float(residual) > float(direct):
        raise ValueError(
            "direct-delta baseline is better than the residual learner; residual is not privileged"
        )
