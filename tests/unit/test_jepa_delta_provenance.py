"""Per-stage provenance for CRCT-JEPA-ACTION-DELTA-001.

IBD-003 confirmation sidecars fused development+confirmation into one command
and reused development seed 31. This experiment must not repeat that pattern.
Do not rewrite IBD-003 artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from causal_workspace_jepa.common.provenance import (
    collect_provenance,
    stage_cli_command,
    write_provenance,
    write_stage_provenance,
)
from causal_workspace_jepa.experiments.cross_domain.crct_jepa_action_delta import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    EXPERIMENT_ID,
    MODULE,
    _authorize_confirmation,
    source_digest,
    threshold_digest,
)


def test_stage_cli_command_is_single_stage() -> None:
    development = stage_cli_command(MODULE, "development", "artifacts/metrics/crct_jepa_action_delta_v1.dev.json")
    confirmation = stage_cli_command(MODULE, "confirmation", "artifacts/metrics/crct_jepa_action_delta_v1.json")
    assert "--stage development" in development
    assert "--stage confirmation" in confirmation
    assert "--require-development" in confirmation
    assert "--stage development" not in confirmation
    assert "--stage confirmation" not in development
    assert " && " not in development
    assert " && " not in confirmation


def test_confirmation_sidecar_uses_confirmation_seed_and_stage(tmp_path: Path) -> None:
    metrics = tmp_path / "crct_jepa_action_delta_v1.json"
    sidecar = tmp_path / "crct_jepa_action_delta_v1.provenance.json"
    write_stage_provenance(
        sidecar,
        module=MODULE,
        stage="confirmation",
        output=metrics.as_posix(),
        experiment_id=EXPERIMENT_ID,
        seeds=CONFIRMATION_SEEDS,
        resource_profile="configs/resource/cpu_vps.yaml",
        extra={"require_development": "artifacts/metrics/crct_jepa_action_delta_v1.dev.json"},
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["seed"] == 1013
    assert payload["seed"] not in DEVELOPMENT_SEEDS
    assert payload["seeds"] == [1013, 1019, 1021]
    assert payload["stage"] == "confirmation"
    assert payload["command_stage"] == "confirmation"
    assert "--stage confirmation" in payload["command"]
    assert "--require-development" in payload["command"]
    assert "--stage development" not in payload["command"]
    assert " && " not in payload["command"]


def test_collect_before_write_keeps_stages_separate() -> None:
    command = stage_cli_command(MODULE, "confirmation", "artifacts/metrics/crct_jepa_action_delta_v1.json")
    provenance = collect_provenance(command, "configs/resource/cpu_vps.yaml", seed=CONFIRMATION_SEEDS[0])
    assert provenance.seed == 1013
    assert "--stage confirmation" in provenance.command
    assert "--stage development" not in provenance.command


def test_authorize_confirmation_rejects_non_development(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "confirmation",
                "status": "MECHANISM_RECOVERY_PASSED",
                "all_seeds_passed": True,
                "seeds": list(DEVELOPMENT_SEEDS),
                "threshold_digest": threshold_digest(),
                "source_digest": source_digest(),
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(path.as_posix())
    except ValueError:
        return
    raise AssertionError("confirmation artifact was accepted as development")


def test_authorize_confirmation_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "stage": "development",
                "status": "MECHANISM_RECOVERY_PASSED",
                "all_seeds_passed": True,
                "seeds": list(DEVELOPMENT_SEEDS),
                "threshold_digest": "not-the-digest",
                "source_digest": source_digest(),
            }
        ),
        encoding="utf-8",
    )
    try:
        _authorize_confirmation(path.as_posix())
    except ValueError:
        return
    raise AssertionError("mismatched threshold digest was accepted")


def test_write_provenance_extra_does_not_overwrite_command(tmp_path: Path) -> None:
    sidecar = tmp_path / "x.provenance.json"
    command = stage_cli_command(MODULE, "development", "artifacts/metrics/crct_jepa_action_delta_v1.dev.json")
    write_provenance(
        sidecar,
        collect_provenance(command, "configs/resource/cpu_vps.yaml", seed=DEVELOPMENT_SEEDS[0]),
        extra={
            "experiment_id": EXPERIMENT_ID,
            "stage": "development",
            "metrics": "artifacts/metrics/crct_jepa_action_delta_v1.dev.json",
        },
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["command"] == command
    assert payload["stage"] == "development"
    assert payload["seed"] == 43
