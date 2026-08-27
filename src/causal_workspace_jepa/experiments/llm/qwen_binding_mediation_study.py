"""Fail-closed orchestration for the frozen Qwen binding-mediation study.

The study deliberately has three separately persisted phases:

``calibration``
    validates the captured causal dataset and records a measured unit cost;
``train_plan``
    uses train rows only to freeze rankings, the population prefix, matched
    random sets, and every analysis contract;
``protected_eval``
    accepts only a byte-identical, git-tracked plan and aggregates directly
    executed outcomes on the protected splits.

This module contains the integrity, budgeting, freezing, resume, aggregation,
decision, and direct-execution layer. Every protected outcome is materialized
as an independently checksummed unit. Specificity controls retain the primary
clean-to-treated denominator and never substitute a superficially similar
intervention.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from causal_workspace_jepa.common.config import load_config
from causal_workspace_jepa.common.types import InterventionSpec
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_capture import (
    assert_capture_not_terminally_closed,
    capture_content_digest,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_evaluator import (
    DirectMediationOutcome,
    FrozenRanking,
    atp_star_graddrop_scores,
    autograd_episode_estimators,
    binding_candidate_nodes,
    compare_population_mediation,
    compare_specificity_controls,
    delta_norm_scores,
    decide_h_llm_15,
    decide_h_llm_16,
    deterministic_random_scores,
    directional_hvp_scores,
    exact_local_atp_scores,
    execute_direct_mediation_episode,
    freeze_ranking,
    leave_value_out_probe_scores,
    matched_random_sets,
    norm_matched_resampled_states,
    population_atp_scores,
)
from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_protocol import (
    MediationEstimate,
    binding_episodes_from_config,
    mediation_estimate,
    render_binding_prompt,
    select_train_prefix,
)


PHASES = ("calibration", "train_plan", "protected_eval")
PROTECTED_SPLITS = ("test", "paraphrase")
REQUIRED_COMPARATORS = (
    "exact_local_atp",
    "directional_hvp",
    "atp_star",
    "leave_value_out_probe",
    "delta_norm",
)
REQUIRED_CONTROLS = (
    "donor_shuffle",
    "norm_matched_resample",
    "irrelevant_position",
    "unqueried_value_swap",
)
PLAN_SCHEMA_VERSION = 2
PROGRESS_SCHEMA_VERSION = 1
PROTECTED_CODE_PATHS = (
    "scripts/run_qwen_binding_mediation_study.py",
    # Freeze the whole package rather than guessing a fragile transitive import
    # closure.  This is deliberately conservative: any later source change
    # requires regenerating and recommitting the train-only plan.
    "src/causal_workspace_jepa",
)


@dataclass(frozen=True)
class WorkEstimate:
    """Conservative operation counts known before protected computation."""

    calibration_episodes: int
    train_episodes: int
    protected_episodes: int
    derivative_backward_calls_per_episode: int
    calibration_backward_calls: int
    train_backward_calls: int
    train_prefix_direct_forward_calls: int
    protected_direct_sets_per_episode: int
    protected_direct_forward_calls: int
    projected_train_seconds: float | None
    projected_protected_seconds: float | None

    @property
    def projected_total_gpu_hours(self) -> float | None:
        if self.projected_train_seconds is None or self.projected_protected_seconds is None:
            return None
        return (self.projected_train_seconds + self.projected_protected_seconds) / 3600.0


@dataclass(frozen=True)
class CaptureBundle:
    """Checksum-verified capture and the identity bound into a frozen plan."""

    arrays: Mapping[str, np.ndarray]
    records: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    manifest: Mapping[str, Any]
    identity: Mapping[str, str]


@dataclass(frozen=True)
class GitState:
    """Minimal git state required by protected evaluation."""

    commit: str
    dirty: bool
    tracked_plan_bytes: bytes | None
    source_commit_is_ancestor: bool = True
    changed_contract_paths: tuple[str, ...] = ()


class ProtectedOutcomeExecutor(Protocol):
    """Direct-execution boundary used by the protected orchestrator."""

    def supported_controls(self) -> Sequence[str]: ...

    def runtime_metadata(self) -> Mapping[str, Any]: ...

    def execute_split(
        self,
        *,
        split: str,
        plan: Mapping[str, Any],
        progress: Mapping[str, Any],
        progress_callback: Any,
    ) -> Mapping[str, Any]: ...


class QwenTrainExecutor:
    """Real Qwen derivative/direct executor for calibration and train planning."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import torch

        from causal_workspace_jepa.adapters.qwen_hf_adapter import (
            QwenAdapterConfig,
            QwenHFAdapter,
        )

        self._torch = torch
        self.config = config
        self.adapter = QwenHFAdapter(
            QwenAdapterConfig(
                model_name=str(config["model"]),
                revision=str(config["revision"]),
                device=str(config["device"]),
                dtype=str(config["dtype"]),
                max_length=int(config["max_sequence_length"]),
                local_files_only=bool(config["local_files_only"]),
                preserve_autograd=True,
                attn_implementation=str(config["attn_implementation"]),
                token=False,
            )
        )
        self.adapter.model.eval()
        self.adapter.model.requires_grad_(False)
        # A differentiable embedding output keeps the downstream graph alive;
        # autograd.grad targets candidate activations and never accumulates this
        # parameter's gradient.
        self.adapter.model.get_input_embeddings().weight.requires_grad_(True)
        if len(self.adapter.layers) != 28:
            raise RuntimeError("registered study requires the 28-layer Qwen3-0.6B target")
        self.sites = tuple(node.site for node in binding_candidate_nodes())

    def synchronize(self) -> None:
        if self._torch.cuda.is_available() and str(self.config["device"]).startswith("cuda"):
            self._torch.cuda.synchronize()

    def derivatives(
        self,
        episode: Any,
        record: Mapping[str, Any],
        clean_candidate: np.ndarray,
        treated_candidate: np.ndarray,
    ) -> Mapping[str, np.ndarray | float]:
        delta = np.asarray(treated_candidate, dtype=np.float32) - np.asarray(
            clean_candidate, dtype=np.float32
        )
        batch = self.adapter.tokenize([episode.recipient_prompt()])
        estimates = autograd_episode_estimators(
            self.adapter,
            batch,
            delta,
            recipient_answer_id=int(record["recipient_answer_id"]),
            donor_answer_id=int(record["donor_answer_id"]),
            candidate_sites=self.sites,
        )
        return {
            "local_gradients": estimates.local_gradients,
            "first_order_effects": estimates.first_order_effects,
            "directional_hvp_terms": estimates.directional_hvp_terms,
            "graddrop_effects": estimates.graddrop_effects,
            "clean_candidate": estimates.clean_candidate,
            "clean_score": estimates.clean_score,
        }

    def direct(
        self,
        episode: Any,
        record: Mapping[str, Any],
        sites: Sequence[str],
    ) -> DirectMediationOutcome:
        return execute_direct_mediation_episode(
            self.adapter,
            recipient_prompt=episode.recipient_prompt(),
            donor_prompt=episode.donor_prompt(),
            treatment_site=str(self.config["treatment"]["site"]),
            treatment_positions=tuple(int(value) for value in record["changed_positions"]),
            recipient_answer_id=int(record["recipient_answer_id"]),
            donor_answer_id=int(record["donor_answer_id"]),
            mediator_sites=sites,
            seed=int(self.config["seed"]),
        )

    def supported_controls(self) -> Sequence[str]:
        return ("donor_shuffle", "norm_matched_resample")


class QwenProtectedOutcomeExecutor:
    """Lazy, unit-checksummed executor for every frozen protected condition."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        adapter: Any | None = None,
        capture: CaptureBundle | None = None,
        unit_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self._adapter = adapter
        self._capture = capture
        self._episodes = binding_episodes_from_config(self.config)
        self._unit_root = Path(unit_root) if unit_root is not None else (
            Path(str(self.config["output_dir"])) / "study_progress" / "protected"
        )

    def supported_controls(self) -> Sequence[str]:
        return REQUIRED_CONTROLS

    def runtime_metadata(self) -> Mapping[str, Any]:
        import torch

        from causal_workspace_jepa.experiments.llm.qwen_binding_mediation_capture import (
            _runtime_fingerprint,
        )

        model = dict(self.adapter._metadata())
        resolved = str(model.get("resolved_revision"))
        if resolved != str(self.config["revision"]):
            raise RuntimeError("protected executor resolved a different model revision")
        runtime = _runtime_fingerprint(torch, model)
        expected = self.capture.metrics.get("capture_identity", {}).get("runtime")
        if not isinstance(expected, Mapping) or dict(expected) != runtime:
            raise RuntimeError("protected executor runtime differs from the frozen capture runtime")
        return {
            "model": str(model.get("model")),
            "requested_revision": str(model.get("requested_revision")),
            "resolved_revision": resolved,
            "device": str(model.get("device")),
            "dtype": str(model.get("dtype")),
            **runtime,
        }

    def bind_capture(self, capture: CaptureBundle) -> None:
        if self._capture is not None and dict(self._capture.identity) != dict(capture.identity):
            raise RuntimeError("protected executor was initialized with a different capture")
        self._capture = capture

    def execute_split(
        self,
        *,
        split: str,
        plan: Mapping[str, Any],
        progress: Mapping[str, Any],
        progress_callback: Any,
    ) -> Mapping[str, Any]:
        if split not in PROTECTED_SPLITS:
            raise ValueError(f"unregistered protected split: {split}")
        validate_frozen_plan(plan, config_path=self.config_path, capture=self.capture)
        capture = self.capture
        split_rows = [
            index for index, episode in enumerate(self._episodes) if episode.split == split
        ]
        if len(split_rows) != int(self.config["splits"][split]["count"]):
            raise RuntimeError(f"capture rows are incomplete for {split}")
        conditions = self._condition_roster(plan)
        expected_conditions = (
            1 + len(REQUIRED_COMPARATORS) + int(plan["matched_random_set_count"])
            + len(REQUIRED_CONTROLS)
        )
        if len(conditions) != expected_conditions:
            raise RuntimeError("protected condition roster is incomplete")
        completed = set(progress.get("conditions_complete", ()))
        unknown = completed.difference(name for name, _sites, _control in conditions)
        if unknown:
            raise RuntimeError(f"protected progress contains unknown conditions: {sorted(unknown)}")
        prepared = self._prepare_controls(split, split_rows, plan)
        results: dict[str, list[DirectMediationOutcome]] = {}
        for condition, sites, control in conditions:
            outcomes: list[DirectMediationOutcome] = []
            for local_index, row_index in enumerate(split_rows):
                episode = self._episodes[row_index]
                identity = {
                    "plan_sha256": str(plan["plan_sha256"]),
                    "source_git_commit": str(plan["source_git_commit"]),
                    "capture_content_sha256": str(capture.identity["capture_content_sha256"]),
                    "split": split,
                    "condition": condition,
                    "episode_id": episode.episode_id,
                }
                path = self._unit_path(identity)
                if path.exists():
                    outcome = load_direct_outcome_unit(path, identity)
                else:
                    outcome = self._execute_condition(
                        episode=episode,
                        record=capture.records[row_index],
                        row_index=row_index,
                        local_index=local_index,
                        sites=sites,
                        control=control,
                        prepared=prepared,
                    )
                    save_direct_outcome_unit(path, identity, outcome)
                outcomes.append(outcome)
            results[condition] = outcomes
            completed.add(condition)
            progress_callback(
                {
                    "conditions_complete": sorted(completed),
                    "units_complete": len(completed) * len(split_rows),
                    "units_expected": len(conditions) * len(split_rows),
                }
            )
        return {
            "outcomes": self._group_outcomes(results, capture, split_rows),
            "conditions_complete": sorted(completed),
            "unit_count": len(conditions) * len(split_rows),
        }

    @property
    def capture(self) -> CaptureBundle:
        if self._capture is None:
            self._capture = load_verified_capture(self.config_path)
        return self._capture

    @property
    def adapter(self) -> Any:
        if self._adapter is None:
            from causal_workspace_jepa.adapters.qwen_hf_adapter import (
                QwenAdapterConfig,
                QwenHFAdapter,
            )

            self._adapter = QwenHFAdapter(
                QwenAdapterConfig(
                    model_name=str(self.config["model"]),
                    revision=str(self.config["revision"]),
                    device=str(self.config["device"]),
                    dtype=str(self.config["dtype"]),
                    max_length=int(self.config["max_sequence_length"]),
                    local_files_only=bool(self.config["local_files_only"]),
                    preserve_autograd=False,
                    attn_implementation=str(self.config["attn_implementation"]),
                    token=False,
                )
            )
            self._adapter.model.eval()
        return self._adapter

    def _condition_roster(
        self, plan: Mapping[str, Any]
    ) -> list[tuple[str, tuple[str, ...], str | None]]:
        selected = plan["selected_sets"]
        conditions = [
            ("population_atp", tuple(selected["population_atp"]), None),
            *[
                (method, tuple(selected[method]), None)
                for method in REQUIRED_COMPARATORS
            ],
            *[
                (f"matched_random:{index:03d}", tuple(sites), None)
                for index, sites in enumerate(plan["matched_random_sets"])
            ],
            *[
                (f"control:{name}", tuple(selected["population_atp"]), name)
                for name in REQUIRED_CONTROLS
            ],
        ]
        if len({name for name, _sites, _control in conditions}) != len(conditions):
            raise RuntimeError("protected condition names are not unique")
        return conditions

    def _prepare_controls(
        self, split: str, split_rows: Sequence[int], plan: Mapping[str, Any]
    ) -> dict[str, Any]:
        capture = self.capture
        roster = [node.site for node in binding_candidate_nodes()]
        indices = [roster.index(site) for site in plan["population_prefix"]["selected_sites"]]
        train_rows = [
            index for index, episode in enumerate(self._episodes) if episode.split == "train"
        ]
        clean = np.asarray(capture.arrays["clean_candidate"])[np.ix_(split_rows, indices)]
        treated = np.asarray(capture.arrays["treated_candidate"])[np.ix_(split_rows, indices)]
        train_clean = np.asarray(capture.arrays["clean_candidate"])[np.ix_(train_rows, indices)]
        train_treated = np.asarray(capture.arrays["treated_candidate"])[np.ix_(train_rows, indices)]
        target_delta = treated - clean
        train_delta = train_treated - train_clean
        seed = int(self.config["controls"]["norm_resample_seed"])
        bins = int(self.config["controls"]["norm_resample_bins"])
        return {
            "site_indices": indices,
            "norm_sufficient": norm_matched_resampled_states(
                clean, target_delta, train_delta, seed=seed, bins=bins
            ),
            "norm_restore": norm_matched_resampled_states(
                treated, -target_delta, -train_delta, seed=seed, bins=bins
            ),
            "shuffle": dict(plan["control_assignments"]["donor_shuffle"][split]),
            "row_by_id": {
                self._episodes[index].episode_id: index for index in split_rows
            },
        }

    def _execute_condition(
        self,
        *,
        episode: Any,
        record: Mapping[str, Any],
        row_index: int,
        local_index: int,
        sites: Sequence[str],
        control: str | None,
        prepared: Mapping[str, Any],
    ) -> DirectMediationOutcome:
        kwargs: dict[str, Any] = {}
        capture = self.capture
        if control == "donor_shuffle":
            donor_id = prepared["shuffle"][episode.episode_id]
            donor_row = prepared["row_by_id"][donor_id]
            indices = prepared["site_indices"]
            kwargs["sufficient_states"] = {
                site: capture.arrays["treated_candidate"][donor_row, index]
                for site, index in zip(sites, indices, strict=True)
            }
            kwargs["restore_states"] = {
                site: capture.arrays["clean_candidate"][donor_row, index]
                for site, index in zip(sites, indices, strict=True)
            }
        elif control == "norm_matched_resample":
            kwargs["sufficient_states"] = {
                site: prepared["norm_sufficient"][local_index, offset]
                for offset, site in enumerate(sites)
            }
            kwargs["restore_states"] = {
                site: prepared["norm_restore"][local_index, offset]
                for offset, site in enumerate(sites)
            }
        elif control == "irrelevant_position":
            kwargs["mediator_position"] = int(
                self.config["controls"]["irrelevant_position_index"]
            )
        elif control == "unqueried_value_swap":
            states = self._unqueried_control_states(episode, sites)
            kwargs["sufficient_states"] = states
            kwargs["restore_states"] = states
        elif control is not None:
            raise RuntimeError(f"unknown protected control {control}")
        return execute_direct_mediation_episode(
            self.adapter,
            recipient_prompt=episode.recipient_prompt(),
            donor_prompt=episode.donor_prompt(),
            treatment_site=str(self.config["treatment"]["site"]),
            treatment_positions=tuple(int(value) for value in record["changed_positions"]),
            recipient_answer_id=int(record["recipient_answer_id"]),
            donor_answer_id=int(record["donor_answer_id"]),
            mediator_sites=sites,
            seed=int(self.config["seed"]),
            **kwargs,
        )

    def _unqueried_control_states(
        self, episode: Any, sites: Sequence[str]
    ) -> dict[str, np.ndarray]:
        prompt, changed_positions = audited_unqueried_swap(self.adapter, episode)
        recipient_batch = self.adapter.tokenize([episode.recipient_prompt()])
        donor_batch = self.adapter.tokenize([prompt])
        treatment_site = str(self.config["treatment"]["site"])
        donor = self.adapter.forward_with_cache(donor_batch, [treatment_site, *sites, "logits"])
        donor_id = f"unqueried-control:{episode.episode_id}"
        self.adapter.register_donor(donor_id, treatment_site, donor.activations[treatment_site])
        spec = InterventionSpec(
            site=treatment_site,
            operation="patch",
            positions=changed_positions,
            donor_example_id=donor_id,
            seed=int(self.config["seed"]),
        )
        try:
            treated = self.adapter.forward_with_intervention(
                recipient_batch, spec, [*sites, "logits"]
            )
        finally:
            self.adapter.unregister_donor(donor_id, treatment_site)
        donor_logits = np.asarray(donor.logits, dtype=np.float32)
        treated_logits = np.asarray(treated.logits, dtype=np.float32)
        error = float(np.max(np.abs(donor_logits - treated_logits)))
        if error > float(self.config["gates"]["treatment_logit_replay_atol"]):
            raise RuntimeError("unqueried-value treatment failed exact donor replay")
        return {site: np.asarray(treated.activations[site][0, -1]) for site in sites}

    def _group_outcomes(
        self,
        results: Mapping[str, Sequence[DirectMediationOutcome]],
        capture: CaptureBundle,
        split_rows: Sequence[int],
    ) -> dict[str, Any]:
        return {
            "population_atp": [asdict(value) for value in results["population_atp"]],
            "comparators": {
                method: [asdict(value) for value in results[method]]
                for method in REQUIRED_COMPARATORS
            },
            "matched_random": [
                [asdict(value) for value in results[f"matched_random:{index:03d}"]]
                for index in range(len([name for name in results if name.startswith("matched_random:")]))
            ],
            "controls": {
                name: [asdict(value) for value in results[f"control:{name}"]]
                for name in REQUIRED_CONTROLS
            },
            "donor_answer_ids": [
                int(capture.arrays["donor_answer_id"][index]) for index in split_rows
            ],
        }

    def _unit_path(self, identity: Mapping[str, str]) -> Path:
        condition = identity["condition"].replace(":", "-")
        return (
            self._unit_root
            / identity["plan_sha256"]
            / identity["split"]
            / condition
            / f"{identity['episode_id']}.json"
        )


def run_calibration_phase(
    config_path: str | Path,
    output_path: str | Path,
    *,
    executor: Any | None = None,
    source_git_commit: str,
) -> dict[str, Any]:
    """Benchmark every calibration episode and persist resumable timings."""

    config_path = Path(config_path)
    config = load_config(config_path)
    capture = load_verified_capture(config_path)
    source_commit = _full_sha(source_git_commit, "source_git_commit")
    preflight_estimate = estimate_study_work(config)
    engine = executor or QwenTrainExecutor(config)
    episodes = binding_episodes_from_config(config)
    calibration_rows = [
        index for index, episode in enumerate(episodes) if episode.split == "calibration"
    ]
    if len(calibration_rows) != int(config["splits"]["calibration"]["count"]):
        raise RuntimeError("calibration capture row count differs from configuration")
    progress_path = Path(output_path).with_suffix(".progress.json")
    fingerprint = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "phase": "calibration",
        "config_file_sha256": capture.identity["config_file_sha256"],
        "capture_content_sha256": capture.identity["capture_content_sha256"],
        "source_git_commit": source_commit,
    }
    progress = load_progress(progress_path, fingerprint)
    timings = dict(progress.get("timings", {}))
    for index in calibration_rows:
        episode = episodes[index]
        if episode.episode_id in timings:
            _validate_timing(timings[episode.episode_id], episode.episode_id)
            continue
        engine.synchronize()
        started = time.perf_counter()
        engine.derivatives(
            episode,
            capture.records[index],
            capture.arrays["clean_candidate"][index],
            capture.arrays["treated_candidate"][index],
        )
        engine.synchronize()
        derivative_seconds = time.perf_counter() - started
        # One ordinary direct outcome is exactly five forwards in the frozen
        # evaluator; measuring the whole outcome avoids noisy one-forward setup.
        engine.synchronize()
        started = time.perf_counter()
        engine.direct(episode, capture.records[index], (binding_candidate_nodes()[0].site,))
        engine.synchronize()
        direct_seconds = time.perf_counter() - started
        row = {
            "derivative_seconds": float(derivative_seconds),
            "direct_outcome_seconds": float(direct_seconds),
            "direct_seconds_per_forward": float(direct_seconds / 5.0),
        }
        _validate_timing(row, episode.episode_id)
        timings[episode.episode_id] = row
        write_progress(progress_path, fingerprint, {"timings": timings})
    derivative_values = np.asarray(
        [timings[episodes[index].episode_id]["derivative_seconds"] for index in calibration_rows],
        dtype=np.float64,
    )
    forward_values = np.asarray(
        [
            timings[episodes[index].episode_id]["direct_seconds_per_forward"]
            for index in calibration_rows
        ],
        dtype=np.float64,
    )
    derivative_median = float(np.median(derivative_values))
    forward_median = float(np.median(forward_values))
    estimate = estimate_study_work(
        config,
        measured_derivative_seconds_per_episode=derivative_median,
        measured_direct_seconds_per_forward=forward_median,
    )
    result = {
        "schema_version": 1,
        "experiment_id": str(config["id"]),
        "phase": "calibration",
        "status": "CALIBRATION_COMPLETE",
        "evidence_level": "Availability",
        "source_git_commit": source_commit,
        "identity": dict(capture.identity),
        "episode_ids": [episodes[index].episode_id for index in calibration_rows],
        "timings": timings,
        "measured_derivative_seconds_per_episode": derivative_median,
        "measured_direct_seconds_per_forward": forward_median,
        "work_estimate": asdict(estimate),
        "preflight_operation_counts": asdict(preflight_estimate),
        "capture_hardware": capture.metrics.get("hardware"),
        "scientific_boundary": (
            "Calibration measures resource cost on the calibration split only; it does not "
            "rank mediators or open train/protected outcomes."
        ),
    }
    result["calibration_sha256"] = sha256_json(result)
    _atomic_json(Path(output_path), result)
    progress_path.unlink(missing_ok=True)
    return result


def run_train_plan_phase(
    config_path: str | Path,
    calibration_path: str | Path,
    plan_path: str | Path,
    *,
    max_gpu_hours: float,
    source_git_commit: str,
    executor: Any | None = None,
) -> dict[str, Any]:
    """Compute train-only estimators and direct prefix outcomes, then freeze plan."""

    config_path = Path(config_path)
    config = load_config(config_path)
    capture = load_verified_capture(config_path)
    calibration = _read_json(Path(calibration_path))
    _validate_calibration(calibration, capture, config)
    estimate = estimate_study_work(
        config,
        measured_derivative_seconds_per_episode=float(
            calibration["measured_derivative_seconds_per_episode"]
        ),
        measured_direct_seconds_per_forward=float(
            calibration["measured_direct_seconds_per_forward"]
        ),
    )
    assert_budget(estimate, max_gpu_hours=max_gpu_hours)
    source_commit = _full_sha(source_git_commit, "source_git_commit")
    engine = executor or QwenTrainExecutor(config)
    episodes = binding_episodes_from_config(config)
    train_rows = [index for index, episode in enumerate(episodes) if episode.split == "train"]
    if len(train_rows) != int(config["splits"]["train"]["count"]):
        raise RuntimeError("train capture row count differs from configuration")
    progress_path = Path(plan_path).with_suffix(".progress.json")
    fingerprint = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "phase": "train_plan",
        "config_file_sha256": capture.identity["config_file_sha256"],
        "capture_content_sha256": capture.identity["capture_content_sha256"],
        "calibration_sha256": str(calibration["calibration_sha256"]),
        "source_git_commit": source_commit,
    }
    progress = load_progress(progress_path, fingerprint)
    derivative_units = dict(progress.get("derivatives", {}))
    derivative_dir = Path(plan_path).with_suffix(".derivatives")
    derivative_dir.mkdir(parents=True, exist_ok=True)
    for index in train_rows:
        episode = episodes[index]
        unit_path = derivative_dir / f"{episode.episode_id}.npz"
        if episode.episode_id in derivative_units:
            load_derivative_unit(
                unit_path,
                fingerprint,
                episode.episode_id,
                str(derivative_units[episode.episode_id]),
            )
            continue
        value = engine.derivatives(
            episode,
            capture.records[index],
            capture.arrays["clean_candidate"][index],
            capture.arrays["treated_candidate"][index],
        )
        unit_sha = save_derivative_unit(unit_path, fingerprint, episode.episode_id, value)
        derivative_units[episode.episode_id] = unit_sha
        write_progress(progress_path, fingerprint, {**progress, "derivatives": derivative_units})
    stacked = _stack_derivatives(
        [
            load_derivative_unit(
                derivative_dir / f"{episodes[index].episode_id}.npz",
                fingerprint,
                episodes[index].episode_id,
                str(derivative_units[episodes[index].episode_id]),
            )
            for index in train_rows
        ]
    )
    delta = np.asarray(capture.arrays["treated_candidate"][train_rows], dtype=np.float32) - np.asarray(
        capture.arrays["clean_candidate"][train_rows], dtype=np.float32
    )
    treatment_effects = np.asarray(capture.arrays["treatment_effect"][train_rows], dtype=np.float64)
    donor_ids = np.asarray(capture.arrays["donor_answer_id"][train_rows], dtype=np.int64)
    ranking_cfg = config["ranking"]
    score_vectors = {
        "population_atp": population_atp_scores(delta, stacked["local_gradients"]),
        "exact_local_atp": exact_local_atp_scores(delta, stacked["local_gradients"]),
        "directional_hvp": directional_hvp_scores(
            stacked["first_order_effects"],
            stacked["directional_hvp_terms"],
            coefficient=float(ranking_cfg["directional_hvp_coefficient"]),
        ),
        "atp_star": atp_star_graddrop_scores(stacked["graddrop_effects"]),
        "leave_value_out_probe": leave_value_out_probe_scores(
            delta,
            treatment_effects,
            donor_ids,
            projection_dim=int(ranking_cfg["leave_value_out_probe"]["projection_dim"]),
            projection_seed=int(ranking_cfg["leave_value_out_probe"]["projection_seed"]),
            ridge=float(ranking_cfg["leave_value_out_probe"]["ridge"]),
        ),
        "delta_norm": delta_norm_scores(delta),
        "random": deterministic_random_scores(
            len(binding_candidate_nodes()), seed=int(ranking_cfg["random_seed"])
        ),
    }
    permutation_diagnostic = compute_answer_row_permutation_diagnostic(
        delta,
        stacked["local_gradients"],
        count=int(config["controls"]["answer_row_permutations"]),
        seed=int(config["controls"]["answer_row_permutation_seed"]),
    )
    rankings = {name: freeze_ranking(name, value) for name, value in score_vectors.items()}
    prefix_rows = dict(progress.get("prefix_outcomes", {}))
    estimates: dict[int, MediationEstimate] = {}
    maximum = int(ranking_cfg["maximum_nodes"])
    for size in range(1, maximum + 1):
        key = str(size)
        selected = rankings["population_atp"].ordered_sites[:size]
        rows = dict(prefix_rows.get(key, {}))
        for index in train_rows:
            episode = episodes[index]
            if episode.episode_id in rows:
                _outcomes([rows[episode.episode_id]], f"train.prefix.{size}")
                continue
            outcome = engine.direct(episode, capture.records[index], selected)
            rows[episode.episode_id] = asdict(outcome)
            prefix_rows[key] = rows
            write_progress(
                progress_path,
                fingerprint,
                {"derivatives": derivative_units, "prefix_outcomes": prefix_rows},
            )
        ordered = _outcomes(
            [rows[episodes[index].episode_id] for index in train_rows],
            f"train.prefix.{size}",
        )
        estimates[size] = mediation_estimate(
            [value.clean_score for value in ordered],
            [value.treated_score for value in ordered],
            [value.sufficient_score for value in ordered],
            [value.restored_score for value in ordered],
        )
    plan = build_frozen_train_plan(
        config_path,
        capture,
        rankings,
        estimates,
        source_git_commit=source_commit,
        calibration_sha256=str(calibration["calibration_sha256"]),
        answer_row_permutation_diagnostic=permutation_diagnostic,
    )
    write_frozen_plan(plan_path, plan)
    progress_path.unlink(missing_ok=True)
    for unit in derivative_dir.glob("*.npz"):
        unit.unlink()
    derivative_dir.rmdir()
    return plan


def estimate_study_work(
    config: Mapping[str, Any],
    *,
    measured_derivative_seconds_per_episode: float | None = None,
    measured_direct_seconds_per_forward: float | None = None,
) -> WorkEstimate:
    """Return preregistered operation counts without touching the model.

    A derivative episode performs one first-gradient call, one diagonal HVP
    call per candidate, and one GradDrop call per candidate.  A direct
    mediation outcome performs five forwards.  The protected count includes
    population, every registered comparator, all matched random sets, and all
    four specificity controls.
    """

    if tuple(config.get("evaluation", {}).get("direct_splits", ())) != PROTECTED_SPLITS:
        raise ValueError("direct evaluation splits differ from the frozen protected roster")
    if config.get("evaluation", {}).get("validation_role") != "capture_task_eligibility_only":
        raise ValueError("validation role differs from the frozen protected protocol")
    nodes = int(config["candidates"]["node_count"])
    if nodes != len(binding_candidate_nodes()):
        raise ValueError("candidate node count differs from the frozen 56-node roster")
    calibration = int(config["splits"]["calibration"]["count"])
    train = int(config["splits"]["train"]["count"])
    protected = sum(int(config["splits"][split]["count"]) for split in PROTECTED_SPLITS)
    max_nodes = int(config["ranking"]["maximum_nodes"])
    random_sets = int(config["controls"]["random_sets"])
    derivative_calls = 1 + 2 * nodes
    train_prefix_forwards = train * max_nodes * 5
    direct_sets = 1 + len(REQUIRED_COMPARATORS) + random_sets + len(REQUIRED_CONTROLS)
    # The unqueried-value control additionally executes its donor and audited
    # upstream replay before the common five-forward mediation program.
    protected_forwards = protected * (direct_sets * 5 + 2)
    derivative_seconds = _optional_positive(
        measured_derivative_seconds_per_episode,
        "measured_derivative_seconds_per_episode",
    )
    forward_seconds = _optional_positive(
        measured_direct_seconds_per_forward,
        "measured_direct_seconds_per_forward",
    )
    return WorkEstimate(
        calibration_episodes=calibration,
        train_episodes=train,
        protected_episodes=protected,
        derivative_backward_calls_per_episode=derivative_calls,
        calibration_backward_calls=calibration * derivative_calls,
        train_backward_calls=train * derivative_calls,
        train_prefix_direct_forward_calls=train_prefix_forwards,
        protected_direct_sets_per_episode=direct_sets,
        protected_direct_forward_calls=protected_forwards,
        projected_train_seconds=(
            train * derivative_seconds + train_prefix_forwards * forward_seconds
            if derivative_seconds is not None and forward_seconds is not None
            else None
        ),
        projected_protected_seconds=(
            protected_forwards * forward_seconds if forward_seconds is not None else None
        ),
    )


def load_verified_capture(config_path: str | Path) -> CaptureBundle:
    """Load a complete eligible capture and verify every bound digest."""

    from causal_workspace_jepa.data.activation_store import read_hdf5_shards

    config_path = Path(config_path)
    config = load_config(config_path)
    assert_capture_not_terminally_closed(config, config_path=config_path)
    config_sha = sha256_file(config_path)
    semantic_config_digest = capture_config_digest(config)
    metrics_path = _capture_metrics_path(config)
    manifest_path = Path(str(config["output_manifest"]))
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("BLOCKED_CAPTURE: capture metrics and manifest must both exist")
    metrics = _read_json(metrics_path)
    manifest = _read_json(manifest_path)
    if metrics.get("status") != "CAUSAL_DATASET_ELIGIBLE":
        raise RuntimeError("BLOCKED_CAPTURE: capture is not CAUSAL_DATASET_ELIGIBLE")
    if metrics.get("config_digest") != semantic_config_digest:
        raise RuntimeError("capture config digest differs from the current configuration")
    capture_digest = str(metrics.get("capture_digest", ""))
    content_digest = str(metrics.get("storage", {}).get("content_sha256", ""))
    if len(capture_digest) != 64 or len(content_digest) != 64:
        raise RuntimeError("capture metrics contain malformed digests")
    if manifest.get("capture_digest") != capture_digest:
        raise RuntimeError("capture manifest and metrics disagree on capture digest")
    if manifest.get("content_sha256") != content_digest:
        raise RuntimeError("capture manifest and metrics disagree on content digest")
    if not bool(metrics.get("storage", {}).get("readback_verified")):
        raise RuntimeError("capture did not record checksum-verified readback")
    local_root = Path(str(manifest.get("local_data_root", "")))
    if not local_root.is_dir():
        raise RuntimeError("BLOCKED_CAPTURE: local activation root is unavailable")
    arrays, records = read_hdf5_shards(local_root)
    observed_content = capture_content_digest(arrays, records)
    if observed_content != content_digest:
        raise RuntimeError("capture content checksum differs after HDF5 readback")
    episodes = binding_episodes_from_config(config)
    expected_ids = [episode.episode_id for episode in episodes]
    observed_ids = [str(record.get("example_id")) for record in records]
    if observed_ids != expected_ids:
        raise RuntimeError("capture rows differ from the canonical episode order")
    expected_rows = sum(int(spec["count"]) for spec in config["splits"].values())
    if len(records) != expected_rows or any(value.shape[0] != expected_rows for value in arrays.values()):
        raise RuntimeError("capture row count differs from the frozen split counts")
    return CaptureBundle(
        arrays=arrays,
        records=tuple(records),
        metrics=metrics,
        manifest=manifest,
        identity={
            "config_file_sha256": config_sha,
            "config_digest": semantic_config_digest,
            "capture_metrics_sha256": sha256_file(metrics_path),
            "capture_manifest_sha256": sha256_file(manifest_path),
            "capture_digest": capture_digest,
            "capture_content_sha256": content_digest,
        },
    )


def build_frozen_train_plan(
    config_path: str | Path,
    capture: CaptureBundle,
    rankings: Mapping[str, FrozenRanking],
    population_prefix_estimates: Mapping[int, MediationEstimate],
    *,
    source_git_commit: str,
    calibration_sha256: str,
    answer_row_permutation_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the train-only plan; protected outcomes are not accepted here."""

    config_path = Path(config_path)
    config = load_config(config_path)
    _assert_capture_matches_config(config_path, capture)
    required = tuple(str(value) for value in config["ranking"]["methods"])
    if set(rankings) != set(required):
        raise ValueError("rankings must contain exactly the registered methods")
    roster = tuple(node.site for node in binding_candidate_nodes())
    for name, ranking in rankings.items():
        if ranking.method != name:
            raise ValueError(f"ranking key/method mismatch for {name}")
        if len(ranking.ordered_sites) != len(roster) or set(ranking.ordered_sites) != set(roster):
            raise ValueError(f"ranking {name} is not a permutation of the frozen roster")
        if set(ranking.scores) != set(roster) or not all(
            np.isfinite(value) for value in ranking.scores.values()
        ):
            raise ValueError(f"ranking {name} has missing or nonfinite scores")
    population = rankings["population_atp"]
    selection = select_train_prefix(
        population.ordered_sites,
        population_prefix_estimates,
        max_nodes=int(config["ranking"]["maximum_nodes"]),
        sufficiency_min=float(config["ranking"]["prefix_sufficiency_min"]),
        necessity_min=float(config["ranking"]["prefix_necessity_min"]),
    )
    k = len(selection.selected)
    if k <= 0:
        raise RuntimeError("population prefix selection produced an empty set")
    selected_sets = {
        name: list(ranking.ordered_sites[:k]) for name, ranking in rankings.items()
    }
    random_sets = matched_random_sets(
        selection.selected,
        count=int(config["controls"]["random_sets"]),
        seed=int(config["controls"]["random_seed"]),
    )
    permutation_contract = {
        "count": int(config["controls"]["answer_row_permutations"]),
        "seed": int(config["controls"]["answer_row_permutation_seed"]),
        "semantics": "permute_complete_train_episode_gradient_rows_relative_to_deltas",
    }
    permutation_contract["index_sha256"] = _permutation_digest(
        int(config["splits"]["train"]["count"]),
        permutation_contract["count"],
        permutation_contract["seed"],
    )
    permutation_diagnostic = _validate_permutation_diagnostic(
        answer_row_permutation_diagnostic,
        expected_count=permutation_contract["count"],
    )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment_id": str(config["id"]),
        "phase": "train_plan",
        "status": "FROZEN_TRAIN_ONLY_PLAN",
        "frozen": True,
        "source_git_commit": _full_sha(source_git_commit, "source_git_commit"),
        "calibration_sha256": _full_sha(calibration_sha256, "calibration_sha256"),
        "identity": dict(capture.identity),
        "train_contract": {
            "split": "train",
            "episodes": int(config["splits"]["train"]["count"]),
            "protected_splits_opened": False,
            "behavioral_score": str(config["ranking"]["behavioral_score"]),
            "score_aggregation": str(config["ranking"]["score_aggregation"]),
        },
        "rankings": {
            name: {
                "ordered_sites": list(ranking.ordered_sites),
                "scores": {site: float(ranking.scores[site]) for site in roster},
            }
            for name, ranking in sorted(rankings.items())
        },
        "population_prefix": {
            "k": k,
            "selected_sites": list(selection.selected),
            "eligible": selection.eligible,
            "reason": selection.reason,
            "train_estimates": {
                str(size): asdict(estimate)
                for size, estimate in sorted(population_prefix_estimates.items())
            },
        },
        "selected_sets": selected_sets,
        "matched_random_sets": [list(value) for value in random_sets],
        "matched_random_set_count": len(random_sets),
        "control_assignments": {
            "donor_shuffle": {
                split: _derangement_assignment(
                    [
                        episode.episode_id
                        for episode in binding_episodes_from_config(config)
                        if episode.split == split
                    ],
                    seed=int(config["controls"]["donor_shuffle_seed"]),
                )
                for split in PROTECTED_SPLITS
            }
        },
        "answer_row_permutation": permutation_contract,
        "answer_row_permutation_diagnostic": permutation_diagnostic,
        "direct_control_contracts": _direct_control_contracts(config),
        "inference_contract": {
            "bootstrap_draws": int(config["inference"]["bootstrap_draws"]),
            "bootstrap_seed": int(config["inference"]["bootstrap_seed"]),
            "bootstrap_denominator": str(config["inference"]["bootstrap_denominator"]),
            "minimum_eligible_fraction": float(
                config["inference"]["bootstrap_min_eligible_fraction"]
            ),
            "monte_carlo_alpha": float(config["inference"]["monte_carlo_alpha"]),
        },
        "scientific_boundary": (
            "Rankings, k, controls, and random sets use train data only. This plan contains "
            "no protected outcomes and makes no mediation, circuit, or workspace claim."
        ),
    }
    payload["plan_sha256"] = sha256_json(payload)
    return payload


def write_frozen_plan(path: str | Path, plan: Mapping[str, Any]) -> None:
    """Atomically write a self-hashed frozen plan."""

    validate_frozen_plan(plan)
    _atomic_json(Path(path), plan)


def validate_frozen_plan(
    plan: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    capture: CaptureBundle | None = None,
) -> None:
    """Reject tampered, incomplete, non-train-only, or stale plans."""

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise RuntimeError("unknown binding plan schema")
    if plan.get("status") != "FROZEN_TRAIN_ONLY_PLAN" or plan.get("frozen") is not True:
        raise RuntimeError("binding plan is not frozen")
    expected = str(plan.get("plan_sha256", ""))
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    if len(expected) != 64 or sha256_json(unsigned) != expected:
        raise RuntimeError("binding plan self-hash mismatch")
    contract = plan.get("train_contract", {})
    if contract.get("split") != "train" or contract.get("protected_splits_opened") is not False:
        raise RuntimeError("binding plan is not demonstrably train-only")
    prefix = plan.get("population_prefix", {})
    selected = tuple(prefix.get("selected_sites", ()))
    if not selected or int(prefix.get("k", 0)) != len(selected):
        raise RuntimeError("binding plan has an invalid population prefix")
    roster = {node.site for node in binding_candidate_nodes()}
    selected_sets = plan.get("selected_sets")
    expected_methods = {
        "population_atp",
        "exact_local_atp",
        "directional_hvp",
        "atp_star",
        "leave_value_out_probe",
        "delta_norm",
        "random",
    }
    if not isinstance(selected_sets, Mapping) or set(selected_sets) != expected_methods:
        raise RuntimeError("binding plan selected-set roster is incomplete")
    if any(
        len(value) != len(selected)
        or len(set(value)) != len(selected)
        or not set(value) <= roster
        for value in selected_sets.values()
    ):
        raise RuntimeError("binding plan selected sets have invalid cardinality or sites")
    rankings = plan.get("rankings")
    if not isinstance(rankings, Mapping) or set(rankings) != expected_methods:
        raise RuntimeError("binding plan ranking roster is incomplete")
    for method in expected_methods:
        ranking = rankings[method]
        if not isinstance(ranking, Mapping):
            raise RuntimeError(f"binding plan ranking {method} is malformed")
        ordered = ranking.get("ordered_sites")
        scores = ranking.get("scores")
        if (
            not isinstance(ordered, Sequence)
            or len(ordered) != len(roster)
            or set(ordered) != roster
            or not isinstance(scores, Mapping)
            or set(scores) != roster
            or not all(np.isfinite(float(value)) for value in scores.values())
            or list(selected_sets[method]) != list(ordered[: len(selected)])
        ):
            raise RuntimeError(f"binding plan ranking {method} violates the frozen contract")
    if list(selected) != list(selected_sets["population_atp"]):
        raise RuntimeError("population prefix and population selected set disagree")
    random_sets = plan.get("matched_random_sets")
    random_count = plan.get("matched_random_set_count")
    if (
        not isinstance(random_count, int)
        or random_count <= 0
        or not isinstance(random_sets, Sequence)
        or len(random_sets) != random_count
    ):
        raise RuntimeError("binding plan matched-random set count differs from the frozen contract")
    if any(
        not isinstance(value, Sequence)
        or len(value) != len(selected)
        or len(set(value)) != len(selected)
        or not set(value) <= roster
        for value in random_sets
    ):
        raise RuntimeError("binding plan contains a malformed matched-random set")
    assignments = plan.get("control_assignments", {}).get("donor_shuffle", {})
    if not isinstance(assignments, Mapping) or set(assignments) != set(PROTECTED_SPLITS):
        raise RuntimeError("binding plan donor-shuffle assignments are incomplete")
    for split, mapping in assignments.items():
        if (
            not isinstance(mapping, Mapping)
            or not mapping
            or set(mapping) != set(mapping.values())
            or any(source == donor for source, donor in mapping.items())
        ):
            raise RuntimeError(f"binding plan donor-shuffle assignment is not a derangement: {split}")
    controls = plan.get("direct_control_contracts")
    if not isinstance(controls, Mapping) or set(controls) != set(REQUIRED_CONTROLS):
        raise RuntimeError("binding plan direct-control contracts are incomplete")
    permutation = plan.get("answer_row_permutation")
    if not isinstance(permutation, Mapping):
        raise RuntimeError("binding plan answer-row permutation contract is missing")
    try:
        _validate_permutation_diagnostic(
            plan.get("answer_row_permutation_diagnostic", {}),
            expected_count=int(permutation["count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("binding plan answer-row permutation diagnostic is invalid") from exc
    if config_path is not None:
        path = Path(config_path)
        if plan.get("identity", {}).get("config_file_sha256") != sha256_file(path):
            raise RuntimeError("binding plan was frozen against different config bytes")
        config = load_config(path)
        if plan.get("experiment_id") != config.get("id"):
            raise RuntimeError("binding plan experiment differs from configuration")
        if tuple(config.get("evaluation", {}).get("direct_splits", ())) != PROTECTED_SPLITS:
            raise RuntimeError("binding direct-evaluation splits differ from the frozen protocol")
        if config.get("evaluation", {}).get("validation_role") != "capture_task_eligibility_only":
            raise RuntimeError("binding validation role differs from the frozen protocol")
        random_contract = config["controls"]
        if not all(
            bool(random_contract.get(name))
            for name in (
                "random_draws_with_replacement",
                "preserve_module_family",
                "preserve_layer_quartile",
            )
        ):
            raise RuntimeError("binding matched-random structural contract is disabled")
        if random_count != int(config["controls"]["random_sets"]):
            raise RuntimeError("binding plan matched-random count differs from configuration")
        expected_random = [
            list(value)
            for value in matched_random_sets(
                selected,
                count=int(config["controls"]["random_sets"]),
                seed=int(config["controls"]["random_seed"]),
            )
        ]
        if list(random_sets) != expected_random:
            raise RuntimeError("binding plan matched-random sets differ from frozen generation")
        expected_assignments = {
            split: _derangement_assignment(
                [
                    episode.episode_id
                    for episode in binding_episodes_from_config(config)
                    if episode.split == split
                ],
                seed=int(config["controls"]["donor_shuffle_seed"]),
            )
            for split in PROTECTED_SPLITS
        }
        if dict(assignments) != expected_assignments:
            raise RuntimeError("binding plan donor-shuffle assignments differ from configuration")
        if dict(controls) != _direct_control_contracts(config):
            raise RuntimeError("binding plan direct-control contracts differ from configuration")
        expected_permutation = {
            "count": int(config["controls"]["answer_row_permutations"]),
            "seed": int(config["controls"]["answer_row_permutation_seed"]),
            "semantics": "permute_complete_train_episode_gradient_rows_relative_to_deltas",
        }
        expected_permutation["index_sha256"] = _permutation_digest(
            int(config["splits"]["train"]["count"]),
            expected_permutation["count"],
            expected_permutation["seed"],
        )
        if dict(permutation) != expected_permutation:
            raise RuntimeError("binding plan answer-row permutation differs from configuration")
    if capture is not None and dict(plan.get("identity", {})) != dict(capture.identity):
        raise RuntimeError("binding plan was frozen against different capture artifacts")


def assert_protected_git_state(
    plan_path: str | Path,
    state: GitState | None = None,
    *,
    source_git_commit: str | None = None,
) -> GitState:
    """Require a clean plan commit with unchanged protected-execution code."""

    path = Path(plan_path)
    source: str | None = None
    if state is None:
        source = _full_sha(
            source_git_commit
            if source_git_commit is not None
            else str(_read_json(path).get("source_git_commit", "")),
            "plan source_git_commit",
        )
        observed = inspect_git_state(path, source_git_commit=source)
    else:
        observed = state
    if observed.dirty:
        raise RuntimeError("protected evaluation requires a clean committed worktree")
    _full_sha(observed.commit, "git commit")
    if observed.tracked_plan_bytes is None:
        raise RuntimeError("protected evaluation requires the frozen plan to be git-tracked")
    if not path.is_file() or path.read_bytes() != observed.tracked_plan_bytes:
        raise RuntimeError("working plan bytes differ from the plan committed at HEAD")
    if source is None:
        _full_sha(
            source_git_commit
            if source_git_commit is not None
            else str(_read_json(path).get("source_git_commit", "")),
            "plan source_git_commit",
        )
    if not observed.source_commit_is_ancestor:
        raise RuntimeError("plan source commit is not an ancestor of protected HEAD")
    if observed.changed_contract_paths:
        raise RuntimeError(
            "protected evaluator code changed after train-plan freeze: "
            f"{list(observed.changed_contract_paths)}"
        )
    return observed


def inspect_git_state(plan_path: str | Path, *, source_git_commit: str) -> GitState:
    """Inspect git without allowing an untracked plan into protected evaluation."""

    path = Path(plan_path).resolve()
    root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError("frozen plan must live inside the git worktree") from exc
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        capture_output=True,
        check=False,
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_git_commit, commit],
        capture_output=True,
        check=False,
    ).returncode == 0
    changed = tuple(
        line.strip()
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", f"{source_git_commit}..{commit}", "--", *PROTECTED_CODE_PATHS],
            text=True,
        ).splitlines()
        if line.strip()
    ) if ancestor else ()
    return GitState(
        commit=commit,
        dirty=dirty,
        tracked_plan_bytes=result.stdout if result.returncode == 0 else None,
        source_commit_is_ancestor=ancestor,
        changed_contract_paths=changed,
    )


def assert_budget(estimate: WorkEstimate, *, max_gpu_hours: float | None) -> None:
    """Fail before computation when no measured projection or budget exists."""

    if estimate.projected_total_gpu_hours is None:
        raise RuntimeError("BLOCKED_BUDGET: calibration timings are required before computation")
    if max_gpu_hours is None or not np.isfinite(max_gpu_hours) or max_gpu_hours <= 0.0:
        raise RuntimeError("BLOCKED_BUDGET: a finite positive --max-gpu-hours is required")
    if estimate.projected_total_gpu_hours > max_gpu_hours:
        raise RuntimeError(
            "BLOCKED_BUDGET: projected study cost "
            f"{estimate.projected_total_gpu_hours:.3f} GPU hours exceeds {max_gpu_hours:.3f}"
        )


def aggregate_protected_outcomes(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    outcomes_by_split: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate complete directly executed outcomes and decide H-LLM-15/16."""

    validate_frozen_plan(plan)
    inference = config["inference"]
    gates = config["gates"]
    bootstrap_draws = int(inference["bootstrap_draws"])
    bootstrap_seed = int(inference["bootstrap_seed"])
    eligible_fraction = float(inference["bootstrap_min_eligible_fraction"])
    task_eligibility = _task_eligibility(config, outcomes_by_split)
    comparisons: dict[str, Any] = {}
    aggregates: dict[str, Any] = {}
    specificity: dict[str, Any] = {}
    serialized_splits: dict[str, Any] = {}
    for split in PROTECTED_SPLITS:
        payload = outcomes_by_split.get(split)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"protected outcomes missing split {split}")
        population = _outcomes(payload.get("population_atp"), f"{split}.population_atp")
        expected_rows = int(config["splits"][split]["count"])
        if len(population) != expected_rows:
            raise RuntimeError(
                f"{split} population outcome count {len(population)} != {expected_rows}"
            )
        comparators_raw = payload.get("comparators")
        if not isinstance(comparators_raw, Mapping) or set(comparators_raw) != set(REQUIRED_COMPARATORS):
            raise RuntimeError(f"{split} comparator outcomes are incomplete")
        comparators = {
            name: _outcomes(comparators_raw[name], f"{split}.comparators.{name}")
            for name in REQUIRED_COMPARATORS
        }
        if any(len(values) != expected_rows for values in comparators.values()):
            raise RuntimeError(f"{split} comparator outcome count differs from configuration")
        random_raw = payload.get("matched_random")
        if not isinstance(random_raw, Sequence) or len(random_raw) != int(config["controls"]["random_sets"]):
            raise RuntimeError(f"{split} matched-random outcomes are incomplete")
        random = [
            _outcomes(value, f"{split}.matched_random.{index}")
            for index, value in enumerate(random_raw)
        ]
        if any(len(values) != expected_rows for values in random):
            raise RuntimeError(f"{split} matched-random outcome count differs from configuration")
        controls_raw = payload.get("controls")
        if not isinstance(controls_raw, Mapping) or set(controls_raw) != set(REQUIRED_CONTROLS):
            raise RuntimeError(f"{split} direct specificity controls are incomplete")
        controls = {
            name: _outcomes(controls_raw[name], f"{split}.controls.{name}")
            for name in REQUIRED_CONTROLS
        }
        if any(len(values) != expected_rows for values in controls.values()):
            raise RuntimeError(f"{split} control outcome count differs from configuration")
        donor_ids = _donor_ids(payload, expected_rows, split)
        comparison = compare_population_mediation(
            population,
            comparators,
            random,
            donor_ids,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed,
            minimum_eligible_fraction=eligible_fraction,
            treatment_effect_signed_mean_min=float(gates["treatment_effect_signed_mean_min"]),
        )
        aggregate = comparison.population
        control_comparison = compare_specificity_controls(population, controls, donor_ids)
        comparisons[split] = comparison
        aggregates[split] = aggregate
        specificity[split] = control_comparison
        serialized_splits[split] = {
            "population_comparison": asdict(comparison),
            "mediation": asdict(aggregate),
            "specificity": asdict(control_comparison),
        }
    decision_15 = decide_h_llm_15(
        comparisons,
        task_eligibility=task_eligibility,
        population_prefix_eligible=bool(plan["population_prefix"]["eligible"]),
        qn_margin_min=float(gates["rank_min_qn_margin"]),
        paired_ci_lower_min=float(gates["rank_ci_lower_min"]),
        matched_random_margin_min=float(gates["rank_random_p99_margin_min"]),
        monte_carlo_alpha=float(inference["monte_carlo_alpha"]),
        required_bootstrap_draws=bootstrap_draws,
        minimum_bootstrap_eligible_fraction=eligible_fraction,
        required_matched_random_count=int(config["controls"]["random_sets"]),
    )
    decision_16 = decide_h_llm_16(
        aggregates,
        specificity,
        task_eligibility=task_eligibility,
        population_prefix_eligible=bool(plan["population_prefix"]["eligible"]),
        mediation_min=float(gates["mediation_sufficiency_min"]),
        mediation_ci_lower_min=float(gates["mediation_ci_lower_min"]),
        treatment_transfer_gap_max=float(gates["treatment_transfer_gap_max"]),
        restoration_transfer_reduction_min=float(gates["restoration_transfer_reduction_min"]),
        specificity_margin_fraction_min=float(gates["specificity_margin_fraction_min"]),
        required_bootstrap_draws=bootstrap_draws,
        minimum_bootstrap_eligible_fraction=eligible_fraction,
    )
    return {
        "status": "PROTECTED_EVALUATION_COMPLETE",
        "evidence_level": (
            "Specificity" if decision_15.passed or decision_16.passed else "Causal mediation"
        ),
        "plan_sha256": plan["plan_sha256"],
        "task_eligibility": task_eligibility,
        "splits": serialized_splits,
        "decisions": {"H-LLM-15": asdict(decision_15), "H-LLM-16": asdict(decision_16)},
        "scientific_boundary": (
            "These decisions concern population localization and compact direct mediation. "
            "They do not establish a global workspace or an Intervention-JEPA advantage."
        ),
    }


def run_protected_evaluation(
    config_path: str | Path,
    plan_path: str | Path,
    executor: ProtectedOutcomeExecutor,
    *,
    max_gpu_hours: float,
    calibration: Mapping[str, Any],
    progress_path: str | Path,
    git_state: GitState | None = None,
) -> dict[str, Any]:
    """Run/resume protected units only after all integrity and budget gates."""

    config_path = Path(config_path)
    config = load_config(config_path)
    capture = load_verified_capture(config_path)
    binder = getattr(executor, "bind_capture", None)
    if binder is not None:
        binder(capture)
    runtime = executor.runtime_metadata()
    plan = _read_json(Path(plan_path))
    validate_frozen_plan(plan, config_path=config_path, capture=capture)
    git = assert_protected_git_state(
        plan_path,
        git_state,
        source_git_commit=str(plan["source_git_commit"]),
    )
    _validate_calibration(calibration, capture, config)
    if str(calibration["calibration_sha256"]) != str(plan["calibration_sha256"]):
        raise RuntimeError("protected calibration differs from the one frozen in the plan")
    supported = set(executor.supported_controls())
    missing = set(REQUIRED_CONTROLS).difference(supported)
    if missing:
        raise RuntimeError(
            "BLOCKED_PROTOCOL_INTERFACE: executor lacks exact direct-control semantics for "
            f"{sorted(missing)}"
        )
    estimate = estimate_study_work(
        config,
        measured_derivative_seconds_per_episode=float(
            calibration["measured_derivative_seconds_per_episode"]
        ),
        measured_direct_seconds_per_forward=float(
            calibration["measured_direct_seconds_per_forward"]
        ),
    )
    assert_budget(estimate, max_gpu_hours=max_gpu_hours)
    progress_file = Path(progress_path)
    fingerprint = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "phase": "protected_eval",
        "plan_sha256": plan["plan_sha256"],
        "config_file_sha256": capture.identity["config_file_sha256"],
        "capture_content_sha256": capture.identity["capture_content_sha256"],
    }
    progress = load_progress(progress_file, fingerprint)
    splits = dict(progress.get("splits", {}))

    def checkpoint(split: str, value: Mapping[str, Any]) -> None:
        splits[split] = value
        write_progress(progress_file, fingerprint, {"splits": splits})

    for split in PROTECTED_SPLITS:
        # Never trust the aggregate progress blob as protected evidence.  Even
        # a completed split is replayed through the executor, which reloads and
        # checksum-verifies every immutable direct-outcome unit before the
        # aggregate is reconstructed.
        result = executor.execute_split(
            split=split,
            plan=plan,
            progress=splits.get(split, {}),
            progress_callback=lambda value, split=split: checkpoint(split, value),
        )
        checkpoint(split, {**result, "complete": True})
    outcomes = {
        split: splits[split]["outcomes"] for split in PROTECTED_SPLITS
    }
    outcomes["task_eligibility"] = _capture_task_eligibility(capture.metrics)
    result = aggregate_protected_outcomes(config, plan, outcomes)
    progress_file.unlink(missing_ok=True)
    return {
        **result,
        "source_git_commit": str(plan["source_git_commit"]),
        "execution_git_commit": git.commit,
        "calibration_sha256": str(calibration["calibration_sha256"]),
        "capture_identity": dict(capture.identity),
        "execution_runtime": dict(runtime),
        "work_estimate": asdict(estimate),
    }


def load_progress(path: str | Path, fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Load only progress bound to the exact plan/config/capture identity."""

    path = Path(path)
    if not path.exists():
        return {}
    payload = _read_json(path)
    if payload.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise RuntimeError("unknown binding-study progress schema")
    if payload.get("fingerprint") != dict(fingerprint):
        raise RuntimeError("stale binding-study progress fingerprint")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("binding-study progress state is malformed")
    return dict(state)


def write_progress(
    path: str | Path, fingerprint: Mapping[str, Any], state: Mapping[str, Any]
) -> None:
    """Atomically checkpoint progress after each independently replayable unit."""

    _atomic_json(
        Path(path),
        {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "fingerprint": dict(fingerprint),
            "state": dict(state),
        },
    )


def save_derivative_unit(
    path: str | Path,
    fingerprint: Mapping[str, Any],
    episode_id: str,
    payload: Mapping[str, Any],
) -> str:
    """Atomically persist one lossless derivative unit and return its SHA-256."""

    path = Path(path)
    arrays = _normalized_derivative_payload(payload)
    metadata = {
        "fingerprint": dict(fingerprint),
        "episode_id": str(episode_id),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
    temporary.replace(path)
    return sha256_file(path)


def save_direct_outcome_unit(
    path: str | Path,
    identity: Mapping[str, str],
    outcome: DirectMediationOutcome,
) -> str:
    """Persist one directly executed protected outcome with a self-checksum."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "identity": dict(identity),
        "outcome": asdict(outcome),
    }
    payload["payload_sha256"] = sha256_json(payload)
    _atomic_json(Path(path), payload)
    return str(payload["payload_sha256"])


def load_direct_outcome_unit(
    path: str | Path, identity: Mapping[str, str]
) -> DirectMediationOutcome:
    """Resume a protected unit only when checksum and identity are exact."""

    payload = _read_json(Path(path))
    expected = str(payload.get("payload_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    if len(expected) != 64 or sha256_json(unsigned) != expected:
        raise RuntimeError(f"protected direct-outcome checksum mismatch: {path}")
    if payload.get("schema_version") != 1 or payload.get("identity") != dict(identity):
        raise RuntimeError(f"stale protected direct-outcome identity: {path}")
    return _outcomes([payload.get("outcome")], str(path))[0]


def load_derivative_unit(
    path: str | Path,
    fingerprint: Mapping[str, Any],
    episode_id: str,
    expected_sha256: str,
) -> dict[str, np.ndarray]:
    """Load a derivative unit only when checksum and phase identity match."""

    path = Path(path)
    if not path.is_file() or sha256_file(path) != _full_sha(expected_sha256, "derivative SHA-256"):
        raise RuntimeError(f"derivative progress checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata"].item()))
        if metadata.get("fingerprint") != dict(fingerprint):
            raise RuntimeError(f"stale derivative progress fingerprint: {path}")
        if metadata.get("episode_id") != episode_id:
            raise RuntimeError(f"derivative progress episode mismatch: {path}")
        payload = {name: value[name].copy() for name in value.files if name != "metadata"}
    return _normalized_derivative_payload(payload)


def sha256_file(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    if Path(path).suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def capture_config_digest(config: Mapping[str, Any]) -> str:
    """Match the exact semantic-config serialization used by capture v2."""

    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _capture_metrics_path(config: Mapping[str, Any]) -> Path:
    final = Path(str(config["output_metrics"]))
    return final.with_name(final.name.replace("mediation", "capture"))


def _validate_calibration(
    calibration: Mapping[str, Any], capture: CaptureBundle, config: Mapping[str, Any]
) -> None:
    if calibration.get("status") != "CALIBRATION_COMPLETE":
        raise RuntimeError("train planning requires a complete calibration artifact")
    expected = str(calibration.get("calibration_sha256", ""))
    unsigned = dict(calibration)
    unsigned.pop("calibration_sha256", None)
    if len(expected) != 64 or sha256_json(unsigned) != expected:
        raise RuntimeError("calibration artifact self-hash mismatch")
    if calibration.get("identity") != dict(capture.identity):
        raise RuntimeError("calibration artifact was measured against another capture")
    expected_ids = [
        episode.episode_id
        for episode in binding_episodes_from_config(config)
        if episode.split == "calibration"
    ]
    if calibration.get("episode_ids") != expected_ids:
        raise RuntimeError("calibration artifact does not cover the exact calibration split")
    timings = calibration.get("timings")
    if not isinstance(timings, Mapping) or set(timings) != set(expected_ids):
        raise RuntimeError("calibration per-episode timings are incomplete")
    for episode_id in expected_ids:
        _validate_timing(timings[episode_id], episode_id)
    _optional_positive(
        float(calibration["measured_derivative_seconds_per_episode"]),
        "calibration derivative median",
    )
    _optional_positive(
        float(calibration["measured_direct_seconds_per_forward"]),
        "calibration direct median",
    )


def _validate_timing(value: Any, episode_id: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"calibration timing is malformed for {episode_id}")
    for name in ("derivative_seconds", "direct_outcome_seconds", "direct_seconds_per_forward"):
        try:
            number = float(value[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"calibration timing {name} is malformed for {episode_id}") from exc
        if not np.isfinite(number) or number <= 0.0:
            raise RuntimeError(f"calibration timing {name} is nonpositive for {episode_id}")


def _normalized_derivative_payload(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected_shapes = {
        "local_gradients": (56, None),
        "first_order_effects": (56,),
        "directional_hvp_terms": (56,),
        "graddrop_effects": (56, 56),
        "clean_candidate": (56, None),
        "clean_score": (),
    }
    if set(payload) != set(expected_shapes):
        raise RuntimeError("derivative payload schema differs from the frozen estimators")
    normalized: dict[str, np.ndarray] = {}
    hidden: int | None = None
    for name, shape in expected_shapes.items():
        array = np.asarray(payload[name], dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"derivative payload contains nonfinite {name}")
        if shape == (56, None):
            if array.ndim != 2 or array.shape[0] != 56 or array.shape[1] <= 0:
                raise RuntimeError(f"derivative payload {name} has invalid shape")
            hidden = array.shape[1] if hidden is None else hidden
            if array.shape[1] != hidden:
                raise RuntimeError("derivative hidden dimensions disagree")
        elif array.shape != shape:
            raise RuntimeError(f"derivative payload {name} has invalid shape {array.shape}")
        normalized[name] = array
    return normalized


def _stack_derivatives(payloads: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not payloads:
        raise RuntimeError("cannot stack an empty derivative collection")
    normalized = [_normalized_derivative_payload(payload) for payload in payloads]
    return {
        name: np.stack([payload[name] for payload in normalized])
        for name in normalized[0]
    }


def _assert_capture_matches_config(config_path: Path, capture: CaptureBundle) -> None:
    if capture.identity.get("config_file_sha256") != sha256_file(config_path):
        raise RuntimeError("capture bundle was loaded for different config bytes")
    if capture.identity.get("config_digest") != capture_config_digest(load_config(config_path)):
        raise RuntimeError("capture bundle semantic config digest mismatch")


def _direct_control_contracts(config: Mapping[str, Any]) -> dict[str, Any]:
    controls = config["controls"]
    if not all(bool(controls.get(name)) for name in REQUIRED_CONTROLS):
        raise ValueError("all preregistered direct controls must remain enabled")
    return {
        "donor_shuffle": {
            "semantics": str(controls["donor_shuffle_contract"]),
            "seed": int(controls["donor_shuffle_seed"]),
            "preserves_primary_clean_and_treated_endpoints": True,
        },
        "norm_matched_resample": {
            "semantics": str(controls["norm_resample_contract"]),
            "seed": int(controls["norm_resample_seed"]),
            "bins": int(controls["norm_resample_bins"]),
            "covariance_matched": False,
            "preserves_primary_clean_and_treated_endpoints": True,
        },
        "irrelevant_position": {
            "semantics": "patch_identical_mediator_donor_states_at_registered_irrelevant_position",
            "position": int(controls["irrelevant_position_index"]),
            "preserves_primary_clean_and_treated_endpoints": True,
        },
        "unqueried_value_swap": {
            "semantics": "use_mediator_states_induced_by_a_swap_excluding_the_query_binding",
            "preserves_primary_clean_and_treated_endpoints": True,
        },
    }


def _permutation_digest(rows: int, count: int, seed: int) -> str:
    if rows <= 1 or count <= 0:
        raise ValueError("answer-row permutation contract requires rows>1 and count>0")
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    for _ in range(count):
        digest.update(np.asarray(rng.permutation(rows), dtype="<i8").tobytes())
    return digest.hexdigest()


def _derangement_assignment(episode_ids: Sequence[str], *, seed: int) -> dict[str, str]:
    identifiers = tuple(str(value) for value in episode_ids)
    if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
        raise ValueError("donor shuffle requires at least two unique episode IDs")
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        permutation = rng.permutation(len(identifiers))
        if np.all(permutation != np.arange(len(identifiers))):
            return {
                source: identifiers[int(donor)]
                for source, donor in zip(identifiers, permutation, strict=True)
            }
    raise RuntimeError("could not construct the frozen donor-shuffle derangement")


def audited_unqueried_swap(adapter: Any, episode: Any) -> tuple[str, tuple[int, int]]:
    """Construct and token-audit the unique swap outside the queried treatment pair."""

    excluded = set(int(value) for value in episode.swapped_indices)
    remaining = tuple(index for index in range(4) if index not in excluded)
    if len(remaining) != 2 or episode.query_index not in excluded:
        raise RuntimeError("episode does not expose exactly two unqueried control slots")
    values = list(episode.recipient_values)
    values[remaining[0]], values[remaining[1]] = values[remaining[1]], values[remaining[0]]
    prompt = render_binding_prompt(
        episode.keys,
        values,
        episode.query_index,
        template=episode.template,
    )

    def ids(text: str) -> tuple[int, ...]:
        batch = adapter.tokenize([text])
        row = batch.input_ids[0]
        if hasattr(row, "detach"):
            row = row.detach().cpu()
        return tuple(int(value) for value in row.tolist())

    recipient = ids(episode.recipient_prompt())
    control = ids(prompt)
    primary = ids(episode.donor_prompt())
    if len(recipient) != len(control) or Counter(recipient) != Counter(control):
        raise RuntimeError("unqueried swap changed token length or token multiset")
    changed = tuple(
        index
        for index, (left, right) in enumerate(zip(recipient, control, strict=True))
        if left != right
    )
    primary_changed = {
        index
        for index, (left, right) in enumerate(zip(recipient, primary, strict=True))
        if left != right
    }
    if len(changed) != 2:
        raise RuntimeError("unqueried swap must change exactly two token positions")
    if primary_changed.intersection(changed):
        raise RuntimeError("unqueried swap overlaps the queried treatment token positions")
    return prompt, (changed[0], changed[1])


def compute_answer_row_permutation_diagnostic(
    deltas: np.ndarray,
    local_gradients: np.ndarray,
    *,
    count: int,
    seed: int,
) -> dict[str, Any]:
    """Test episode-local delta/score-gradient alignment on train only.

    This is explicitly not a null for the population-mean gradient, whose mean
    is invariant to row permutation.  The endpoint is the maximum node score
    under exact local AtP, recomputed after permuting complete gradient rows.
    """

    delta = np.asarray(deltas, dtype=np.float64)
    gradients = np.asarray(local_gradients, dtype=np.float64)
    if delta.shape != gradients.shape or delta.ndim != 3 or delta.shape[0] <= 1:
        raise ValueError("permutation diagnostic requires aligned [episodes,nodes,hidden] arrays")
    if count <= 0:
        raise ValueError("answer-row permutation count must be positive")
    observed = float(np.max(exact_local_atp_scores(delta, gradients)))
    rng = np.random.default_rng(seed)
    null = tuple(
        float(np.max(exact_local_atp_scores(delta, gradients[rng.permutation(delta.shape[0])])))
        for _ in range(count)
    )
    p_value = float((1 + np.count_nonzero(np.asarray(null) >= observed)) / (count + 1))
    return {
        "endpoint": "maximum_exact_local_atp_node_score",
        "scope": "train_only_episode_local_alignment_not_population_mean_gradient",
        "observed": observed,
        "null_values": list(null),
        "upper_tail_monte_carlo_p": p_value,
    }


def _validate_permutation_diagnostic(
    value: Mapping[str, Any], *, expected_count: int
) -> dict[str, Any]:
    if value.get("endpoint") != "maximum_exact_local_atp_node_score":
        raise ValueError("answer-row permutation endpoint differs from the frozen contract")
    if value.get("scope") != "train_only_episode_local_alignment_not_population_mean_gradient":
        raise ValueError("answer-row permutation scope differs from the frozen contract")
    observed = float(value.get("observed", float("nan")))
    null = np.asarray(value.get("null_values", ()), dtype=np.float64)
    p_value = float(value.get("upper_tail_monte_carlo_p", float("nan")))
    if (
        not np.isfinite(observed)
        or null.shape != (expected_count,)
        or not np.all(np.isfinite(null))
        or not np.isfinite(p_value)
        or not 0.0 <= p_value <= 1.0
    ):
        raise ValueError("answer-row permutation diagnostic is incomplete or nonfinite")
    expected_p = float((1 + np.count_nonzero(null >= observed)) / (expected_count + 1))
    if abs(p_value - expected_p) > 1e-15:
        raise ValueError("answer-row permutation p-value is inconsistent with null values")
    return {
        "endpoint": str(value["endpoint"]),
        "scope": str(value["scope"]),
        "observed": observed,
        "null_values": [float(item) for item in null],
        "upper_tail_monte_carlo_p": p_value,
    }


def _task_eligibility(
    config: Mapping[str, Any], outcomes_by_split: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    gates = config["gates"]
    result: dict[str, bool] = {}
    # Task competence is a property of the immutable capture, not of mediator
    # outcomes.  Callers must carry the capture gate for every registered split.
    supplied = outcomes_by_split.get("task_eligibility")
    if not isinstance(supplied, Mapping):
        raise RuntimeError("protected outcomes must include capture-derived task_eligibility")
    for split in ("train", "validation", "test", "paraphrase"):
        value = supplied.get(split)
        if not isinstance(value, bool):
            raise RuntimeError(f"task eligibility missing boolean split {split}")
        result[split] = value
    _ = gates  # gates were applied by capture_eligibility_gates; do not recompute loosely.
    return result


def _capture_task_eligibility(metrics: Mapping[str, Any]) -> dict[str, bool]:
    gates = metrics.get("gates")
    if not isinstance(gates, Mapping):
        raise RuntimeError("capture metrics do not contain task eligibility gates")
    result: dict[str, bool] = {}
    for split in ("train", "validation", "test", "paraphrase"):
        competent = gates.get(f"{split}_competent")
        groups = gates.get(f"{split}_groups_competent")
        if not isinstance(competent, bool) or not isinstance(groups, bool):
            raise RuntimeError(f"capture eligibility gates are incomplete for {split}")
        result[split] = competent and groups
    return result


def _outcomes(value: Any, name: str) -> tuple[DirectMediationOutcome, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise RuntimeError(f"{name} must be a nonempty outcome sequence")
    converted: list[DirectMediationOutcome] = []
    for row in value:
        if isinstance(row, DirectMediationOutcome):
            converted.append(row)
        elif isinstance(row, Mapping):
            converted.append(DirectMediationOutcome(**row))
        else:
            raise RuntimeError(f"{name} contains a malformed outcome")
    return tuple(converted)


def _donor_ids(payload: Mapping[str, Any], expected: int, split: str) -> tuple[int, ...]:
    values = payload.get("donor_answer_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(f"{split} donor_answer_ids are missing")
    result = tuple(int(value) for value in values)
    if len(result) != expected:
        raise RuntimeError(f"{split} donor_answer_ids differ from outcome count")
    return result


def _full_sha(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 40 and len(text) != 64:
        raise ValueError(f"{name} must be a full SHA-1 or SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{name} is not hexadecimal")
    return text.lower()


def _optional_positive(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
