# Reproducibility

Status: `SMOKE_VALIDATED` for the complete bounded suite and cross-platform control-plane checks.

Every experiment must record:

- config path and serialized config;
- data manifest and checksums;
- code commit and dirty flag;
- hardware/resource profile;
- seed;
- command;
- logs;
- metrics with `evidence_level`;
- failure status when applicable.

CPU path:

```bash
export PYTHONPATH=src
python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/cpu_vps.yaml
python -m unittest discover -s tests -p 'test_*.py'
```

Windows GPU-host control path:

```powershell
$env:PYTHONPATH = "src"
python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/gpu_12gb.yaml
python -m unittest discover -s tests -p "test_*.py"
python scripts/audit_reproducibility.py
python scripts/audit_completion.py
```

Validated smoke commands:

```bash
PYTHONPATH=src python scripts/generate_tier0.py --config configs/data/tier0_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/tiny_jepa_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/mock_qwen_intervention_jepa_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/tier0_mechanistic_study.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/workspace_discovery_study.yaml
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --config configs/experiments/multitask_workspace_study.yaml
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --config configs/experiments/gpt2_medium_mechanistic_study.yaml
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --config configs/experiments/gpt2_medium_semantic_composition_study.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/qwen3_0_6b_instrumentation_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/qwen_country_code_layer_geometry_v1.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/lewm_action_path_calibration_v1.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/lewm_action_path_calibration_v2.yaml
PYTHONPATH=src python scripts/generate_qwen_interventions.py --config configs/experiments/qwen_intervention_dataset_v1.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/intervention_jepa_v1.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/lewm_small_reproduction_v1.yaml
PYTHONPATH=src python scripts/prepare_eb_jepa.py --target .cache/upstream/eb_jepa
PYTHONPATH=src python scripts/prepare_eb_jepa_torch_runtimes.py --mode both
PYTHONPATH=src python scripts/doctor_eb_jepa.py
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/eb_jepa_official_contract_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/eb_jepa_runtime_compatibility.yaml
PYTHONPATH=src python scripts/capture_qwen_activations.py --config configs/llm/qwen3_4b_selected_layers.yaml
PYTHONPATH=src python scripts/audit_completion.py
```

GPT-2 Medium uses `local_files_only: true`; the command must fail instead of downloading a missing
model. Its generated float16 shard is ignored, while the checksum manifest and summarized metrics
are committed.

No reported result may depend on uncommitted code.

`LLM-QWEN-001` pins both model revision and clean code commit. The initial execution was rejected
because provenance was collected after its untracked output was created; only the corrected clean
rerun from `0d6a37b` is retained.

`LLM-IJEPA-001` consumed the immutable checksum-verified `LLM-INTDATA-001` shard and ran from clean
commit `a54f2ed6a2491fb905978cb3c10af655a36c7b42`. Its three seeds, thresholds, holdouts, and direct
verification prompts were fixed before execution. Ignored checkpoints are replay-checked by the run.

`LLM-QWEN-JVP-AUDIT-001` was preregistered at `236e5d9` and executed from clean commit
`686368e792598aaeb3d0aff7349d34f8f70a3c36`. It wrote complete metrics/provenance and a checksummed
ignored HDF5 shard before intentionally exiting nonzero because one numerical gate failed. The
result is retained as `REJECTED` on a numerical gate; it does not decide H-LLM-01.

Post-diagnostic v2 was preregistered and executed from clean synchronized commit
`a779ff6ea617f77e2b0c252c79b5a1a1fa66cfdc`. Its 432-row shard checksum verifies, every numerical
gate passes, and the emitted disposition is `WITHDRAWN`. The source-level semantic repair and its
post-v1 status are explicit in `docs/HYPOTHESES.md`.

`WM-LEWM-001` ran unchanged from clean commit `4dbc38856b2f1aa6e42754ade72941f0399d3b93`.
The prior clean computation at `9c3239a` was discarded because a hardware dataclass prevented JSON
serialization before metrics/provenance were written. Only the post-fix run is retained. The runner
intentionally exits nonzero after writing a complete `FAILED_REGISTERED_GATES` artifact; this is a
valid negative scientific outcome, not a missing run.

`scripts/audit_reproducibility.py` checks required control-plane files, every summarized metrics and
provenance pair, `git_dirty: false`, recorded commit/path fields, JSON validity, and every available
local dataset checksum. Missing ignored datasets are reported as skipped so a fresh clone can audit
metadata before regeneration; checksum mismatches fail the audit.

Recorded relative paths are normalized to POSIX separators before comparison, allowing provenance
created on Linux to be audited on Windows without weakening the target-path check.

`LLM-COUNTRY-CODE-LAYER-GEOMETRY-001` was preregistered at `ab07627`. Its first clean launch
completed model computation but emitted no artifact because the frozen element metrics predated a
cached eligibility field. The compatibility-only fallback was tested/pushed at `48226c6`; the
unchanged clean retry wrote metrics/provenance and a 30,887,576-byte ignored shard with SHA-256
`13f3792ab221f2a795d9529010cfef4da6e622cf21ecc70e08e46e0570224235`.

`WM-ACTION-PATH-CALIBRATION-001` ran from clean `eb943a5` and records
`protected_test_goals_touched=false`, empty hypothesis decisions, the exact 24 stratified chords per
seed/horizon, and both 128/256-node estimates. Its numerical underresolution is retained; later
calibration must use a new ID rather than overwrite this artifact.

`WM-ACTION-PATH-CALIBRATION-002` completed from clean `288f663` after two documented OOM attempts
that emitted no metrics. The retained implementation streams batches of 64 exact action Jacobians,
immediately contracts them with the action chord and registered decoder, detaches the three decoded
coordinates to CPU, and releases each graph. The orchestration wrapper stopped waiting after its
20-minute default, but the child process continued and completed after `19,176.20` seconds. The
retained provenance records commit `288f663`, `git_dirty: false`, seed 647, and the configured metric
path. The artifact records `protected_test_goals_touched=false` and empty decisions.
Before its result was available, adversarial review identified the shared decoded-direct-effect
denominator in cancellation and normalized local error. This is a scientific-design issue rather
than an implementation change. A proposed derived audit was rejected before commit because v2
does not store scalar path length at both resolutions or the unclamped direct norm, and two chords
per action pair cannot support the required joint conditional null. V2 remains numerical/vector
calibration, the family is closed, and protected test stays locked.

Future action-path launches write an ignored sibling `*.progress.json` after every completed
seed/horizon. The write uses a temporary sibling followed by atomic replacement. Resume is permitted
only when the stored experiment ID and SHA-256 fingerprint of the exact YAML bytes plus Git commit
match; duplicate or unexpected seeds fail closed. Final metrics/provenance record whether resume was
used and the number of loaded horizon blocks, then remove the progress file. This mechanism was
implemented after the `288f663` process imported its code, so it did not provide resume state for
that completed run.

Official EB-JEPA source is fetched only into ignored `.cache/upstream/eb_jepa` and validated at
commit `966e61e9285b3a876f49b9774e9720d9a99a7925`. The contract smoke imports the real official
Impala encoder and one-layer GRU, checks native-versus-decomposed recurrence at `1e-6`, and records
that it uses random weights. It is distinct from exact training reproduction. Upstream pins Python
3.12 and Torch 2.6; the present Python 3.14/Torch 2.10 runtime is explicitly
`BLOCKED_OFFICIAL_ENV` for that stronger claim even though the contract works.
The retained smoke ran from clean `979c2d692383a471f3f46de15168f475629020e3`; its provenance
records `git_dirty=false`, exact source revision, reconstruction error `4.768e-7`, and the explicit
random-weight/no-planning limitation.

The exact GPU dependency boundary is tested in two ignored Python 3.12.13 runtimes created by
`scripts/prepare_eb_jepa_torch_runtimes.py`. Torch 2.6.0+cu126 is the upstream pin; Torch
2.10.0+cu128 is the declared local SM120 compatibility deviation. The configured runtime diagnostic
requires the former to omit `sm_120` and fail matmul/Conv2D/GRU with the observed missing-kernel
error, while the latter must include `sm_120` and pass all three with finite outputs. Neither
environment is a committed binary artifact; the preparation commands and diagnostic metrics are.
The retained `WM-EBJEPA-RUNTIME-001` run started from clean
`15d88ce15f7ecec3dc924800471937c48b0c4629`; provenance records `git_dirty=false`, and all eight
frozen identity/architecture/kernel gates pass. Exact and compatible subprocess probes took
`1.51` and `1.75` seconds respectively. The root orchestration runtime remains Python 3.14/Torch
2.10; each isolated probe records its own Python, Torch, CUDA, device, and compiled architecture list.

The SM120-compatible Two Rooms closure is frozen in
`configs/resource/eb_jepa_two_rooms_py312_sm120.lock.txt` and installed with
`scripts/prepare_eb_jepa_two_rooms.py`. Torch is intentionally absent from that lock: the script
first verifies the existing `2.10.0+cu128` wheel and `sm_120`, installs exact non-Torch packages,
installs the pinned official checkout editable with `--no-deps`, and fails if Torch changes. Do not
use upstream `uv sync` on this host. The official path imports `scipy`, `pandas`, and `yaml`, while
its `pyproject.toml` declares neither scipy, pandas, nor PyYAML; `ruamel.yaml` is not an import-level
substitute for PyYAML.

`WM-EBJEPA-INTEGRATION-001` froze a tiny generated dataset, exact model dimensions, one BF16
forward/backward/AdamW step, exact checkpoint restoration, an integrated random-weight MPPI call,
and peak GPU memory. Its clean `f0e7a3e` artifact is preserved but superseded because the probe did
not seed Python's `random` or require an independent replay. The corrected
`WM-EBJEPA-INTEGRATION-002` additionally sets Python/NumPy/Torch seeds, deterministic algorithms,
CUDNN determinism, and the CUBLAS workspace contract. Two fresh subprocesses must exactly match a
SHA-256 fingerprint over batch bytes, post-update model state, loss, parameter delta, action, and
planner losses. V2 writes a new path and never overwrites v1.
The retained v2 run started from clean `9a1800818404fde5841c74eef1dae5d0cab38232` and both
subprocesses produced fingerprint
`16650872debe76818197183888877d90e76ed27408f67c569cf3d9df478234a1`; all 12 gates pass.

`WM-EBJEPA-PLANNER-CONSTRAINT-001` is separately registered as post-discovery:
it compares official CEM and MPPI over seeds 0--31 under an identical deterministic objective. No
retained result may be described until both run from a clean pushed implementation commit. The
retained run started from clean `da30443011f25ccbc689bc0595063720aadbb6d6`; all seven gates pass
in `3.04` seconds. The artifact stores all 64 per-seed planner maxima and source-contract booleans,
not only the aggregate violation counts.

The official MPPI source remains untouched. `ConstrainedMPPIPlanner` is a repository-owned,
separately named reproduction arm. `WM-EBJEPA-MPPI-CORRECTION-001` first disables the bound and
requires action/loss equivalence to official MPPI across 32 independent seeds; it then enables the
bound and instruments every action batch reaching the cost function as well as the returned action.
This prevents a superficial return-only clamp from claiming a faithful corrected optimization.
The retained run started from clean `f58308aac5815342ca12d0565cf98d8eef5fd99c` and completed in
`3.16` seconds. All five gates pass: the unbounded action/loss differences are exactly zero, and
the bounded cost-input/returned-action violation counts are both zero across 32 seeds. The original
official planner remains available as a separately labeled baseline.

`WM-EBJEPA-TRAIN-RESOURCE-001` runs every candidate batch in a fresh Python process so CUDA peaks,
OOMs, and compiler state cannot leak between measurements. It profiles two complete JEPA-plus-probe
updates for eager batches 64/128/256/384 and records Dynamo frames/graphs for the official
`torch.compile(jepa)` followed by `jepa.unroll(...)` path at batch 64. A zero-graph wrapper is not
reported as successful compilation. Raw generated batches are not persisted.
The retained clean `fed920e` run passed all five gates in `93.08` seconds. Batch 384 peaked at
`5,821,693,952` reserved bytes and is retained for training. The compile arm returned an
`OptimizedModule` but zero Dynamo frames/graphs; later source-faithful configs may leave the flag
enabled, while reports must state that the observed `unroll` path remained eager.

`WM-EBJEPA-PLANNER-CONFIG-001` additionally freezes the proposal-scale contract. The official MPPI
YAML contains `var_scale=1.5`, but the constructor expects `max_std` and executes at default `2.0`;
CEM consumes `1.5`. Reproductions must record instantiated planner attributes, not only copied YAML,
and the bound-only correction retains `2.0` to isolate one change.

The long Two Rooms portfolio is `WM-EBJEPA-TRAIN-001`. Run it through the isolated Python 3.12
interpreter; its driver validates the complete frozen upstream config before importing training,
requires a clean repository/source checkout, resumes only through upstream checkpoint loading, and
writes atomic ignored status files. It hashes all 13 checkpoint files per seed. The tracked config
freezes seeds 1/1000/10000 and epochs 9/10/11 for later evaluation; generated data/checkpoints/logs
must not be committed.

Competence evaluation uses a repository-owned no-render loop because upstream `main_eval` calls
`save_gif` unconditionally but only defines `save_path` inside the optional-plots branch. The loop
preserves the discarded initial environment reset, deterministic RNG-0 episode sequence, zero step,
200-step receding-horizon interaction, representation objective, and instantiated planner classes.
It writes one ignored job JSON per seed/checkpoint/arm and only then a compact tracked aggregate.

Qwen ordered intervention programs freeze the caller-supplied sequence. Hooks execute in model
order, while repeated specifications at one site execute in list order. Offline tests require an
upstream layer-0 token treatment to replay the donor and a later residual restoration to replay the
clean recipient within `1e-6`. Multi-site scientific runs must log the complete ordered list; a set
of sites is insufficient provenance.

The superseded Qwen binding-mediation v1 preregistration froze all token pools, seeds, templates, split counts,
candidate modules, ranking baselines, matched controls, and decision gates in
`configs/experiments/qwen_binding_mediation_v1.yaml`. Its clean token audit is retained, but the
paraphrase rows were not paired to the test episodes and no model outcomes were opened. The
operative v2 registration is `configs/experiments/qwen_binding_mediation_v2.yaml`. Before model
execution, run from a clean
commit:

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_qwen_binding_tokenization.py --config configs/experiments/qwen_binding_mediation_v2.yaml
```

The committed audit may establish only deterministic token identity: equal recipient/donor token
multisets, exactly two changed token positions, one-token answers, bounded sequence length, and
balanced query positions. It cannot decide task competence, mediation, circuit, JEPA, or workspace
hypotheses.

V2 additionally requires single-token/disjoint key and value IDs, exact test/paraphrase factor
pairing, and exact resolved revision. Capture recomputes the token digest and binds every progress
unit to config, commit, model revision, token audit, and runtime fingerprint. Candidate/final causal
states are FP32. Every numeric array must be finite; the exact canonical episode roster/order,
intervention record, split counts, shapes, and dtypes are enforced. Full context IDs and the
single-token/disjoint pool roster are separately hashed. Final HDF5 shards are read back and compared to an ordered
array-plus-record content hash. Protected capture remains prohibited until its complete downstream
evaluator and controls are committed, pushed, and independently tested.

The audit executed from clean `4e6624f7561e2646fa9beb65297a5b953f0ac237`; all four gates pass and
the deterministic episode hash is
`3ac7a80d1ebeefd9e208b5c3d12fdaa8fb611cc3ad789c7f0e448da244ebaf59`. Protected Qwen outcomes
remain unopened.

Seed completion additionally requires an explicit marker, both horizon blocks, and the derived
horizon-amplification summary; the presence of horizon 4 alone cannot skip final aggregation.

The working paper is validated separately from scientific experiments:

```powershell
cd papers
latexmk -pdf -interaction=nonstopmode -halt-on-error causal_workspace_jepa.tex
```

Run the command inside `papers/` so BibTeX resolves the tracked local `references.bib`; the MiKTeX
installation on the RTX host contains an unrelated system bibliography with the same basename.
Generated PDF and auxiliary files are ignored. Before a paper milestone, require zero undefined
citations/references and no overfull boxes, then compare every table against its committed JSON.

Updated `AUDIT-COMPLETE-001` ran from clean synchronized commit `3593475`; all 14 criteria and all
software checks passed, including 105 tests, 28 metric/provenance pairs, 10 locally verified
checksums, and the corrected exact-JVP disposition. Its own
metrics/provenance pair is committed, so later runs can audit the audit.
