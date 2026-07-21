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

`WM-ACTION-PATH-CALIBRATION-002` was launched from clean `288f663` after two documented OOM attempts
that emitted no metrics. The retained implementation streams batches of 64 exact action Jacobians,
immediately contracts them with the action chord and registered decoder, detaches the three decoded
coordinates to CPU, and releases each graph. The orchestration wrapper stopped waiting after its
20-minute default, but the child process continued; monitor the configured JSON path and original
PID before relaunching. The start-time provenance object already records commit `288f663` and
`git_dirty: false`, so later documentation edits cannot change the code/config identity of that run.
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
implemented after the active `288f663` process imported its code, so it cannot retroactively provide
partial recovery for that run.
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

Updated `AUDIT-COMPLETE-001` ran from clean synchronized commit `98a9e62`; all 14 criteria and all
software checks passed, including 68 tests and the corrected exact-JVP disposition. Its own
metrics/provenance pair is committed, so later runs can audit the audit.
