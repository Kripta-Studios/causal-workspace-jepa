# SUMMARY

## 2026-07-20

### Done

- Read `AGENTS.md` and `VPS_RUNBOOK.md` fully.
- Confirmed current machine is the CPU VPS path: no GPU, 4 CPU cores, about 7 GB free at start.
- Added package metadata, resource profiles, package tree, resource guard, doctor CLI, typed interfaces, provenance helpers, and docs skeleton.
- Added standard-library unit/integration tests for config parsing, resource guards, CLI output, intervention serialization, and reproducibility scaffolding.
- Validated Milestone 0 with `doctor`, `python -m unittest discover -s tests -p 'test_*.py'`, `scripts/audit_reproducibility.py`, and `git diff --check`.
- Implemented deterministic Tier 0 generators: PointMass2D, BouncingBall2D, TwoBodyCollision, TinyMaze, and MiniPush.
- Implemented a tiny NumPy action-conditioned JEPA with fixed encoder, ridge-fitted predictor, save/load, named activation points, and no-action/shuffled-action controls.
- Implemented a random-shooting planner and PointMass closed-loop cost check.
- Added Milestone 1 tests. At that milestone the full command passed 14 tests; before the current
  `WM-T0-004` code milestone the full suite had 34 tests.
- Committed and pushed Milestone 1 code as `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- Reran Tier 0 generation and `WM-T0-001` tiny JEPA smoke from clean committed code.
- Committed and pushed result summaries as `b0b7796`: `data/manifests/tier0_smoke_manifest.json`, `artifacts/metrics/tiny_jepa_smoke.json`, and `artifacts/metrics/tiny_jepa_smoke.provenance.json`.

### Discovered

- `README.md` initially only contained the project title.
- `pytest` is not installed, but NumPy is available system-wide.
- Historical decision: GPT-2 Medium was initially treated as `BLOCKED_RESOURCE`. A later explicit
  user override and the final `AGENTS.md` CPU guard permitted the cached model under strict limits.
- A first dirty-run tiny JEPA smoke pass succeeded; generated summaries were removed before the code commit so the final reported metrics can be rerun from committed code.
- Final smoke provenance reports `git_dirty: false` and code commit `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- `WM-T0-001` beat the mean, no-action, and shuffled-action baselines on latent prediction, and the planner beat average random rollout cost. This is evidence level Availability only.
- Implemented intervention operators: zero, mean, resample, patch, replace_feature, steer, project_out, scale, and suppress_module.
- Implemented activation cache, stable site naming, finite-difference Jacobian, normalized patch recovery, ridge probes, random-label control, sparse dictionary, circuit graph JSON/GraphML, and mock transformer adapter.
- Implemented mock intervention-JEPA smoke runner and tests. Exploratory dirty-run metrics passed, then were removed before the code commit.
- Committed and pushed Milestone 2 code as `85c1dbfbe9c824bcca415af13f4a6f34acc95267`.
- Reran `LLM-MOCK-001` from clean committed code. Provenance reports `git_dirty: false`.
- Committed and pushed mock result summaries as `948347c`: `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json` and `artifacts/metrics/mock_qwen_intervention_jepa_smoke.provenance.json`.
- Added `scripts/bootstrap_cpu.sh`, config `.gitkeep` files, and reporting stubs to complete the requested architecture.
- `uv` is not installed on the VPS. The bootstrap script checks resources and reports the exact install command instead of auto-installing tools.
- Historical audit reported `/root/.cache/pip` at about 4.4 GB and left it untouched. The later
  resource re-audit measured all of `/root/.cache` at about 233 MB, so that old size is no longer
  current.
- Historical `WM-T0-002` run reported action recovery `0.984`; adversarial review later found that
  result tautological and superseded it with the clean replay result documented below.
- Under explicit user override, created `.venv`, installed `transformers`/`safetensors`, downloaded `gpt2-medium`, and ran `LLM-GPT2-001`.
- GPT-2 Medium result: direct residual steering at `transformer.h.12.resid_post` changed logits with mean absolute delta `0.0797`; tiny intervention-JEPA MSE `0.00220` beat no-change `0.0114`.
- Historical provenance: the superseded `WM-T0-002` ran on `e5db938`; `LLM-GPT2-001` remains valid
  on `59795a4`.
- Committed and pushed GPT-2 metrics/docs as `a7ea6da`.

### Next Ideas

- On `gpu_12gb`, test GPT-2/Qwen semantic, composed, resampling, and larger interventions while
  retaining prompt-local Jacobians as the baseline to beat.

### Adversarial Milestone 3 Re-audit

- Re-read `AGENTS.md` and `VPS_RUNBOOK.md` completely and reran the baseline suite: 25 tests passed;
  doctor, reproducibility audit, and bytecode compilation passed.
- Found a material flaw in `WM-T0-002`: `action_patch` was assigned to the donor target rather than
  produced by an internal intervention. The `0.984` recovery is preserved but no longer interpreted
  as Specificity evidence.
- Implemented actual replay through `InterventionSpec` at `predictor.input`, nonzero donor sampling,
  and L2 norm-matched latent/action controls.
- Implemented a normalized multi-consumer Jacobian detector, compactness criterion, direct projection
  ablations, random and PCA controls, and known shared/disjoint detector controls.
- Added a preregistered `WM-T0-003` five-consumer study. It can identify a shared causal state
  subspace candidate, but it cannot declare a full workspace because key functional criteria remain
  untested.
- Reran corrected `WM-T0-002` from clean commit `315d8cf`. Actual `InterventionSpec` replay recovered
  the donor effect exactly (`1.0`, max error `0.0`) while L2 norm-matched controls did not. This is
  narrow Specificity evidence for the explicit action-input pathway, not a learned workspace.
- Ran `WM-T0-003` from clean commit `5223a54`. The detector passed planted shared/disjoint controls
  and proposed a `3/16` sensitivity subspace, but the preregistered candidate decision failed:
  uncertainty R2 was `-3.639`, PCA damage exceeded candidate damage, and random rollout controls were
  badly off-manifold. No workspace was found.
- Best next JEPA experiment: a deeper learned predictor with calibrated ensemble/heteroscedastic
  uncertainty and in-manifold donor or activation-density-matched controls. Do not tune the current
  failed proxy after observing its result; register a new experiment.

### Current Handoff

- Worktree was clean at pushed commit `93431c3` before starting the next code milestone.
- Audited every repository Markdown file and all committed provenance logs. Corrected stale
  statements during the GPT-2 code milestone: the original action patch is superseded,
  `WM-T0-003` is no longer pending, and cached GPT-2 is allowed while Qwen remains blocked.
- Implemented `LLM-GPT2-002`: 288 batched direct interventions, local-only model loading, storage
  budget/checksum manifest, prompt/magnitude/layer holdouts, linear and bilinear predictors, trained
  MLP, nearest-neighbor and sparse-context baselines, and prompt-local/corpus Jacobians.
- Preregistered all splits and thresholds before execution. Do not change them after seeing results.
- At the preregistration checkpoint, 34 tests passed before code commit/push and clean execution.
  The measured result is recorded below.

### LLM-GPT2-002 Result

- Ran from clean commit `8fbab8c`; provenance reports `git_dirty: false` and no model download.
- Generated 288 direct residual-intervention outcomes and an ignored 87,378-byte float16 shard.
  Committed manifest checksum:
  `06a3a75c91076422d73fd62a85694c8366c4db854b2c303e203256adffe73abf`.
- Primary unseen-prompt/magnitude result: local Jacobian MSE `7.79e-7`, bilinear
  Intervention-JEPA `0.003499`, linear `0.006360`, no-change `0.006461`, MLP `0.007361`.
- H-LLM-01 nonlinear advantage failed. The narrow H-LLM-02 compression test passed because
  bilinear beat linear/no-change, but local direct linearization was about 4,490 times lower MSE.
- Held-out layer 18: bilinear MSE `0.01009` was worse than no-change `0.006228`; cross-layer transfer
  failed. Local Jacobian remained nearly exact (`3.71e-8`).
- No intervention changed the top token. This is hidden/logit causal mediation only, not behavior,
  semantic feature, J-space, workspace, Qwen, or consciousness evidence.
- Runtime was `647.85` seconds, exceeding the 10-minute CPU expectation by `47.85` seconds; record
  this as a resource-limit miss.
- Metrics, clean provenance, checksum manifest, and synchronized result docs were committed and
  pushed as `fdf6506`.
- Best next LLM experiment: on `gpu_12gb`, use semantic directions, activation patch/resampling,
  combined interventions, larger magnitudes, and enough prompts/seeds to expose genuine nonlinearity.
  Keep prompt-local Jacobians as the baseline to beat.
- Strengthened `scripts/audit_reproducibility.py`: it now validates every metrics/provenance pair,
  clean commit metadata, JSON structure, and all available local dataset checksums instead of only
  checking that seven control-plane paths exist.
- The strengthened reproducibility audit and tests were committed and pushed as `3eedcb5`.
- Final handoff target: full suite, doctor, checksum/provenance audit, and `git diff --check` must
  pass; branch `main` must be clean and synchronized with `origin/main`.

### WM-T0-004 Implementation In Progress

- Re-audited every Markdown handoff/registry and the committed metrics/provenance logs. Corrected the
  stale test count and the stale instruction to run the already-completed `LLM-GPT2-002`.
- Primary-source verification confirms that Anthropic's J-space is a sparse nonnegative token-aligned
  frame tested for report, directed modulation, internal reasoning, flexible reuse, and selectivity.
  A JEPA Jacobian eigensubspace is therefore only an analogue candidate, never equivalent by name.
- Added a NumPy-only two-hidden-layer action-conditioned JEPA whose hidden weights are learned and
  exposed at `predictor.hidden1` and `predictor.hidden2`; its fixed encoder prevents JEPA collapse.
- Added a five-member bootstrap ensemble, split interval calibration, held-out coverage/NLL, OOD rank
  AUC, and uncertainty/error correlation. No uncertainty claim is accepted unless all registered
  thresholds pass.
- Added conditional donor resampling: candidate coordinates come from validation activations matched
  in the orthogonal complement. Random/PCA controls must match candidate density distance and
  perturbation magnitude before they count.
- Preregistered `WM-T0-004` before execution. It fits all five consumers after freezing the predictor,
  discovers candidates separately at two depths, and requires direct plus multistep specificity.
- Resource incident: free disk fell to 3.8 GB because broken external `AppTurnos` launchers produced
  about 37 GB of PM2 restart-loop logs. Per user instruction, the systemd service is now inactive and
  masked, its PM2 entry and saved startup record are removed, port 3001 is closed, and 40+ GB is free.
  The source/database remain as a recoverable archive outside this repository.
- Current boundary: all 37 tests, doctor, reproducibility audit, compile check, and `git diff --check`
  pass with 40+ GB free. Do not execute `WM-T0-004` until this implementation is committed and
  pushed; do not change its registered thresholds after observing results.
