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
- This led to the separately preregistered `WM-T0-004` deeper predictor, calibrated ensemble, and
  in-manifold donor study recorded below.

### Current Handoff

- Worktree was clean at pushed commit `93431c3` before starting the next code milestone.
- After the `WM-T0-004` null, implemented and preregistered `WM-T0-005`: four goals crossed with
  two mass modes, one held-out composition, three seeds, a three-model calibrated ensemble, five
  frozen consumers, random and local-tangent controls, direct task-context counterfactual swaps,
  and multistep selective-necessity tests.
- The joint candidate gate requires valid action dependence, calibrated OOD uncertainty, all five
  held-out consumer heads, compact shared sensitivity, at least 50 percent donor recovery,
  specificity over both matched control families, and multistep selectivity on two of three seeds.
  The full workspace decision remains false by construction because reportability and published
  model replication are absent.
- The preregistered implementation was committed and pushed as `7a9e510`, then executed once from
  that clean commit. Provenance reports `git_dirty: false`; runtime was `57.51` seconds and disk
  remained near 40 GB free.
- `WM-T0-005` is a replicated null: zero of three seeds passed. Action MSE ratios were `0.712`,
  `1.012`, and `1.003` versus the registered maximum `0.50`. OOD uncertainty passed jointly only on
  seed 29. Every seed failed held-out transfer for the five-consumer set.
- Candidate dimensions saturated at `6/24`, while minimum consumer sensitivity capture was only
  `0.583`, `0.561`, and `0.574` versus `0.70`. Mean goal/dynamics counterfactual recovery was
  `-10.18`, `-17.38`, and `-3.11` versus the required `+0.50`.
- Seed 37 exceeded random-control p95 for counterfactual and rollout effects despite moving away
  from the donor. The absolute recovery gate and local-tangent controls correctly rejected this
  misleading relative win. No shared task-workspace candidate and no workspace were found.
- Practical conclusion: appending task context and fitting consumers afterward is not enough. A
  new JEPA architecture should train value/risk/action consumers or a planner jointly; do not tune
  the observed `WM-T0-005` seeds or thresholds.
- Implemented and preregistered `LLM-GPT2-003` without running it: two orthogonalized residual
  contrast directions from eight disjoint calibration prompts, six evaluation prompts, magnitude
  `0.5` finite differences, magnitude `6.0` singles/compositions, and exactly 72 direct outcomes.
- Predictors train only on 32 singles from four prompts. The primary test is eight compositions from
  two unseen prompts. Required controls are prompt-local and corpus additive Jacobians plus direct
  addition of the matching large single effects. Direction labels describe construction, not proven
  semantics. The full pre-commit suite passes 43 tests; doctor, reproducibility audit, compilation,
  and `git diff --check` pass. Commit/push this preregistration before executing it once.
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

### WM-T0-004 Implementation And Result

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
- Committed and pushed the preregistered implementation as `6785fb1`, then executed once from that
  clean commit without changing thresholds. Runtime was `31.22` seconds and provenance is clean.
- The action-conditioning gate failed: primary MSE was `0.003416` versus shuffled-action `0.004994`,
  ratio `0.684` against the registered maximum `0.25`. Shuffling hurts, but not enough for the strong
  causal-use claim.
- In-distribution uncertainty calibration worked: test coverage was `0.887` for a 90 percent target,
  and uncertainty/error Spearman was `0.698`. The joint uncertainty claim failed because OOD AUC was
  only `0.574` versus `0.65` and hidden uncertainty-head R2 was `-1.327`/`0.232` versus `0.50`.
- Neither hidden site produced the registered compact sensitivity candidate: minimum five-consumer
  capture was `0.635` and `0.701`, below `0.75`.
- The old off-manifold confound is repaired. Candidate density ratios were `1.029`/`1.018`, and
  `63/64` plus `64/64` random controls matched density and magnitude. Candidate direct damage was
  below random-control p95 at both sites (`3.652 < 9.252`; `1.239 < 4.369`).
- Multistep amplification ratios (`6.03`, `1.95`) looked large in isolation but candidate rollout
  damage also remained below matched-control p95. This is nonspecific accumulation, not selective
  necessity. PCA controls were too large to be matched and correctly did not decide the claim.
- Scientific result: no shared causal subspace and no JEPA workspace. Restricted H-WM-05, H-WM-06,
  and H-WM-08 are false for this run. The reusable output is the conditional donor control method.
- ELI5 novelty boundary: this repository now has a fairer way to ask whether many JEPA outputs share
  a special internal whiteboard without smashing the model's activations into nonsense. The answer
  for this tiny model is no. This is methodologically useful and falsification-first, but it is not
  state-of-the-art performance, a published discovery, or evidence that larger JEPAs lack such a
  mechanism.
- Best next JEPA direction: preregister a goal-conditioned multi-task model where action and task
  variables are indispensable, use local-tangent PCA controls, and replicate across seeds. Do not
  tune `WM-T0-004` after seeing its null.
- Final pre-commit validation: 37 tests pass; seven metric/provenance pairs and six local checksums
  audit without errors; compilation and `git diff --check` pass; the CPU guard reports 40.1 GB free.
- Exact clean-code reproduction command:
  `PYTHONPATH=src ./.venv/bin/python scripts/run_experiment.py --config configs/experiments/manifold_workspace_study.yaml`.
