# Causal Workspace JEPA

Reproducible research codebase for action-conditioned JEPA world-model interpretability and intervention-conditioned LLM meta-modeling.

The source-of-truth working paper is
[`papers/causal_workspace_jepa.tex`](papers/causal_workspace_jepa.tex). It consolidates the common
finite-intervention formalism, preregistered controls, Qwen and world-model evidence, negative
results, novelty boundary, and reproducibility record. The strongest current conclusion is a
falsification boundary: exact/second-order/population transports rank differently by endpoint,
the registered threshold-and-four-layer-grid onset rules fail across the studied factual relations,
one EMA/stop-gradient target-encoder Intervention-JEPA variant fails its registered gates, and no
tested JEPA workspace proxy or circuit meets its acceptance rule. Decoded recurrent action-path
cancellation is closed as a scientific route after numerical and shared-denominator design
failures; the completed high-resolution run is retained as vector calibration only.

The 2026-07-21 primary-source refresh adds official EB-JEPA as the next published world-model
target and Qwen-Scope/Circuit Tracing/AtP*/HVP as required Qwen comparators. No primary source was
found that already performs the repository's exact intervention-conditioned JEPA causal-simulator
program, but the current learned JEPA result is negative; the gap is not claimed as solved or SOTA.

The official EB-JEPA source is now pinned at `966e61e...`, and its real one-layer 512-dimensional
GRU transition has a typed adapter with reset/update/candidate/hidden intervention sites. Unit tests
reconstruct native `torch.nn.GRU` within `1e-6`; the clean retained source-contract smoke passes.
Exact upstream training remains separate because the official Python 3.12/Torch
2.6/cu126 runtime cannot execute kernels on this host's SM120 GPU. A Python 3.12/Torch 2.10/cu128
compatibility runtime passes matmul, Conv2D, and GRU; that deviation is explicit and auditable.

The first Qwen binding-mediation design is now marked `SUPERSEDED_DESIGN` before any model forward.
Its tokenizer audit is retained, but it did not isolate paraphrase from episode shift. The operative
v2 preregistration pairs each paraphrase with the identical test binding, stores causal states in
FP32, rejects non-finite or malformed captures, and binds resume/readback to exact content and
runtime identity. The protected capture is deliberately not authorized until every ranking,
direct-mediation, restoration, and matched-control evaluator is frozen in committed code.

## Current Status

- `SMOKE_VALIDATED`: repository control plane, resource profiles, `doctor`, typed interfaces, standard-library tests, Tier 0 generators, tiny NumPy JEPA, random-shooting planner, save/load, and the tiny JEPA smoke experiment.
- `SMOKE_VALIDATED`: NumPy intervention operators, activation cache, ridge probes, sparse dictionary, finite-difference lenses, circuit graph I/O, mock transformer, and mock intervention-JEPA smoke runner.
- `SMOKE_VALIDATED`: adversarial Milestone 3 re-audit. The flawed original patch metric was
  superseded by a clean replayable action-input intervention with L2 norm-matched controls.
- `SMOKE_VALIDATED`: multi-consumer workspace detector with planted shared/disjoint controls. The
  tiny JEPA result is null after uncertainty, PCA, and off-manifold rollout-control failures.
- `SMOKE_VALIDATED`: `WM-T0-004` deep NumPy JEPA ensemble and conditional donor controls. The old
  off-manifold confound is repaired, but action, OOD uncertainty, consumer, and specificity gates
  fail; no shared causal subspace or workspace is accepted.
- `SMOKE_VALIDATED`: `WM-T0-005` multi-seed goal/dynamics JEPA study. Zero of three seeds passed:
  action dependence, held-out consumers, compact sensitivity, counterfactual recovery, and
  random/local-tangent specificity rejected the candidate. No workspace is accepted.
- `SMOKE_VALIDATED`: GPT-2 Medium hidden-state intervention smoke under the user's explicit override.
- `SMOKE_VALIDATED`: strengthened GPT-2 Medium study with 288 batched direct interventions. The local
  Jacobian dominates learned meta-models; bilinear compression helps on unseen prompts but fails on
  the held-out layer.
- `SMOKE_VALIDATED`: `LLM-GPT2-003` contrast-direction composition study. Effects remained almost
  additive; prompt-local Jacobians generalized to compositions, while all singles-only learned
  predictors failed on held-out prompts. All three registered hypotheses failed.
- `SMOKE_VALIDATED`: documentation/source registries, data/artifact policy, typed package tree,
  provenance helpers, and executable reproducibility checks.
- `ACTIVE`: GPU continuation on an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM, 32 logical CPU cores, and about 370 GB free at the 2026-07-21 audit.
- `SMOKE_VALIDATED`: `WM-EBJEPA-CONTRACT-001` ran from clean `979c2d6` against official commit
  `966e61e...`; native/decomposed GRU error was `4.768e-7`, and a targeted update-gate edit had a
  nonzero downstream latent effect. Random weights mean no planning, circuit, or workspace evidence.
- `BLOCKED_GPU_SM120`: isolated Python 3.12/Torch 2.6+cu126 detects the RTX 5070 Ti but omits
  `sm_120`; matmul, Conv2D, and GRU fail with no compatible kernel image.
- `SMOKE_VALIDATED`: `WM-EBJEPA-RUNTIME-001` ran from clean `15d88ce` and passed all eight frozen
  runtime gates. The exact Python 3.12/Torch 2.6+cu126 wheel omits `sm_120` and fails all three
  kernels; Python 3.12/Torch 2.10+cu128 includes `sm_120` and passes all three with finite outputs.
- `SUPERSEDED_NONDETERMINISTIC`: clean `WM-EBJEPA-INTEGRATION-001` passed its original dataset,
  train, checkpoint, planner, and memory gates, but omitted Python RNG seeding and independent
  replay. Its artifact is preserved and no result is accepted from it.
- `SMOKE_VALIDATED`: corrected `WM-EBJEPA-INTEGRATION-002` ran from clean `9a18008` over the pinned Two Rooms
  official dataset, Impala/GRU forward/backward, checkpoint restoration, MPPI call, and peak memory.
  All 12 gates pass, including exact two-process fingerprint replay (`16650872...234a1`).
  Upstream omits scipy, pandas, and PyYAML from the dependencies required by this path.
- `CONFIRMED_UPSTREAM_DEFECT`: the post-discovery planner confirmation ran from clean `da30443`.
  Under the frozen matched control, CEM had `0/32` action-norm violations (maximum `2.3474`) while
  MPPI had `32/32` (median `6.4485`, maximum `8.3018`) under configured maximum `2.45`. `DotWall`
  adds no independent bound check. This is an implementation result, not learned-model evidence.
- `CONFIRMED_UPSTREAM_CONFIG_MISMATCH`: `WM-EBJEPA-PLANNER-CONFIG-001` ran from clean `4f0cc80`.
  Both planner YAMLs specify `var_scale=1.5`, but MPPI expects `max_std`, silently consumes the
  unknown key through `**kwargs`, and executes with default `2.0`; CEM actually uses `1.5`.
- `SMOKE_VALIDATED`: `WM-EBJEPA-MPPI-CORRECTION-001` ran from clean `f58308a`. Across 32 seeds,
  the separately named constrained planner matched official MPPI exactly when bounds were disabled
  (maximum action/loss difference `0.0`) and produced zero cost-input or returned-action violations
  with the `2.45` bound enabled. This validates the correction, not trained planning competence.
- `PROFILED`: `WM-EBJEPA-TRAIN-RESOURCE-001` ran from clean `fed920e`. Every eager batch through
  the official 384 completed; batch 384 peaked at 5,821,693,952 reserved bytes. The official
  `torch.compile(jepa)` wrapper completed two `unroll` updates but Dynamo captured zero frames and
  zero graphs, so compilation is ineffective on the executed training entrypoint.
- `RUNNING`: `WM-EBJEPA-TRAIN-001` launched from clean commit `5065108` at
  `2026-07-21T21:53:46Z`. It freezes official seeds 1/1000/10000, batch 384, 12 epochs, all
  checkpoint hashes, and evaluation of epochs 9/10/11 under official-unbounded and
  bound-corrected MPPI at the actual proposal scale 2.0. Seed 1 is active; epochs 0--4 completed in
  roughly 900 seconds each, with latest non-monotone total loss `1.2349`. No competence
  result exists yet.
- `SMOKE_VALIDATED`: torch-aware Hugging Face Qwen3 adapter with selected residual,
  attention, MLP, and logit capture; replayable Torch interventions; registered donors/statistics;
  autograd preservation; ordered multi-site patch/restore with exact tiny-Qwen treatment replay;
  offline tests; and a preregistered Qwen3-0.6B smoke runner.
- `PREREGISTERED`: `LLM-QWEN-BINDING-MEDIATION-002` replaces v1 before any model forward. V1's clean
  audit at `4e6624f` still verifies 560 token treatments, but its paraphrase was not episode-paired.
  V2 freezes exact pairing, 24/6/6 disjoint pools, 56 candidates, `k<=4`, FP32 causal states,
  grouped competence, content/readback integrity, direct sufficiency/restoration, and matched
  nulls. Protected outcomes remain unopened pending the complete evaluator. This tests a mediator
  set, not a JEPA, circuit, J-space, or workspace.
- `SMOKE_VALIDATED`: `LLM-QWEN-001` executed pinned Qwen3-0.6B on the RTX 5070 Ti. Clean replay was
  exact, real autograd was nonzero, and five intervention operations changed hidden states/logits.
- `SMOKE_VALIDATED`: split-controlled 432-outcome Qwen intervention generator with
  resumable/checksummed sharded HDF5 storage and per-example local-linear direct probes.
- `WITHDRAWN`: `LLM-IJEPA-001` beat its originally registered comparators, but the so-called
  local Jacobian was a BF16 one-sided secant and the learned model is a supervised conditional
  bottleneck rather than a genuine target-encoder JEPA. `LLM-QWEN-JVP-AUDIT-001` is preregistered
  to re-execute FP32 effects and compare exact autograd JVP, converged central differences, and a
  quadratic baseline. The corrected v2 audit passed and exact JVP/quadratic transport decisively
  beat the legacy conditional bottleneck on all three seeds. The ranked graph remains `REJECTED`.
- `REJECTED` (numerical gate): the first exact-JVP audit passed five of six numerical gates and showed
  strong JVP/central agreement, but missed its absolute semantic-endpoint gate (`1.335e-4` versus
  `1e-5`). Its preliminary exact-JVP MSE (`0.6143`) beat the legacy conditional bottleneck
  (`3.1899`), but v1 could not decide H-LLM-01; this prompted the source-semantic v2 below.
- `COMPLETED_NEGATIVE`: v2 passed exact direct-source semantics, float32 endpoint, clean replay,
  and JVP/central-convergence gates. Exact JVP/quadratic MSE were `0.6143`/`0.07870`, versus
  conditional bottleneck `3.1899`; zero of three learned seeds passed. Restricted H-LLM-01 is
  withdrawn, not supported.
- `SMOKE_VALIDATED`: `LLM-CAPITAL-PATCH-001` executed 612 entity-disjoint Qwen capital donor
  patches from clean preregistration commit `95018cb`. All numerical and behavior gates passed:
  93.6% of patches changed the top token and 50% transferred the donor capital on the test split.
  Exact-JVP candidate agreement was 35.1%, versus 74.3% for quadratic Taylor; this is a direct
  causal dataset, not yet learned-meta-model or circuit evidence.
- `COMPLETED_NEGATIVE`: `LLM-TARGET-IJEPA-001` is a genuine 32-dimensional EMA/stop-gradient
  target-encoder JEPA with variance/covariance anti-collapse losses, executed from clean commit
  `3086cd4`. All three hypotheses failed on unseen entities. Every target latent missed the fixed
  effective-rank floor, and even oracle target-embedding decodes had normalized MSE above `1.0`.
  Exact JVP won full-vector fidelity, linear ridge won logit MSE, and quadratic Taylor won answer
  behavior; no single metric supports a nonlinear-JEPA advantage.
- `COMPLETED_MIXED`: `LLM-CONTEXT-GEOMETRY-001` executed from clean `49d68b7`. The real pooling-
  illusion and context-specificity hypotheses failed; the gauge-invariant coupling control passed.
  A function-preserving diagonal reparameterization changed naive pooled overlap about 120-fold
  while `J D^T` was invariant to `1.7e-16`. Unexpectedly, the 24-context train-mean Jacobian beat
  each test context's own Jacobian on finite logit MSE, correlation, and answer-set agreement.
- `COMPLETED_POSITIVE`: `LLM-POPULATION-JACOBIAN-001` confirmed all three post-discovery gates on
  six validation entities unused by v1 decisions. The 24-context mean cut local-Jacobian normalized
  logit MSE `0.737→0.354`, improved correlation `0.835→0.866`, and candidate agreement
  `0.300→0.533`; averaging showed a registered dose response and answer-row permutations failed.
  Corpus Jacobian averaging is prior art; the result is bounded finite-intervention fidelity, not a
  new lens algorithm, circuit, or workspace.
- `COMPLETED_MIXED`: `LLM-ELEMENT-LAYER-GEOMETRY-001` ran unchanged from clean `5d8de9a`; all four
  derivative gates passed. Donor-symbol control was `0/0%` at layers 18/21 and `90/100%` at 24/26
  on test (`0/0/60/100%` on validation), confirming H-LLM-08. Late population transports passed
  answer-row specificity (H-GEO-09), but the strict local/population inversion ratios failed on
  both splits, so H-GEO-08 and the cross-relation H-CROSS-03 conjunction are false. Late factual
  crystallization and population averaging remain prior art; no feature, circuit, or workspace is
  claimed.
- `REJECTED` (behavior gate): `LLM-STATE-LAYER-GEOMETRY-001` ran unchanged from clean `27ebe43` and
  passed every numerical gate, but clean validation/test accuracy was only `0.667/0.667` versus the
  frozen `0.75` floor. No scientific hypothesis is decided. Descriptively only, donor-control versus
  population-advantage Spearman was `0.949` on both splits and advantage turned positive at layer 26,
  not the registered layer 24; this is a new follow-up hypothesis, not rescued evidence.
- `COMPLETED_MIXED`: `LLM-STATE-ONESHOT-LAYER-GEOMETRY-001` ran unchanged from clean `c1daa46`.
  Competence and all numerical gates passed; late population semantic specificity passed. Exact
  boundary alignment failed because validation donor control crossed at layer 24 while population
  advantage crossed at 26; test aligned at 24. The early-control gate also failed on test (`0.233`
  at layer 21 versus `0.10`). H-GEO-12, H-LLM-12, and H-CROSS-05 are false as registered.
- `COMPLETED_MIXED`: `LLM-COUNTRY-CODE-LAYER-GEOMETRY-001` passed competence, numerics, monotone
  donor control, and semantic row-null specificity. The directional bounded-lag claim failed:
  validation population advantage crossed at layer 21 while 50% donor control crossed at 24; test
  crossed both at 21. H-GEO-14/H-CROSS-06 are false. This is a third-relation causal profile, not a
  new method, circuit, JEPA result, workspace, or SOTA claim.
- `COMPLETED_NEGATIVE`: `WM-LEWM-001` faithfully reproduces the small LeWorldModel recipe and all
  three seeds pass prediction/action/latent/probe gates. Planner interventions pass on two seeds,
  but hidden-patch specificity and the full restricted circuit pass only one; the aggregate graph
  is `REJECTED` and no workspace is found.
- `REJECTED` (numerical gate): `WM-POPULATION-JACOBIAN-001` ran from clean `89b2e14`, but its fixed
  12-node line integral underresolved stiff recurrent action chords. Gauge checks passed, yet no
  scientific transport claim survives. The global mean failed correlation/action-label controls;
  the provisional 3/3 advantage of within-context valid-action-vertex averaging requires a newly
  registered adaptive confirmation on the five untouched test goals.
- `CALIBRATION_ONLY`: `WM-ACTION-PATH-CALIBRATION-001` replaces that confounded vertex-MSE route
  with a validation-only decoded path-cancellation audit. Composite exact-JVP integration,
  refinement checks, local finite-effect error, concentration metrics, and within-action-pair nulls
  were run on exposed validation goals only. Horizon one converged, but horizon four remained
  underresolved for seeds 101/103. Cancellation/local-error association cleared stratified null p95
  by only `0.045/0.027/0.022`, and both normalized quantities share the direct-effect denominator;
  no scientific claim or protected-test launch is authorized.
- `CALIBRATION_ONLY`: `WM-ACTION-PATH-CALIBRATION-002` completed from clean `288f663` in
  `19,176.20` seconds on the identical exposed validation chords at 512/1024 nodes. It wrote no
  decisions and touched no protected test goals. Horizon-four integration/refinement maxima fell
  to `.0129/.0260`, `.0339/.0314`, and `.00139/.000250` for seeds 101/103/107; the first two remain
  underresolved. Spearman exceeded its stratified null p95 by only `.0454/.0283/.0165`. Because the
  shared-denominator confound and missing joint-conditional design remain, V2 is numerical/vector
  calibration only and the route is closed.
- `SMOKE_VALIDATED`: pinned Qwen3-4B bounded availability capture. Five residual layers at three
  selected semantic positions produced 180 rows in a 574,308-byte checksummed shard; this is not
  4B causal-intervention evidence.
- `SMOKE_VALIDATED`: updated `AUDIT-COMPLETE-001` passed all 14 explicit bounded completion criteria
  from clean synchronized commit `3593475`; the audit includes 105 tests, Ruff,
  provenance/checksum validation, the exact-JVP correction, multi-seed evidence, and retention of
  the rejected/null findings.
- `BLOCKED_EXTERNAL`: SkyJEPA reproduction until official implementation assets are available.

Real Qwen3-0.6B instrumentation, intervention-data generation, and meta-model experiments have run
on this GPU from clean commits. The original nonlinear-advantage result was withdrawn after its
corrective audit; the new behavior-changing capital dataset is eligible for a separately
preregistered genuine target-encoder JEPA. These are bounded Qwen results, not workspace evidence.
The explicit bounded completion suite is implemented and audited. Larger Tier-1/2 adapters and the
cluster-scale Qwen3-30B-A3B/SkyJEPA routes remain explicit research extensions or external blocks.

## Objective

Build reproducible experiments for two tracks:

1. Mechanistic interpretability of action-conditioned JEPA world models using deterministic physical environments, probes, causal interventions, circuit audits, planning effects, and workspace-candidate controls.
2. JEPA as an intervention-conditioned causal meta-model of Qwen computation, beginning with a mock transformer on CPU and continuing to Hugging Face Qwen only on suitable GPU hardware.

The repository distinguishes decodability from causal use. It does not claim consciousness, sentience, or equivalence between Anthropic J-space and any JEPA representation.

## Quick Start: CPU VPS

The default VPS path avoids model weights, external datasets, CUDA, Docker, Conda, Transformers, and large simulation suites. The GPT-2 Medium smoke was run only after an explicit user override.

```bash
export PYTHONPATH=src
python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/cpu_vps.yaml

python -m unittest discover -s tests -p 'test_*.py'
```

When a dev environment is allowed, the intended install path is:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,cpu]"
python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/cpu_vps.yaml
pytest -q tests/unit tests/integration -m "not gpu and not slow"
```

## Commands To Keep Working

```bash
python scripts/generate_tier0.py \
  --config configs/data/tier0_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/tiny_jepa_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/mock_qwen_intervention_jepa_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/tier0_mechanistic_study.yaml

python scripts/run_experiment.py \
  --config configs/experiments/workspace_discovery_study.yaml

python scripts/run_experiment.py \
  --config configs/experiments/manifold_workspace_study.yaml

python scripts/run_experiment.py \
  --config configs/experiments/multitask_workspace_study.yaml

python scripts/run_experiment.py \
  --config configs/experiments/gpt2_medium_intervention_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/gpt2_medium_mechanistic_study.yaml

python scripts/run_experiment.py \
  --config configs/experiments/gpt2_medium_semantic_composition_study.yaml

python scripts/audit_reproducibility.py
```

The Tier 0, tiny JEPA, mock-Qwen, four Milestone 3 JEPA studies, and all three GPT-2 Medium commands
are smoke validated. The mock-Qwen command uses a deterministic local mock model, not Qwen weights.

## GPU Continuation

This repository is now on the suitable GPU machine. On PowerShell, use:

```powershell
$env:PYTHONPATH = "src"
python -m causal_workspace_jepa.cli doctor `
  --resource-profile configs/resource/gpu_12gb.yaml
python -m unittest discover -s tests -p "test_*.py"
python scripts/audit_reproducibility.py

python scripts/run_experiment.py `
  --config configs/experiments/qwen3_0_6b_instrumentation_smoke.yaml

python scripts/generate_qwen_interventions.py `
  --config configs/experiments/qwen_intervention_dataset_v1.yaml

python scripts/run_experiment.py `
  --config configs/experiments/intervention_jepa_v1.yaml

python scripts/run_experiment.py `
  --config configs/experiments/lewm_small_reproduction_v1.yaml

python scripts/run_experiment.py `
  --config configs/experiments/lewm_population_geometry_v1.yaml

python scripts/run_experiment.py `
  --config configs/experiments/qwen_element_layer_geometry_v1.yaml

python scripts/run_experiment.py `
  --config configs/experiments/qwen_state_layer_geometry_v1.yaml

python scripts/run_experiment.py `
  --config configs/experiments/qwen_state_oneshot_layer_geometry_v1.yaml

# Optional primary-scale selected-site capture; downloads the pinned 8.06 GB Qwen3-4B repository.
python scripts/capture_qwen_activations.py `
  --config configs/llm/qwen3_4b_selected_layers.yaml

python scripts/audit_completion.py
```

The following Unix-style commands remain the intended clean-environment workflow:

```bash
source .venv/bin/activate

python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/gpu_12gb.yaml

python scripts/download_dataset.py \
  --name lewm_pusht \
  --manifest data/manifests/datasets.yaml \
  --confirm-size

python scripts/run_experiment.py \
  --config configs/experiments/lewm_pusht_probe.yaml

python scripts/run_experiment.py \
  --config configs/experiments/lewm_pusht_causal_patch.yaml

python scripts/run_experiment.py \
  --config configs/experiments/lewm_small_reproduction_v1.yaml

python scripts/capture_qwen_activations.py \
  --config configs/llm/qwen3_4b_selected_layers.yaml

python scripts/generate_qwen_interventions.py \
  --config configs/experiments/qwen_intervention_dataset_v1.yaml

python scripts/run_experiment.py \
  --config configs/experiments/intervention_jepa_v1.yaml
```

## Resource Modes

- `cpu_vps`: allowed for scaffolding, deterministic toy data, tiny NumPy models, mock Qwen dynamics, intervention schemas, unit/integration tests, and tiny smoke experiments. Hard disk floor: 4 GB free.
- `gpu_12gb`: continuation profile for selected Qwen3-0.6B/4B layers, LeWorldModel-scale smoke runs, reduced Jacobian tests, small SAEs, and direct interventions.
- `gpu_cluster`: required for Qwen3-30B-A3B hidden-state/autograd studies, large activation datasets, broad Jacobians, and SkyJEPA-scale reproduction.

## Dataset Policy

Tier 0 data is generated locally from explicit seeds and must stay below 512 MB by default. Raw datasets, model weights, activation shards, caches, and large generated artifacts are ignored by Git. Manifests, configs, checksums, summarized metrics, source, tests, and Markdown reports are committed.

## Repository Map

- `configs/`: resource, data, model, and experiment configs.
- `src/causal_workspace_jepa/`: package code.
- `scripts/`: runnable entry points for generation, experiments, capture, and audits.
- `tests/`: unit, integration, and scientific tests.
- `docs/`: registries, roadmap, hypotheses, risk log, results, and reproducibility notes.
- `data/manifests/`: committed dataset and prompt manifests.
- `artifacts/metrics/` and `artifacts/tables/`: committed summarized outputs.
- `papers/`: literature source registry and BibTeX.

## Latest Validated Results

Validated CPU smoke results:

- CPU resource profile loads through the local config parser.
- `doctor` checks disk, CPU count, and GPU-profile blocking.
- core intervention and model protocol dataclasses serialize correctly.
- no heavy download occurs.
- Tier 0 smoke generation created five deterministic local datasets totaling 79,217 bytes.
- Tiny JEPA smoke (`WM-T0-001`) ran from clean code commit `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- `WM-T0-001` evidence level: Availability. It validates plumbing and controls, not a causal mechanism.
- Tiny JEPA conditioned latent MSE: `1.09e-09`; mean baseline: `0.249`; no-action: `0.135`; shuffled-action: `0.153`.
- Planner true cost: `1.479`; random rollout mean cost: `1.939`.
- Mock intervention-JEPA smoke (`LLM-MOCK-001`) ran from clean code commit `85c1dbfbe9c824bcca415af13f4a6f34acc95267`.
- Mock intervention-JEPA MSE: `1.32e-07`; no-change: `0.002113`; mean-effect: `0.002113`; linear-context: `0.002113`; effect correlation: `0.99997`.
- `LLM-MOCK-001` evidence level: Availability. It validates the mock pipeline only and is not evidence about Qwen.
- Corrected Milestone 3 JEPA study (`WM-T0-002`) ran from clean commit `315d8cf`: displacement action
  R2 `~1.0` versus endpoint R2 values `-0.111` and `0.083`; actual action-input replay recovery `1.0`,
  max replay error `0.0`, L2 norm-matched latent control `-12.784`, random-action control `-0.316`.
  This is a trivial explicit-input circuit, not a learned workspace.
- Workspace study (`WM-T0-003`) ran from clean commit `5223a54`: the detector found its planted
  shared control and rejected disjoint consumers, but the JEPA uncertainty head failed (`R2=-3.639`),
  PCA was more damaging than the candidate, and random rollout controls went off-manifold. No shared
  causal subspace candidate or workspace was accepted.
- Deep manifold-controlled study (`WM-T0-004`) ran from clean commit `6785fb1` in `31.22` seconds.
  Test interval coverage was `0.887`, but OOD uncertainty AUC was only `0.574`; hidden uncertainty
  R2 was `-1.327`/`0.232`. Candidate direct damage (`3.652`/`1.239`) was below matched random-control
  p95 (`9.252`/`4.369`) at both hidden sites. No shared candidate or workspace was found.
- Multi-task study (`WM-T0-005`) ran from clean commit `7a9e510` in `57.51` seconds. Zero of three
  seeds passed. Held-out action MSE ratios were `0.712`, `1.012`, and `1.003` versus the registered
  `0.50`; minimum five-consumer sensitivity capture was `0.583`, `0.561`, and `0.574` versus
  `0.70`; task-counterfactual mean recovery was negative on every seed. No shared task-workspace
  candidate or workspace was found.
- GPT-2 Medium smoke (`LLM-GPT2-001`) directly intervened at `transformer.h.12.resid_post`; mean absolute logit delta was `0.0797`, intervention-JEPA MSE was `0.00220` vs no-change `0.0114`, effect correlation `0.976`. This is a small causal-mediation smoke, not a J-space/workspace discovery.
- Strengthened GPT-2 study (`LLM-GPT2-002`) generated 288 direct outcomes from clean commit
  `8fbab8c`. On unseen prompts and magnitude, local-Jacobian MSE was `7.79e-7`, bilinear
  Intervention-JEPA `0.00350`, linear regression `0.00636`, and no-change `0.00646`. Bilinear
  compression therefore beat weak regressions but did not beat the strong Jacobian. On held-out
  layer 18 it was worse than no-change. No intervention changed the top token.
- Semantic-composition study (`LLM-GPT2-003`) generated 72 outcomes from clean commit `1e57e30` in
  `392.85` seconds. Held-out composition interaction was only `0.043%` of effect power. Prompt-local
  Jacobian MSE was `0.000990`, versus MLP `0.725`, bilinear `1.346`, and no-change `0.418`. The
  bilinear model fit seen-prompt compositions but failed on unseen prompts. Two of 72 interventions
  changed the top token. No semantic feature, J-space, workspace, or Qwen result is claimed.
- Qwen instrumentation smoke (`LLM-QWEN-001`) ran from clean commit `0d6a37b` at pinned revision
  `c1899de...`. Deterministic replay error was `0.0`; selected-logit gradient norm was `0.944`;
  mean absolute logit deltas were `0.0565` zero, `0.0111` mean, `0.00805` donor patch/resample, and
  `0.0321` steer. This is real-Qwen causal-mediation smoke, not a circuit or workspace result.
- Qwen intervention data (`LLM-INTDATA-001`) ran from clean commit `0aa80ac`: 432 effects across
  12 split prompts and three layers in `33.85` seconds; 17 edits changed the top token. The 412 KB
  HDF5 shard checksum is committed in the manifest. Local-linear MSE was `139.83`, which motivates
  but does not prove a nonlinear meta-model advantage.
- Qwen Intervention-JEPA (`LLM-IJEPA-001`) ran from clean commit `a54f2ed` on the frozen 432-effect
  dataset with seeds 61/67/71. Primary held-out MSE/correlation were `3.923`/`0.677`, versus
  no-change `7.243`, mean `7.119`, linear `27.169`, MLP `9.642`, local Jacobian `116.156`, and
  nearest-neighbor `5.720`. On the unseen resampling operation, Intervention-JEPA reached `2.141`
  MSE but nearest-neighbor was slightly better at `2.095`. Direct execution on 16 edits over four
  new prompts yielded effect correlation `0.673`, but predicted coordinate 128 was not the observed
  winner (coordinate 0); precision@1 was `0`, so H-LLM-06 failed and the graph is rejected.
- Corrective JVP audit v1 ran from clean commit `686368e` in `167.99` seconds. Exact JVP and central
  differences agreed (median relative error `2.49e-4`), and exact JVP/quadratic MSE were
  `0.6143`/`0.07870` versus legacy conditional-bottleneck `3.1899` and old BF16 secant `120.8994`.
  The audit nevertheless failed its downstream semantic-endpoint absolute gate, so these are
  preliminary audit measurements; v2 subsequently resolved H-LLM-01 as withdrawn.
- Corrective v2 (`LLM-QWEN-JVP-AUDIT-002`) ran from clean commit `a779ff6` in `170.77` seconds.
  Direct source semantics were exact, all derivative gates passed, and zero of three learned seeds
  beat exact JVP/quadratic Taylor. The old nonlinear-advantage claim is withdrawn.
- Faithful small LeWorldModel (`WM-LEWM-001`) ran unchanged from clean commit `4dbc388` with seeds
  101/103/107 in `54.50` seconds. All three reproduction gates passed: test embedding MSE was
  `0.134`-`0.292`, clean/shuffled ratios `0.155`-`0.643`, latent standard deviation
  `0.762`-`0.862`, and nonlinear state-probe R2 `0.746`-`0.879`. The hidden action-subspace planner
  intervention selectively changed future trajectories, candidate costs, and selected actions on
  two seeds. Donor-patch specificity and the full restricted circuit passed only seed 107, so the
  replicated claims failed and the aggregate graph is rejected. Five-consumer workspace tests were
  null on every seed.
- Qwen3-4B selected-site capture (`LLM-QWEN-CAPTURE-001`) ran from clean commit `55087ea`: exact
  pinned revision, 180 rows, one 574,308-byte checksum-verified shard, and all budget gates passed.
- Updated final audit (`AUDIT-COMPLETE-001`) ran from clean synchronized commit `3593475`. All 14
  explicit criteria passed; 105 tests, Ruff, Git diff, and reproducibility checks returned zero. It
  now requires the valid exact-JVP audit and withdrawn H-LLM-01 result. This audit
  certifies implementation/evidence coverage, not a positive workspace or circuit discovery.

## Limitations

This host has a 12,227 MiB RTX 5070 Ti and the `gpu_12gb` profile is active. The bounded Qwen3-0.6B
pipeline and bounded Qwen3-4B capture are validated; the source-informed small LeWorldModel run is retained
as a mixed/negative result. Qwen3-30B-A3B, broad all-layer Jacobians, and large video
training remain `gpu_cluster` work. The pre-existing GPT-2 artifacts came from a different Linux CPU
host; their ignored activation shards are absent here and the audit reports their checksums as
skipped until regenerated.
