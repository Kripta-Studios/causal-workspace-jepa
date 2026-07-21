# Causal Workspace JEPA

Reproducible research codebase for action-conditioned JEPA world-model interpretability and intervention-conditioned LLM meta-modeling.

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
- `SCAFFOLDED`: documentation registries, data/artifact policy, package tree, provenance helpers.
- `ACTIVE`: GPU continuation on an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM, 32 logical CPU cores, and about 370 GB free at the 2026-07-21 audit.
- `SMOKE_VALIDATED`: torch-aware Hugging Face Qwen3 adapter with selected residual,
  attention, MLP, and logit capture; replayable Torch interventions; registered donors/statistics;
  autograd preservation; offline tiny-Qwen tests; and a preregistered Qwen3-0.6B smoke runner.
- `SMOKE_VALIDATED`: `LLM-QWEN-001` executed pinned Qwen3-0.6B on the RTX 5070 Ti. Clean replay was
  exact, real autograd was nonzero, and five intervention operations changed hidden states/logits.
- `IMPLEMENTED_UNVALIDATED`: split-controlled 432-outcome Qwen intervention generator with
  resumable/checksummed sharded HDF5 storage and per-example local-linear direct probes.
- `NOT_STARTED`: real Qwen experiments and published world-model experiments; their former local resource blocker has been removed, but no result is claimed yet.
- `BLOCKED_EXTERNAL`: SkyJEPA reproduction until official implementation assets are available.

No GPU scientific experiment has yet been run or claimed. The GPU profile and CUDA runtime have only passed control-plane checks.

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
```

The following Unix-style commands remain the intended clean-environment workflow; the Qwen and
published-model commands are still implementation targets rather than validated commands:

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

## Limitations

This host has a 12,227 MiB RTX 5070 Ti and the `gpu_12gb` profile is active. Qwen and published-JEPA
code is still placeholder-only at the start of the GPU continuation, so hardware availability must
not be confused with completed evidence. Qwen3-30B-A3B, broad all-layer Jacobians, and large video
training remain `gpu_cluster` work. The pre-existing GPT-2 artifacts came from a different Linux CPU
host; their ignored activation shards are absent here and the audit reports their checksums as
skipped until regenerated.
