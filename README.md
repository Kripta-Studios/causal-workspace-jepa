# Causal Workspace JEPA

CPU-first research codebase for action-conditioned JEPA world-model interpretability and intervention-conditioned LLM meta-modeling.

## Current Status

- `SMOKE_VALIDATED`: repository control plane, resource profiles, `doctor`, typed interfaces, standard-library tests, Tier 0 generators, tiny NumPy JEPA, random-shooting planner, save/load, and the tiny JEPA smoke experiment.
- `SMOKE_VALIDATED`: NumPy intervention operators, activation cache, ridge probes, sparse dictionary, finite-difference lenses, circuit graph I/O, mock transformer, and mock intervention-JEPA smoke runner.
- `SMOKE_VALIDATED`: adversarial Milestone 3 re-audit. The flawed original patch metric was
  superseded by a clean replayable action-input intervention with L2 norm-matched controls.
- `SMOKE_VALIDATED`: multi-consumer workspace detector with planted shared/disjoint controls. The
  tiny JEPA result is null after uncertainty, PCA, and off-manifold rollout-control failures.
- `SMOKE_VALIDATED`: GPT-2 Medium hidden-state intervention smoke under the user's explicit override.
- `IMPLEMENTED_UNVALIDATED`: strengthened GPT-2 Medium study with 288 batched direct interventions,
  prompt/magnitude/layer holdouts, a trained MLP, and local/corpus Jacobian baselines.
- `SCAFFOLDED`: documentation registries, data/artifact policy, package tree, provenance helpers.
- `NOT_STARTED`: real Qwen experiments and published world-model experiments.
- `BLOCKED_RESOURCE`: real Qwen hidden-state instrumentation, published JEPA checkpoints, Tier 1/Tier 2 datasets, GPU Jacobian/SAE work.
- `BLOCKED_EXTERNAL`: SkyJEPA reproduction until official implementation assets are available.

No GPU experiment has been run or claimed on this VPS.

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
  --config configs/experiments/gpt2_medium_intervention_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/gpt2_medium_mechanistic_study.yaml

python scripts/audit_reproducibility.py
```

The Tier 0, tiny JEPA, mock-Qwen, Milestone 3 JEPA, and both GPT-2 Medium commands are
implemented. `LLM-GPT2-002` is not validated until it runs from committed code. The mock-Qwen
command uses a deterministic local mock model, not Qwen weights.

## GPU Continuation

Run only on a suitable GPU machine after pulling the pushed branch:

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
- GPT-2 Medium smoke (`LLM-GPT2-001`) directly intervened at `transformer.h.12.resid_post`; mean absolute logit delta was `0.0797`, intervention-JEPA MSE was `0.00220` vs no-change `0.0114`, effect correlation `0.976`. This is a small causal-mediation smoke, not a J-space/workspace discovery.

## Limitations

This VPS has no GPU and limited free disk. Real Qwen instrumentation through Hugging Face, published JEPA checkpoints, Tier 1/Tier 2 datasets, and scientifically meaningful Jacobian/SAE work remain blocked until a larger resource profile is active. GPT-2 Medium weights are cached locally under `.cache/` and ignored by Git.
