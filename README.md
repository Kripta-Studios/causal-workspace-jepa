# Causal Workspace JEPA

CPU-first research codebase for action-conditioned JEPA world-model interpretability and intervention-conditioned LLM meta-modeling.

## Current Status

- `SMOKE_VALIDATED`: repository control plane, resource profiles, `doctor`, typed interfaces, and standard-library tests.
- `IMPLEMENTED_UNVALIDATED`: Tier 0 generators, tiny NumPy JEPA, random-shooting planner, save/load, and smoke experiment runner. The clean committed run is pending.
- `SCAFFOLDED`: documentation registries, data/artifact policy, package tree, provenance helpers.
- `NOT_STARTED`: intervention experiments and mock Qwen experiments.
- `BLOCKED_RESOURCE`: real Qwen hidden-state instrumentation, GPT-2/GPT-style real model smoke tests, published JEPA checkpoints, Tier 1/Tier 2 datasets, GPU Jacobian/SAE work.
- `BLOCKED_EXTERNAL`: SkyJEPA reproduction until official implementation assets are available.

No GPU experiment has been run or claimed on this VPS.

## Objective

Build reproducible experiments for two tracks:

1. Mechanistic interpretability of action-conditioned JEPA world models using deterministic physical environments, probes, causal interventions, circuit audits, planning effects, and workspace-candidate controls.
2. JEPA as an intervention-conditioned causal meta-model of Qwen computation, beginning with a mock transformer on CPU and continuing to Hugging Face Qwen only on suitable GPU hardware.

The repository distinguishes decodability from causal use. It does not claim consciousness, sentience, or equivalence between Anthropic J-space and any JEPA representation.

## Quick Start: CPU VPS

The current VPS path avoids model weights, external datasets, CUDA, Docker, Conda, Transformers, and large simulation suites.

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

python scripts/audit_reproducibility.py
```

The Tier 0 and tiny JEPA commands are implemented; mock Qwen remains scheduled for Milestone 2.

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

Milestone 0 validates only the control plane:

- CPU resource profile loads through the local config parser.
- `doctor` checks disk, CPU count, and GPU-profile blocking.
- core intervention and model protocol dataclasses serialize correctly.
- no heavy download occurs.

The tiny JEPA smoke experiment has an exploratory dirty-run pass, but the committed metrics are intentionally pending until the code is committed and rerun cleanly.

## Limitations

This VPS has no GPU and limited free disk. Real Qwen instrumentation through Hugging Face, published JEPA checkpoints, Tier 1/Tier 2 datasets, and scientifically meaningful Jacobian/SAE work remain blocked until a larger resource profile is active.
