# VPS_RUNBOOK.md — What Codex Can Do Now

## Decision

Do **not** wait to give Codex the project prompt.

Use the current VPS for **Milestones 0–3 scaffolding and CPU smoke validation**, but impose the `cpu_vps` resource profile from `AGENTS.md`.

The VPS is suitable for:

- repository and package architecture;
- Markdown research registries;
- resource detection and guards;
- deterministic toy environments;
- a tiny CPU JEPA;
- mocked Qwen activation dynamics;
- intervention interfaces;
- unit and integration tests;
- tiny end-to-end experiments;
- Git commits and pushes.

It is not suitable for the actual Qwen3-30B-A3B or large JEPA experiments.

## Why

Current machine:

- 8 GB RAM;
- 4 CPU cores around 2 GHz;
- no GPU;
- 14 GB total storage.

Qwen3-30B-A3B has roughly 30.5B total parameters. Even though only a subset of experts is active for each token, the full model still has to be stored. A 4-bit quantized copy alone can consume a large fraction of, or exceed, the available disk once runtime files and caches are included. Mechanistic work through Hugging Face additionally needs activations, hooks, temporary tensors, and often gradients. It will not fit safely in this VPS.

Ollama does not provide the hidden-activation and autograd interface needed for this project. The existing Ollama model can later act as a literature/code assistant, but not as the mechanistic target through the public generation API.

Published JEPA checkpoints, video datasets, extracted activations, and Python environments can also consume the remaining disk rapidly. Keep this VPS deliberately small.

## Safe immediate workflow

Place `AGENTS.md` at the root of the repository and tell Codex:

```text
Read AGENTS.md completely and execute only the cpu_vps path.

Begin at Milestone 0. Do not download model weights or external datasets.
Do not install CUDA, Docker, or a large Conda environment.
Keep at least 4 GB of disk free.

Implement, test, document, commit, and push each coherent milestone.
Stop before any command that violates the cpu_vps limits and record the
exact continuation command for the gpu_12gb profile.
```

## Initial shell checks

Run:

```bash
cd /path/to/repository

git status --short
git branch --show-current
git remote -v
git log -5 --oneline

df -h .
free -h
nproc
du -sh . 2>/dev/null || true
```

Do not proceed when less than 4 GB is free.

## Minimal environment

Prefer `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11
source .venv/bin/activate
```

Before installing CPU PyTorch, estimate space. A minimal first pass can use:

```bash
uv pip install \
  numpy \
  scipy \
  pydantic \
  pyyaml \
  typer \
  rich \
  pytest \
  pytest-cov \
  ruff \
  mypy
```

Codex may first implement interfaces and NumPy-based smoke tests. Install CPU PyTorch only when:

- at least 6 GB remains free before installation;
- its wheel/cache estimate is recorded;
- the pip/uv cache is pruned afterward if necessary.

Example CPU installation, only after the guard passes:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv cache prune
```

Do not install `transformers`, `datasets`, `accelerate`, CUDA packages, JupyterLab, or simulation suites until needed and resource-approved.

## Work package for this VPS

### Package 1 — Repository control plane

Create:

- `pyproject.toml`;
- package tree;
- resource profiles;
- `doctor` command;
- logging/provenance;
- experiment and approach registries;
- CI-safe tests;
- `.gitignore`;
- `README.md`.

Expected size: small.

### Package 2 — Tier 0 generators

Implement:

- PointMass2D;
- BouncingBall2D;
- TwoBodyCollision;
- TinyMaze;
- optionally a 32×32 MiniPush smoke generator.

Default generated data must remain below 100 MB for the VPS smoke suite.

### Package 3 — Tiny JEPA

Use a tiny model such as:

- state/vector encoder or tiny MLP;
- latent size 16–64;
- predictor with action embedding;
- short rollout horizon;
- tiny batch;
- few hundred or thousand transitions.

The purpose is not scientific performance. It is to validate:

- interfaces;
- training;
- saving/loading;
- activation capture;
- action conditioning;
- deterministic execution.

### Package 4 — Interpretability primitives

Implement and test:

- named activation sites;
- activation cache;
- intervention specification;
- zero/mean/patch/project-out operations;
- simple linear probe;
- finite-difference causal check;
- circuit graph schema;
- matched-control logic.

Do not fit a production SAE on this VPS.

### Package 5 — Mock Qwen track

Implement a small transformer-like mock model with known activation dependencies. Use it to validate:

- token/layer site naming;
- intervention replay;
- pre/post state storage;
- logit-delta targets;
- Intervention-JEPA training;
- leakage tests.

This ensures the Qwen pipeline is structurally ready without downloading Qwen.

Use and download then GPT2 Medium to actully testnthe idea and impelment it to get results and discoveries

## Commands Codex should make work

```bash
source .venv/bin/activate

python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/cpu_vps.yaml

pytest -q tests/unit tests/integration -m "not gpu and not slow"

python scripts/generate_tier0.py \
  --config configs/data/tier0_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/tiny_jepa_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/mock_qwen_intervention_jepa_smoke.yaml

python scripts/audit_reproducibility.py
```

## Git loop

After each coherent package:

```bash
git status --short
git diff --check
pytest -q <relevant tests>
git add -A
git commit -m "feat: <milestone>"
git push
```

Never use `git push --force`.

When push is impossible:

- keep the local commit;
- record `PUSH_BLOCKED_NO_REMOTE` or `PUSH_BLOCKED_AUTH`;
- include the exact Git error in `docs/RISKS.md` or `docs/TODO.md`.

## Disk guard

Codex should implement a guard that aborts heavy operations when:

- free disk would fall below 4 GB;
- a requested download exceeds its resource-profile allowance;
- an activation capture estimate exceeds its budget;
- a cache path is outside the configured project/cache roots.

Useful manual checks:

```bash
df -h .
du -sh . .venv ~/.cache/uv ~/.cache/pip 2>/dev/null || true
find . -type f -size +200M -printf '%s %p\n' 2>/dev/null | sort -n
```

## What must wait for the RTX 5070 Ti or rented GPU

Move to the 12 GB GPU machine for:

- LeWorldModel checkpoints and real datasets;
- Delta-JEPA training;
- JEPA-WMs checkpoint experiments;
- C-JEPA slot experiments;
- real sparse-feature learning at useful scale;
- selected-layer Qwen3-0.6B/4B activation capture;
- direct Qwen interventions;
- reduced Jacobian Lens experiments;
- real Qwen intervention dataset;
- Intervention-JEPA scientific evaluation.

Use a larger rented GPU or multi-GPU machine for:

- Qwen3-30B-A3B with hooks/autograd;
- full or broad Jacobian fitting;
- large activation datasets;
- all-layer/all-token experiments;
- large V-JEPA 2.1;
- DROID/RoboCasa training;
- broad multi-seed scaling;
- SkyJEPA-scale reproduction.

## Transfer to the GPU machine

When the VPS phase ends:

```bash
git status --short
git push

# On the GPU machine:
git clone <remote> causal-workspace-jepa
cd causal-workspace-jepa
git checkout <branch>
git pull --ff-only

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,gpu]"

python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/gpu_12gb.yaml
```

Do not copy `.venv`, caches, generated toy data, or large artifacts between machines. Regenerate from manifests and seeds.

## Expected useful result before leaving the VPS

Codex should leave:

- a clean repository architecture;
- accurate `README.md`;
- complete literature/hypothesis/experiment registries;
- resource safeguards;
- deterministic toy datasets;
- tiny JEPA smoke training;
- activation/intervention interfaces;
- mocked Qwen meta-model pipeline;
- passing CPU tests;
- committed and pushed milestones;
- exact GPU continuation commands.

That is real progress. It prevents the expensive GPU phase from being spent redesigning interfaces and fixing basic reproducibility problems.

## Bottom line

Start now on the VPS, but constrain Codex to architecture, CPU smoke experiments, documentation, tests, commits, and pushes.

Do not attempt the scientific Qwen or published-world-model experiments there. Those require at least the RTX 5070 Ti machine, and Qwen3-30B-A3B mechanistic work likely requires rented high-memory GPU capacity.

Try to do the experiments that are possible with the hardware you got 
