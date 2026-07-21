# VPS_RUNBOOK.md — What Codex Can Do Now

> Historical runbook note (2026-07-21): the CPU-VPS phase described below is complete. The current
> host is the RTX 5070 Ti/32 GB RAM machine and must follow the `gpu_12gb` continuation rules. This
> document is retained to preserve the original resource boundary and handoff commands.

## Current GPU continuation: EB-JEPA

The official EB-JEPA source is pinned in an ignored checkout and must never be silently upgraded:

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_eb_jepa.py --target .cache/upstream/eb_jepa
python scripts/prepare_eb_jepa_torch_runtimes.py --mode both
python scripts/doctor_eb_jepa.py
python scripts/run_experiment.py --config configs/experiments/eb_jepa_official_contract_smoke.yaml
python scripts/run_experiment.py --config configs/experiments/eb_jepa_runtime_compatibility.yaml
```

Expected source revision is `966e61e9285b3a876f49b9774e9720d9a99a7925`. The contract smoke
uses the real upstream Impala encoder and one-layer 512-dimensional GRU, but random weights and CPU
execution only. Passing it validates shapes, exact gate decomposition (`atol=1e-6`), and intervention
plumbing; it does not reproduce planning or establish a circuit/workspace.

The official `pyproject.toml` requires Python 3.12 and Torch 2.6. The isolated 2.6+cu126 wheel
detects compute capability 12.0 but its compiled architecture list stops at SM90; matmul, Conv2D,
and GRU each fail with `no kernel image is available`. The controlled Python 3.12/Torch
2.10+cu128 environment includes SM120 and passes all three. Therefore use the latter for local GPU
training, label every run `compatibility_deviation`, and preserve the exact-pin CPU probe. Do not
alter the working Qwen environment in place.

Prepare the isolated Two Rooms closure without replacing Torch:

```powershell
python scripts/prepare_eb_jepa_two_rooms.py
$env:PYTHONPATH='src'
python scripts/run_experiment.py --config configs/experiments/eb_jepa_two_rooms_integration_smoke.yaml
python scripts/run_experiment.py --config configs/experiments/eb_jepa_planner_constraint.yaml
```

Do not use upstream `uv sync` in this environment. The committed lock supplies the undeclared
scipy/pandas/PyYAML imports and exact resolved transitive versions while deliberately excluding
Torch. The integration and planner-constraint artifacts are engineering evidence only.
The current integration config is V2 and runs two independent deterministic probes; do not restore
the historical V1 output path or treat its incomplete RNG gates as a pass.

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

## Current RTX 5070 Ti continuation (2026-07-21)

The project is now on the intended 12 GB GPU host. `configs/resource/gpu_12gb.yaml` is active;
Qwen3-0.6B causal studies, a bounded Qwen3-4B capture, and the source-informed small LeWorldModel study
have been executed from clean commits. Qwen3-30B-A3B remains a cluster-scale target.

High-resolution recurrent-JEPA action-path calibration must stream exact Jacobians and immediately
decode/detach them to CPU. The 512/1024-node profile uses `jacobian_outer_batch_size: 64` and
`jacobian_chunk_size: 2`; two earlier non-streamed launches exhausted memory and emitted no
scientific artifact. A valid launch records the clean starting commit before any calculation. The
command can exceed the default 20-minute orchestration timeout even while the child Python process
continues normally, so monitor both the configured metrics path and the process rather than
launching a duplicate.

Newer action-path runner commits also write an ignored atomic `*.progress.json` after each completed
seed/horizon. A restart resumes only from the same experiment ID, exact config bytes, and Git commit;
stale progress aborts. The v2 process launched from `288f663` predates this protection. Before the
next high-resolution launch, benchmark bounded outer/chunk batch pairs on the target GPU rather than
guessing from VRAM alone.

Do not promote a converged cancellation/local-error correlation to protected test. Both normalized
quantities divide by the decoded direct-effect norm, while v2 omits scalar path length at its lower
resolution and stores a floored direct norm. A proposed retrospective audit was rejected before
commit; there are also only two chords per action pair, insufficient for a joint conditional null.
Retain v2 as numerical/vector calibration and do not launch a downstream analysis from it.

The working scientific paper is built locally with:

```powershell
cd papers
latexmk -pdf -interaction=nonstopmode -halt-on-error causal_workspace_jepa.tex
```

Compile inside `papers/` so BibTeX uses the repository's `references.bib`; this MiKTeX installation
also contains an unrelated global file with the same basename. Generated PDF and LaTeX auxiliary
files are ignored, while the `.tex` source and bibliography are committed.
