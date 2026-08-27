# AGENTS.md — Causal Workspace JEPA

Compact navigation map. Full historical operating rules are archived in
`docs/agents/AGENTS_FULL.md`. Current recovery task: `docs/codex-handoff/00_INDEX.md`.
Scientific invariants: `docs/codex-handoff/02_SCIENTIFIC_INVARIANTS.md`.

## Mission

Build a reproducible research codebase for:

1. Mechanistic interpretability of action-conditioned JEPA world models.
2. Intervention-conditioned JEPA as a causal meta-model of open-weight Qwen.

Do not claim consciousness, sentience, or literal equivalence to Anthropic J-space.
A workspace candidate must be tested; it must not be assumed.

## Non-negotiable scientific rules

- Never change frozen thresholds after the outcome they govern is observed.
- `CRCT-STAGE0-HARD-002` remains `NEGATIVE_RESULT`.
- Qwen Binding V3 remains `INELIGIBLE_TASK_PHASE0`.
- Calibration-only competence recovery cannot rescue V3.
- Do not open `test` or `paraphrase` unless a new prospective protocol says so.
- Do not fabricate provenance, metrics, or experiment IDs.
- Candidate-only accuracy is never the competence eligibility metric.
- Direct-delta and strong differential baselines are mandatory for residual claims.
- Literal graph recall is not epsilon-functional sufficiency.
- No concurrent write agents in the same checkout.

## Start here

- `README.md`, `docs/RESULTS.md`, `docs/EXPERIMENT_REGISTRY.md`
- Protocol/result file for the experiment you touch
- `docs/exec-plans/active/` for the current plan
- Resource profile before any download or GPU job

## Engineering gate

Before finishing a milestone: relevant tests, Ruff (`E4,E7,E9,F`), reproducibility
audit, Git status/diff, docs sync. CPU CI: `.github/workflows/cpu-ci.yml`.

## Resource modes

- `cpu_vps`: no Qwen/JEPA weight downloads, no CUDA installs, tiny smoke only.
- `gpu_12gb`: Qwen3-0.6B/4B selected-layer work; estimate VRAM first. Qwen
  tokenizer/model tests and forwards **require CUDA**. There is no CPU fallback.
- `gpu_cluster`: Qwen3-30B-A3B full hidden-state studies.

GitHub `cpu-ci.yml` never downloads Qwen and excludes `@pytest.mark.gpu`.
Run the GPU suite with `scripts/run_gpu_suite.ps1` or `.github/workflows/gpu-ci.yml`.
Skipped GPU tests are `SKIPPED_RESOURCE`, never `PASS`.

## Current critical path (2026-08-27)

1. Keep HARD-002 and V3 frozen.
2. Confirmation passed; do not execute `LLM-QWEN-BINDING-ALGEBRA-004` until
   a later authorization commit. V3 stays ineligible.
3. **Primary track: mechanistic interpretability.** Coalition-aware CRCT →
   IBD validation → learned action-Δ mechanisms (supervised WM first; a
   JEPA-objective successor is a later ID) → contact → cross-model
   equivalence. Planning, geometry, and probes are secondary readouts.
4. `CRCT-COALITION-IBD-001` confirmation is smoke. `CRCT-COALITION-IBD-002`
   remains `PREREGISTERED_NOT_RUN` (not executed). `CRCT-COALITION-IBD-003`
   is `MECHANISM_RECOVERY_PASSED` (synthetic IBD).
   `CRCT-JEPA-ACTION-DELTA-001` is `MODEL_INCOMPETENT` (confirmation closed).
   `CRCT-LEARNED-WM-ACTION-DELTA-002` is `INCONCLUSIVE` (confirmation closed).
   `CRCT-LEARNED-WM-ACTION-DELTA-003` is `MODEL_INCOMPETENT` (confirmation
   closed; CRCT not run). `CRCT-LEARNED-WM-ACTION-DELTA-004` is
   `PREREGISTERED_NOT_RUN` (ladder 800/2000/5000; new seeds). Do not mutate
   001/002/003.
5. Platonic WM + LeFlow paper-scale work remains plan-only (no DINO-WM/LeWM
   downloads, no stitching). CPU `WM-PLATONIC-MKNN-001` passed with an
   encoder-geometry caveat (Availability only). CPU `WM-LEFLOW-AMORTIZE-001` is
   `NEGATIVE_RESULT`. CPU `WM-AMORTIZED-PLANNING-MINIPUSH-002` is
   `UNINFORMATIVE_SUBSTRATE`. `WM-AMORTIZED-PLANNING-REACHABLE-003` is
   `DRAFT_NOT_PREREGISTERED` (not authorized). Do not execute 004, IBD-002,
   stitching, or Reachable-003.

Detailed milestone, dataset, interface, and literature requirements remain in
`docs/agents/AGENTS_FULL.md` and the `docs/` registries.
