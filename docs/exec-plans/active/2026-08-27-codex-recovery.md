# Execution plan — 2026-08-27 Codex recovery

Authoritative plan for `CODEX_MASTER_PROMPT.md` on `crct-stage0-001`.

## Verified baseline

- Branch: `crct-stage0-001`
- HEAD at start: `3f58ec5a25ad8374c4678f94171a18892fdc0ead`
- Remote: `origin/crct-stage0-001` (in sync at start)
- Hardware: RTX 5070 Ti Laptop 12 GB; system Python 3.14.2 + torch 2.10.0+cu128; repo `.venv` is Python 3.13.5 CPU torch
- Model routing: `UNVERIFIED` for Luna/Sol Codex role overrides; parent continued without claiming routed workers ran. Independent review used available Sol-medium subagents, not native Codex role routing.

## Scientific invariants (frozen)

- `CRCT-STAGE0-HARD-002` = `NEGATIVE_RESULT`
- `LLM-QWEN-BINDING-ALGEBRA-003` / V3 B0 = `INELIGIBLE_TASK_PHASE0`
- Competence recovery cannot rescue V3
- Protected `test` / `paraphrase` remain closed

## Untracked classification

| Path | Class | Action |
|---|---|---|
| `artifacts/metrics/qwen_binding_algebra_token_contract_audit.json` | C/D scratch audit, missing provenance | moved to `artifacts/local_scratch/` |
| `artifacts/metrics/qwen_capital_crct_dev_hotfix_check.json` | D; contains train/validation/test diagnostics | moved; **not used** for any threshold or selector |
| `artifacts/metrics/bridge002_yaml_hotfix_audit/` | C | moved |
| `*.patch`, `J_A_AUTOMATION_*.md` | C local pack/scratch | left untracked |
| `.codex/`, `docs/codex-handoff/` | A agent harness | commit |
| new source/tests/configs/docs/CI | A | commit |

No provenance was invented for the moved JSON files.

## Decisions

### D-001 — Ruff rule pin

- Evidence: `ruff check .` with extra local rules reported 325 issues; isolated E4/E7/E9/F matched the handoff's 5 issues.
- Decision: pin `[tool.ruff.lint] select = ["E4", "E7", "E9", "F"]` and `ruff==0.15.22`.

### D-002 — Competence recovery adjudication

- Evidence: `artifacts/reports/qwen_competence_recovery/QWEN-BINDING-COMPETENCE-RECOVERY-001_20260818_101736/`
- Selected: `qwen_chat_prefill_v1`; clean 0.9375; direct 0.979166...; calibration only; protected splits empty.
- Decision: adjudicate `COMPETENCE_RECOVERY_PROMPT_SELECTED` as development-only. V3 remains ineligible.

### D-003 — Confirmation tokens

- Tokenizer-only, no model forward, revision `c1899de...`
- Rejected: onyx, silt, flint, grove, indigo (not strict spaced single-token)
- Frozen confirmation keys: maple, quartz, ridge, frost
- Frozen confirmation values: teal, ivory, coral, peach
- Seed 701, 32 episodes, split name `confirmation`

### D-004 — Coalition thresholds a priori

- epsilon 0.02 and related floors frozen in code before confirmation seeds 811/823/829
- HARD-002 seeds 1009/2027/4093 rejected by constructor

### D-005 — IBD-001 gauge caveat

- Independent review: original Spearman compared an untransformed copy.
- Decision: apply compensated gauge in code now; do not relabel IBD-001 confirmation; preregister IBD-002.

### D-006 — 004 is draft-only

- Confirmation pass authorizes drafting `LLM-QWEN-BINDING-ALGEBRA-004` / `CRCT-QWEN-BRIDGE-003`.
- `execution_authorized: false`. No Qwen forward.

## Phase checklist

- [x] P0 hygiene: ruff, audit, scratch move, CI
- [x] P0 competence recovery adjudication from artifact
- [x] P0 confirmation protocol preregistered
- [x] P0 confirmation model-forward (CUDA Python after protocol commit)
- [x] P1 coalition IBD evaluator + tests
- [x] P1 Intervention-JEPA fair-baseline policy module
- [x] P2 LeVLJEPA MiniPush factorial preregistered, not run
- [x] Platonic WM + LeFlow integration plan (docs only)
- [x] Independent review of full diff (Sol-medium subagents; Codex Sol High routing UNVERIFIED)
- [x] P1 review remediations: real gauge, committed ledger copy, Ruff pin, 004 draft, failure-path test
- [x] Full non-protected suite after this docs/code pass (315 passed, Ruff green, audit SMOKE_VALIDATED)
- [ ] Push remaining commits to origin/crct-stage0-001

## Review notes

Protocol-boundary verifier: PASS on protected splits, V3, HARD-002, confirmation freshness, Platonic/LeFlow plan-only.

Adversarial reviewer P1s addressed in this increment except remote CI (requires push) and IBD-001 temporal-separation (frozen as smoke; IBD-002 is the prospective sequence).
