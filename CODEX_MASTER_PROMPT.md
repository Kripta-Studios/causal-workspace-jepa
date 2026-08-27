# Master prompt for Codex

You are operating on `Kripta-Studios/causal-workspace-jepa`, branch
`crct-stage0-001`.

Your job is to **repair the current repository completely, preserve all frozen scientific
adjudications, close the Qwen competence path correctly, implement the next coalition-aware CRCT
research stage, add durable agent/CI guardrails, synchronize documentation, and leave the branch in
a reproducible review-ready state**.

## Mandatory startup sequence

1. Read the repository-root `AGENTS.md`.
2. Read every file in `docs/codex-handoff/`, beginning with `00_INDEX.md`.
3. Read `.codex/skills/cwj-orchestrate/SKILL.md` and use `$cwj-orchestrate`.
4. Use `$cwj-scientific-integrity` for every experiment/protocol decision.
5. Inspect the actual local Git state. Do not assume the handoff snapshot is still exact.
6. Verify branch, HEAD, origin synchronization, tracked changes, untracked files, installed
   dependencies, CUDA/runtime state, pytest, Ruff, and reproducibility audit before editing.
7. Perform the model-routing smoke described in
   `docs/codex-handoff/05_MODEL_AND_SUBAGENT_ROUTING.md`. If model/effort overrides cannot be
   verified, keep the root session on Sol High and do not pretend Luna/Sol routing occurred.
8. Create/update an execution plan under `docs/exec-plans/active/` (or the repo's canonical plan
   location if one already exists) with checkboxes, decisions, evidence, commands, and artifact
   hashes. Then execute it; do not stop after planning.

## Agent orchestration

Use subagents aggressively for **independent read-only analysis**, but avoid concurrent writes to the
same checkout.

Preferred roles:

- `sol_architect` — Sol High, read-only: architecture, scientific protocol, leakage review.
- `sol_verifier` — Sol Medium, read-only: repository survey, test triage, docs/code consistency.
- `luna_worker` — Luna Max, workspace-write: bounded implementation tasks with explicit file scope.
- `luna_test_worker` — Luna Max, workspace-write: tests, CI, lint/provenance fixes.
- `sol_reviewer` — Sol High, read-only: final adversarial review.

If the current Codex build ignores role-specific model or reasoning settings, use the safest
fallback: keep the parent on Sol High and continue, or use separate Codex worktree tasks with the
desired model selected manually. Never report a routing configuration as verified if it was not
observable.

## Existing skills to use when available

- `superpowers`: planning, decomposition, test-first execution, debugging.
- `sol-advisor`: model/effort routing advice; this handoff's scientific safety rules take precedence.
- `gh-fix-ci`: diagnose/fix GitHub Actions after CI exists.
- `gh-address-comments`: address PR review comments later.
- `stop-slop`: final code/docs cleanup only after correctness is established.
- Do not use `frontend-design`; it is irrelevant to this task.

## MCP/tool priority

1. GitHub — repository history, branches, PRs, Actions, upstream evidence.
2. Context7 — current APIs for Python dependencies such as pytest/Ruff/PyTorch/Transformers/h5py.
3. Hugging Face — exact Qwen model/tokenizer revisions and model-card/revision evidence.
4. DeepWiki — inspect upstream implementations/repositories when code-level understanding is needed.
5. Exa — primary literature and current research verification.
6. W&B — only for newly preregistered *development* experiments if explicitly incorporated before
   execution; do not retrofit it into an already frozen confirmation protocol.
7. OpenAI Developer Docs — only for Codex/orchestration/config questions.
8. Sentry — not needed unless a real runtime/service failure makes it relevant.

Prefer primary sources and exact upstream revisions. Do not use MCP access merely because it exists.

## Phase A — establish a clean baseline

Treat the supplied audit as a hypothesis to verify locally. It reported:

- 284 passed, 1 failed, 20 warnings in the quick/full-ish suite;
- 21 focused Qwen bridge/competence tests passed;
- 6 focused CRCT scientific tests passed;
- 5 Ruff issues;
- 12 untracked entries;
- reproducibility audit failure caused by two local metric JSON files with missing required metadata
  and provenance sidecars;
- global docs lagging behind the latest 2026-08-18 execution.

Re-run the relevant checks and record exact current results.

### Untracked-file policy

Classify every untracked file before touching it:

A. legitimate scientific source/protocol/adjudication that should be committed;
B. legitimate generated result that belongs in the repo and has real, reconstructible provenance;
C. local scratch/transient output that should be ignored or moved out of audited artifact paths;
D. suspicious/ambiguous evidence that must remain untouched until its provenance is established.

Never invent `experiment_id`, `status`, `evidence_level`, hashes, timestamps, or provenance just to
make an audit green. If metadata can be reconstructed from actual run logs/config/source hashes,
document the derivation. Otherwise move/ignore the scratch artifact rather than falsifying provenance.

### Baseline fixes required

- Make the complete relevant test suite green.
- Make Ruff green.
- Repair the reproducibility audit honestly.
- Add/repair GitHub Actions CI for at least:
  - unit/scientific tests that do not require protected data or GPU;
  - Ruff;
  - reproducibility/static artifact audit;
  - a targeted competence-entrypoint syntax regression test.
- Ensure the competence recovery runner itself includes
  `tests/unit/test_qwen_competence_recovery_entrypoints.py` in its preflight tests and source snapshot
  if that remains semantically correct after inspection.
- Keep GPU/protected experiments out of ordinary CI.

## Phase B — freeze the already-observed competence recovery result

The handoff states that the already-opened calibration-only run selected:

- renderer: `qwen_chat_prefill_v1`
- clean full-vocabulary accuracy: `0.9375`
- direct-permuted full-vocabulary accuracy: `0.9792`

First verify the local report, manifest, hashes, ledger, config, source snapshot, and exact prompt
renderer. Do not rely on this prompt as evidence.

If and only if the artifact verifies:

1. Write a formal adjudication for `QWEN-BINDING-COMPETENCE-RECOVERY-001`.
2. Freeze the exact renderer/token contract/source hashes.
3. State explicitly that this is calibration-only development selection and **does not rescue V3**.
4. Preserve `LLM-QWEN-BINDING-ALGEBRA-003` / V3 as `INELIGIBLE_TASK_PHASE0`.

If the artifact is absent or inconsistent, classify the discrepancy and do not manufacture a
positive adjudication.

## Phase C — preregister a fresh competence confirmation

Create a new experiment/milestone; do not mutate V3.

Recommended semantic identity (adapt names to existing registry conventions, but do not overload old
IDs):

- `QWEN-BINDING-COMPETENCE-CONFIRM-001`
- successor binding experiment such as `LLM-QWEN-BINDING-ALGEBRA-004`
- successor bridge such as `CRCT-QWEN-BRIDGE-003`

Requirements:

- exact selected renderer frozen before confirmation data generation;
- exact model/tokenizer revision frozen;
- new deterministic confirmation seed/manifest committed before model forwards;
- fresh examples that have never been used for prompt selection;
- no test/paraphrase access;
- clean and direct-permuted **full-vocabulary** accuracy gates remain `>= 0.90`;
- candidate-only accuracy remains diagnostic, never the eligibility metric;
- forward-access ledger is mandatory;
- failure closes this prompt/task path negative without threshold changes.

Commit the preregistration before executing the confirmation. Verify the commit is pushed/contained
by origin if the repo's protocol requires pushed provenance. Then execute only what the frozen
protocol authorizes.

If confirmation passes, it authorizes only the next explicitly preregistered mechanistic phase. It
does not retroactively change V3.

## Phase D — coalition-aware CRCT extension

Do not rerun HARD-002 on its primary seeds and do not lower its gates.

Implement a successor development/confirmation design inspired by the redundancy problem exposed by
HARD-002 and by interpretable-by-design decomposition work.

At minimum, distinguish and test:

1. literal planted-graph recall;
2. epsilon-functional sufficiency/completeness;
3. minimal sufficient sets;
4. individual and group necessity;
5. redundancy-group coverage;
6. cancellation-group coverage;
7. signed contributions, not only absolute attribution;
8. equivalent/functionally substitutable circuit classes;
9. stability under gauge/basis transformations;
10. specificity against frozen matched controls;
11. IID and OOD generalization;
12. intervention-support validity: operation, site, magnitude, and combination must be labeled
    in-support/out-of-support relative to training/development support.

### Required positive control

Add a tiny interpretable-by-design transformer / concept-bottleneck synthetic system with planted:

- known components;
- unknown components;
- residual components;
- redundant paths;
- cancelling paths;
- at least two functionally equivalent minimal circuits.

CRCT must be able to distinguish “recover every planted node” from “recover a minimal causally
sufficient equivalence class”.

Freeze metric/selector design on development plants first, then use fresh untouched primary seeds for
confirmation. Never use HARD-002 primary outcomes to tune the successor threshold.

## Phase E — Intervention-JEPA target policy

Do not assume that `T2 + learned residual` is privileged.

The HARD-002 evidence says the equal-capacity direct-delta MLP beat the learned-residual MLP on every
primary seed IID and OOD. Therefore every future learned residual claim must include fair baselines:

- direct-delta predictor;
- first-order/JVP;
- T2/quadratic;
- relinearized JVP where applicable;
- differential + learned residual;
- simple capacity-matched MLP baseline.

A learned residual branch becomes scientifically eligible only when a real model exhibits a
substantial, stable residual beyond strong differential baselines and the residual is predictably
better modeled than direct delta under frozen evaluation.

You may implement a decomposition target (`known`, `unknown`, `residual`) as a development feature,
but add leakage controls, signed reconstruction, support labels, and direct original-model replay.

## Phase F — secondary LeVLJEPA track

Do not make this the critical path.

After Qwen confirmation and repository hygiene are closed, preregister a small MiniPush factorial:

- current encoder vs frozen LeVLJEPA encoder;
- with/without SIGReg where technically meaningful;
- identical action-conditioned predictor and data budget;
- effective rank;
- object localization/probes;
- causal patching;
- specificity;
- planning/closed-loop effect.

Treat it as a representation/control experiment, not evidence of a causal workspace by itself.

## Phase G — documentation and agent harness

Synchronize at least:

- `README.md`
- `SUMMARY.md`
- `docs/RESULTS.md`
- `docs/ROADMAP.md`
- `docs/TODO.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/DECISIONS.md`
- `docs/REPRODUCIBILITY.md`
- any Qwen/CRCT track-specific docs affected by the new milestones.

Do not delete historical negative results.

Refactor the current large `AGENTS.md` only after auditing its contents. Use
`AGENTS_CANDIDATE.md` as a structural target: short map at root, detailed versioned rules in docs.
Preserve all binding project-specific instructions.

Add mechanical checks for stale or broken documentation links where practical.

## Required review loop

Before finalizing:

1. Have a Sol High reviewer inspect the full diff read-only for:
   - data leakage;
   - look-ahead/outcome leakage;
   - protected split access;
   - post-hoc tuning;
   - provenance fabrication;
   - incorrect baseline comparisons;
   - stale docs;
   - tests that assert implementation rather than scientific semantics;
   - missing failure-path tests.
2. Have a separate verifier inspect only protocol boundaries and result claims, with minimal exposure
   to the implementer's rationale.
3. Resolve all P0/P1 findings.
4. Run full non-protected tests, Ruff, reproducibility audit, and relevant scientific tests again.
5. Record exact commands and results.
6. Inspect `git diff`, `git status`, and untracked files.
7. Commit in coherent milestones; do not squash distinct preregistration and post-outcome adjudication
   into one commit if doing so would destroy temporal evidence.
8. Push only when the current protocol requires remote provenance or when the user has already
   authorized normal repository publishing in this session. Do not merge to `main` without explicit
   authorization.

## Final response required

Return:

- exact final branch and HEAD;
- commits created, one line each;
- tests/lint/audits with counts;
- untracked files remaining and why;
- experiments executed vs only preregistered;
- every protected split touched (expected: none unless a newly frozen protocol explicitly authorizes
  a fresh non-protected confirmation split);
- Qwen competence confirmation outcome;
- scientific status of HARD-002 (must remain negative);
- status of coalition-aware successor;
- remaining blockers;
- exact next command/task if work remains.

Do not claim completion merely because code compiles. Completion means the repository, evidence,
tests, provenance, CI, and documentation agree.
