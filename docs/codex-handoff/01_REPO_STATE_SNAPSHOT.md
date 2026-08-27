# Repository State Snapshot

Snapshot date: 2026-08-20.

Target:
- repository: `Kripta-Studios/causal-workspace-jepa`
- branch: `crct-stage0-001`
- expected remote HEAD: `3f58ec5a25ad8374c4678f94171a18892fdc0ead`
- expected relation to `main`: 7 commits ahead, 0 behind at the audited snapshot

## Audited local state supplied at handoff

The supplied Codex audit reports:

- no tracked modifications;
- 12 untracked entries:
  - eight adjudication/bridge patches;
  - two audit JSON files;
  - one metrics directory;
  - one deployment specification;
- two untracked JSON metrics are caught by the reproducibility audit without required schema fields
  and `.provenance.json` sidecars;
- quick/full-ish test run: `284 passed, 1 failed, 20 warnings`;
- focused Qwen bridge/competence tests: `21 passed`;
- focused CRCT scientific tests: `6 passed`;
- Ruff: five tracked-code issues (four unused imports and one unused variable);
- main documentation is behind the 2026-08-18 competence work.

All of this must be re-verified locally before edits.

## Scientific state

### CRCT-STAGE0-001

Retained as a synthetic positive control. It demonstrated planted sparse mechanism recovery but its
original random-specificity statistic had selection-on-evaluation contamination. Do not upgrade it
to real-model evidence.

### CRCT-STAGE0-HARD-002

Frozen scientific result: `NEGATIVE_RESULT`.

Key observed pattern:
- residual power after T2: `0.0547`, `0.1290`, `0.0420`;
- IID/OOD functional recovery > `0.992`;
- planted node recall: `0.8`, `0.4`, `0.6`;
- QK-like edge recovery: perfect;
- matched control plus-one p-value: `1/257`;
- direct-delta MLP beat residual MLP on all primary seeds IID and OOD.

Interpretation: literal graph recall and functional causal sufficiency diverge under redundancy and
cancellation. The negative result must remain frozen.

### Qwen Binding V3 B0

Frozen outcome: `INELIGIBLE_TASK_PHASE0`.

Instrumentation/replay integrity was strong, but full-vocabulary behavioral competence under the V3
prompt was essentially zero. This is a task/prompt ineligibility result, not evidence that CRCT
fails in Qwen.

### Qwen competence recovery

The supplied local audit reports an already-executed calibration-only development run selecting:

- `qwen_chat_prefill_v1`
- clean full-vocabulary accuracy `0.9375`
- direct-permuted full-vocabulary accuracy `0.9792`

This is **not confirmation**. Verify the exact local artifact before adjudicating it.

## Current critical path

1. hygiene/provenance/Ruff/tests/CI;
2. freeze competence-recovery development result if artifact verifies;
3. preregister and execute a genuinely fresh competence confirmation;
4. only after competence confirmation passes, reopen a new mechanistic Qwen experiment;
5. build coalition/equivalence-aware CRCT successor with fresh synthetic confirmation seeds;
6. keep LeVLJEPA as a secondary representation track;
7. keep Steerling-8B execution resource-blocked unless hardware changes.
