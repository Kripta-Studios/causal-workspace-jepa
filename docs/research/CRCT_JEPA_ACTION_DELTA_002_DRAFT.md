# CRCT-JEPA-ACTION-DELTA-002 — DRAFT ONLY

```text
STATUS:                 DRAFT_NOT_PREREGISTERED
EXECUTION_AUTHORIZED:   false
CONFIRMATION:           CLOSED
PARENT:                 CRCT-JEPA-ACTION-DELTA-001 MODEL_INCOMPETENT
```

This file is **not** a freeze. Do not train, do not confirm, do not download.

`CRCT-JEPA-ACTION-DELTA-001` failed the frozen full-state competence conjunction
at 200 Adam steps: `Δvx`/`Δvy` NMSE met `<= 0.05` on development seeds 43/47/53,
but `Δx`/`Δy` did not. Circuit search was not run. Do not mutate 001.

## Proposed question (future freeze)

Same as 001: can label-blind CRCT recover a causally sufficient/necessary
coalition for learned `ax → Δvx` on a **competent** residual MLP?

## Proposed repair (must be frozen before any 002 outcome)

Keep architecture `H=6`, site ontology, interventions, and primary M1 gates
from 001 unless independent review finds a remaining P0.

Preregister a **finite training ladder** (example, not executed):

```text
rung A: 200 steps   (already failed on 001; do not reuse 001 seeds)
rung B: 800 steps
rung C: 2000 steps
```

Climb only on development competence, fail-closed, no architecture growth.
New model seeds, disjoint from 001/HARD/IBD/WM.

Do not drop `Δx`/`Δy` competence solely to interpret `Δvx`. Full-state
competence remains the WM bar unless a reviewed protocol argues otherwise
*before* outcomes.

## Non-actions

No MiniPush, no friction/contact, no Qwen 004, no stitching, no IBD-002.
