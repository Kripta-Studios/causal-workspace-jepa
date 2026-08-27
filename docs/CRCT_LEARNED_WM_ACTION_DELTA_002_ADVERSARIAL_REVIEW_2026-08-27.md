# Adversarial review — CRCT-LEARNED-WM-ACTION-DELTA-002 (pre-freeze)

## Verdict

**Freeze allowed.** Attacks that survive are fail-closed outcomes, not silent
passes.

## Attack: 001 already proved 200 fails, so skip rung A

Rejected. Rung A on **new** seeds is the frozen control that the 001 budget
fails out of sample. Climbing starts at 200.

## Attack: drop Δx/Δy because M1 is Δvx

Rejected. Frozen 001 conjunction retained. Velocity-only interpretation is
forbidden.

## Attack: change the loss to per-channel NMSE

That would be an optimizer/loss change after seeing 001. Not authorized.
More Adam steps is the declared repair for pooled-MSE underweighting of Δx.

## Attack: JEPA naming

ID and claim_boundary forbid calling this a JEPA.

## Attack: interpret incompetent confirmation models

`MODEL_INCOMPETENT_CONFIRMATION`; CRCT skipped when competence fails.

## Attack: trivial action-layer circuit

Unchanged: max size 4; full `act_*` is inconclusive.

## Attack: fused provenance / extra rungs

Per-rung CLI; climbing refused unless previous is `MODEL_INCOMPETENT`.
No post-hoc rungs.

## Must not claim even on pass

JEPA mechanisms; Qwen; workspace; Platonic; planning; MiniPush; 001 rescue;
HARD-002 rescue.
