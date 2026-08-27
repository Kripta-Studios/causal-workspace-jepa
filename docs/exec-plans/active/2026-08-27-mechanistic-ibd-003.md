# Execution plan — 2026-08-27 mechanistic IBD-003

Continuation of `daf94b6` on `crct-stage0-001`.

## Intent

Primary track is mechanistic interpretability. Planning/Platonic/LeFlow stay paused.
Do **not** execute IBD-002. Do **not** mutate its gates. Advance via a **new** ID.

## Frozen boundaries (unchanged)

- HARD-002 `NEGATIVE_RESULT` seeds 1009/2027/4093 — no rerun
- V3 ineligible; Qwen 004 unauthorized
- T1/T2/MiniPush-002/Reachable-003 statuses unchanged
- IBD-001 smoke; IBD-002 `PREREGISTERED_NOT_RUN`

## Decision from independent review

Protocol [IBD-002 audit](fa8fe000-1a02-4796-ae60-9e863b3f3f87) and adversarial
[IBD-002 review](dc9cd773-f572-4761-88a7-1e51021dd055): **do not execute IBD-002**.

## Work

- [x] Document IBD-002 defects without executing it.
- [x] Preregister `CRCT-COALITION-IBD-003` (interventional recovery).
- [x] Independent protocol review of 003; resolve P0/P1; freeze; then run once.
- [x] Draft `CRCT-JEPA-ACTION-DELTA-001` only if 003’s frozen primary gate passes.
- [x] Do not run JEPA-DELTA, 004, stitching, Reachable-003, RF, DINO-WM, LeWM.
