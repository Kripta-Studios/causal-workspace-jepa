# CRCT-LEARNED-WM-ACTION-DELTA-004 — DRAFT ONLY

```text
STATUS:                 DRAFT_NOT_PREREGISTERED
EXECUTION_AUTHORIZED:   false
PARENT:                 CRCT-LEARNED-WM-ACTION-DELTA-003 MODEL_INCOMPETENT
```

The freeze candidate is `docs/CRCT_LEARNED_WM_ACTION_DELTA_004_PROTOCOL.md`.
This draft is not the freeze. Independent reviews of the repaired candidate
returned FREEZE_ALLOWED.

## Why 003 did not answer the question

003 asked whether a sparse action-embedding coalition is an information
gateway or a target-specific pathway for `ax → Δvx`. Seed 79 failed Δy
NMSE at the frozen 800-step budget. CRCT stayed closed.

Independent reviews of the 003 **protocol** later returned **NO-FREEZE**.
Receipt:
`docs/CRCT_LEARNED_WM_ACTION_DELTA_003_INDEPENDENT_REVIEW_POST_FREEZE_2026-08-27.md`.
Those P0s did not change 003’s competence stop. They must be repaired here
before any freeze. Do not retune 003. Do not convert 002 seed 59 into a pass.

## Question (unchanged)

When CRCT finds a sparse action-embedding coalition, is it an information
gateway or part of the target-specific computation `ax → Δvx`?

Action-stem MSRS are **eligible for CRCT and path diagnostics**. They
**cannot** be a Level-3 pass. That is the 002 seed-59 *pattern* veto, not
a relabel of seed 59.

## Competence

002 showed 800 steps can work. 003 showed it is not seed-universal.
Freeze a finite ladder on **new** seeds, disjoint from 001/002/003.
Stop at the first fully competent rung. No extra rungs after development.

## Level-3 pass machine (P0 repairs)

`PATH_MECHANISM_RECOVERY_PASSED` only if:

- all development seeds competent;
- MSRS sufficient, necessary, inclusion-minimal, specific vs `Δvy` and `Δy`;
- random same-size controls: 0 sufficient (if MSRS is action-only, also
  0 sufficient among same-size subsets of `{act_*}`);
- `G_full >= 0.50`;
- path class is a **split**: `DIRECT` xor `DISTRIBUTED`
  (`G_skip` and `G_res` on opposite sides of 0.50);
- MSRS is **residual-inclusive** (not ⊆ `{act_*}`);
- gauged re-search sufficiency `<= 0.05`, gauged necessity `>= 0.10`, and
  gauged path class equals the original class.

Action-stem MSRS, even with a skip/residual split, is
`INFORMATION_GATEWAY_ONLY`. `REDUNDANT_ROUTES` is Level 2; confirmation
closed. Probe overlap on a residual-inclusive split is recorded.

Not a path-mechanism pass:

| class / condition | status |
|---|---|
| `INTERACTING` (both holds `< 0.50`, `G_full >= 0.50`) | `MEDIATOR_FOUND_PATH_UNRESOLVED` |
| action-stem MSRS (including skip-split / seed-59 pattern) | `INFORMATION_GATEWAY_ONLY` |
| `REDUNDANT_ROUTES` | own label, Level 2; confirmation closed |
| mixed path classes across seeds | experiment `INCONCLUSIVE` |

`H_DISTRIBUTED` only for class `DISTRIBUTED`. `H_EQUIVALENT` only if ≥2
seeds pass with the **same** split class and different literal MSRS.

## Path-hold equations (freeze before code)

```text
h0 = mix(z, e_patched)

G_skip:  hold hid1=hid1_A, hid2=hid2_A
         residual MSRS members unused by construction
         interpret as architecture skip-route from the action-carrier subset

G_res:   hold skip1 = h0_A only
         residual branches recompute from patched mix
         block 2 sees residual-updated h1
```

Do not freeze `skip2=h1_A` while `hid2` still depends on a different `h1`.

## Specificity ontology

Unchanged from 003 physics graph: `Δvx` direct; `Δx` downstream of `ax`
(not a negative control); `Δvy` and `Δy` independent of `ax`.

## Explicit non-actions

No 003 rerun. No 002 seed 59 pass. No JEPA objective, friction, MiniPush,
planning, stitching, Qwen 004, or IBD-002.
