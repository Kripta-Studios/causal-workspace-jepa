# CRCT-LEARNED-WM-ACTION-DELTA-005 protocol (freeze candidate)

```text
STATUS:                 PREREGISTERED_NOT_RUN at freeze (this file is the freeze)
EXECUTION_AUTHORIZED:   true only on the freeze commit until the run is closed
PARENT:                 CRCT-LEARNED-WM-ACTION-DELTA-004 INCONCLUSIVE
```

This file is the freeze. Independent protocol and adversarial reviews of the
repaired candidate (F2 not Level 3; action-stem not Level 3; plants via
`edge_factorial`) returned **FREEZE_ALLOWED**.

Parent 004 remains `INCONCLUSIVE` (evidence `None`; confirmation closed).
Seed 97 remains `INFORMATION_GATEWAY_ONLY` (development-only). Seed 101
`REDUNDANT_ROUTES` is diagnostic, not a pass. Do not mutate 001/002/003/004.
IBD-003 remains synthetic `MECHANISM_RECOVERY_PASSED`. HARD-002 remains
`NEGATIVE_RESULT`. IBD-002 was not executed.

Nomenclature: supervised residual-MLP PointMass predictor. **Not** a JEPA
objective.

## Question

Given a causal mediator `V` for learned `ax → Δvx`, can identifiable
edge interventions recover the computational route `E` that transmits
`V`’s counterfactual effect?

004 localized a development-only Level-2 action-stem mediator and could
not identify exclusive skip vs residual edges. 005 repairs **intervention
identifiability**. It does not primarily search for another node coalition.

## Mechanistic object

```text
M = (V, E, I)
V : Stage A MSRS (node coalition)
E : skip1, F1 (r1), skip2, F2 (r2)
I : cached-message factorial compositions defined below
```

## Finite competence ladder (prospective; historical parents only)

| Source | Steps | Result |
|---|---:|---|
| 001 | 200 | incompetent |
| 002 | 800 | competent on those seeds; 2000 not run |
| 003 | 800 | seed 79 incompetent |
| 004 | 800 | all three incompetent; 2000 competent |

005 uses **new** seeds. 2000 is **not** assumed universal.

| Rung | Adam steps | Role |
|---|---:|---|
| A | 800 | 003/004 first budget |
| B | 2000 | 004’s selected rung, new seeds only |
| C | 5000 | last cap; not historically attested on these seeds |

Optimizer unchanged: Adam `lr=3e-3`, batch 64, pooled MSE, `H=6`.
No extra rungs. No architecture/optimizer/threshold changes after outcomes.

Stop at the first rung where **all** development seeds pass full-state
competence (all four Δ NMSE `<= 0.05` on `S*1000+67`). If C fails:
`MODEL_INCOMPETENT`, CRCT **CLOSED**, confirmation **CLOSED**.

## Seeds (new; disjoint from 001–004)

Forbidden: 004 forbidden set plus 004 seeds `97,101,107,1063,1069,1087`.

- Development: `109, 113, 127`
- Confirmation: `1103, 1109, 1117`

```text
train:    S*1000+61   256 traj × 6 steps
dev:      S*1000+67   64 traj
confirm:  S*1000+71   64 traj  (confirmation stage only)
```

## Specificity (unchanged physics graph)

Direct target: `Δvx`. Downstream of `ax`: `Δx` (not a negative control).
Independent controls: `Δvy`, `Δy`. Ratios `>= 2` vs both independents.

## Stage A — mediator recovery

Label-blind greedy restore + inclusion-minimal prune on `Δvx`
(`max_coalition=4`, `min_step_nmse=0.02`, restore `<= 0.05`). Encoder
excluded. Residual stream `h0`/`h1` are **not** searchable nodes.

Level-2 conjunction (same numeric bars as 004):

- nonempty `V`
- sufficiency `<= 0.05`
- inclusion-minimal
- necessity `>= 0.10`
- specificity vs `Δvy` and `Δy`
- random same-size: 0 sufficient of 32; if `V ⊆ {act_*}`, also 0 of 32
  (or remainder) same-size act-subsets
- `G_V >= 0.50` (mediator patch from B into A; no path holds)

If Stage A fails: stop. Do **not** run Stage B as a status-determining
procedure. Record `path_class=null`.

`G_V` is Level 2. It is not a path class.

## Reference forwards (paired A/B)

Same 002/004 pair generator (`n=64`, same state, same `ay`, `ax` differs).

- **A:** factual `(state, action_A)`
- **B:** counterfactual `(state, action_B)` (used only for `y_B` and to
  supply `V` activations)
- **P:** hybrid `(state, action_A)` with **only** `V` activations patched
  from B (`G_V` forward)

## Residual-message semantics (frozen)

`r1` and `r2` used in Stage B are **cached** from reference forwards A and
P. They are **not** recomputed as `F2(hybrid_h1)`.

```text
r1_A = b1_w2(hid1_A)     # F1 output on A
r1_P = b1_w2(hid1_P)     # F1 output on P
r2_A = b2_w2(hid2_A)     # F2 output on A; F2 saw h1_A
r2_P = b2_w2(hid2_P)     # F2 output on P; F2 saw h1_P
s1_A = h0_A
s1_P = h0_P
h1_A = s1_A + r1_A
h1_P = s1_P + r1_P
```

Claim: `r2_P` is the residual-2 **message from the P run**, not
`F2(current h1)`. Independent review must accept this cached-message
reading.

## Composition (output head)

```text
compose(s1, r1, r2, s2=None):
    h1 = s1 + r1
    skip2 = h1 if s2 is None else s2
    h2 = skip2 + r2
    y  = W_out(h2)
```

Sanity: `compose(s1_A, r1_A, r2_A) = y_A` and
`compose(s1_P, r1_P, r2_P) = y_P`.

## Stage B — edge factorial, conditioned on V

Only if Stage A passed. Question: where does the effect of **this** `V`
go?

Gap-closed (median, `Δvx`, bar `0.50`):

`G(y_hyb) = median 1 - ||y_B - y_hyb||^2 / ||y_B - y_A||^2` on `Δvx`.

### Block 1 2×2 (F2 message held factual)

`s2` defaults to constructed `h1`. `r2 := r2_A`.

| cell | s1 | r1 | name |
|---|---|---|---|
| (A,A) | s1_A | r1_A | reconstruct A |
| (P,A) | s1_P | r1_A | **E_skip** (skip1 CF, residual1 factual) |
| (A,P) | s1_A | r1_P | **E_F1** (skip1 factual, residual1 CF) |
| (P,P) | s1_P | r1_P | both block-1 messages CF; F2 still factual |

`G_skip1 = G(P,A)`, `G_res1 = G(A,P)`, `G_both1 = G(P,P)`.

### Block 2 2×2 (cached h1 and r2 from A/P)

```text
y_skip2 = W_out(h1_P + r2_A)     # skip2 CF stream from P, F2 factual
y_res2  = W_out(h1_A + r2_P)     # skip2 factual, F2 message from P
y_both2 = W_out(h1_P + r2_P)     # equals y_P
```

`G_skip2`, `G_res2`, `G_both2`.

### Combined residual path

`E_F1F2`: skip1 factual, both residual messages CF (cached):

```text
y_f1f2 = compose(s1_A, r1_P, r2_P)   # s2 defaults to s1_A+r1_P
G_f1f2 = G(y_f1f2)
```

### Interaction (block 1, Δvx channel)

Per pair, then median:

```text
I1_y = y(P,P) - y(P,A) - y(A,P) + y(A,A)     # Δvx component
```

If `W_out` is linear and `h2 = s1+r1+r2_A`, `I1_y = 0` identically.
Nonzero `I1_y` is recorded. Classification uses gap-closed exclusive
routes, not a numeric `I1_y` bar (no extra tuned threshold).

## Path class (only if `G_V >= 0.50`)

Let `bar = 0.50`. A route is **independently sufficient** iff its gap
`>= bar`.

| class | rule |
|---|---|
| `DIRECT` | `G_skip1 >= bar` and `G_res1 < bar` and `G_res2 < bar` |
| `DISTRIBUTED_F1` | `G_res1 >= bar` and `G_skip1 < bar` and `G_res2 < bar` |
| `F2_CACHED_UNIDENTIFIED` | `G_res2 >= bar` and `G_skip1 < bar` and `G_res1 < bar` |
| `REDUNDANT_ROUTES` | two or more of `{G_skip1, G_res1, G_res2}` independently sufficient |
| `INTERACTING` | `G_V >= bar` but none of the exclusive classes above |

`G_skip2` is **recorded** and is an alias of `G_both1` (`h1_P + r2_A`). It is
**not** an independent skip2 edge and does **not** assign class.

`DIRECT` is **not** 004 `G_skip`. `DISTRIBUTED_F1` means the F1 **message**
carries `V` and then rides the additive stream. Holding skip2 while swapping
`r1` is a no-op under cached `r2_A`; that hold is therefore **not** used as
an exclusive-F1 test.

`F2_CACHED_UNIDENTIFIED` is **not Level 3**. Cached `r2_P` is the F2
**message from the P forward**. If `V` does not include `b2_*`, that
message is `F2(h1_P)` and therefore can smuggle block-1 content. If `V`
includes `b2_*`, `r2_P` may be a patched `hid2` and is not “F2 saw
`h1_P`”. Either way it is not an isolated F2 computational edge. Record
`G_res2`; do not emit `DISTRIBUTED_F2_PATH_MECHANISM_PASSED`.

`REDUNDANT_ROUTES` is independent sufficiency of two named **messages**.
That is the non-unique-route / non-minimal path outcome. There is no
separate tautological `PATH_MINIMALITY_FAILED` gate.

Combined residual-only transmission that no single message can carry is
`INTERACTING`, not a Level-3 `DISTRIBUTED_F1F2` class.

Action-stem `V ⊆ {act_*}` **cannot be Level 3**, including exclusive
`DIRECT`. Status `INFORMATION_GATEWAY_ONLY`. Residual-inclusive `V` is
required for Level 3. This keeps 004 seed 97’s pattern from counting as
path recovery. Stage B still **runs** as a diagnostic on action-stem `V`.

## Edge necessity / sufficiency / minimality

- Sufficiency: the named message’s `G >= 0.50`.
- Unique-route classes are exclusive (the other two named messages are
  below bar). `REDUNDANT_ROUTES` is the explicit non-unique case.
- Do not emit a separate `PATH_MINIMALITY_FAILED` that restates those
  inequalities.

## Matched edge controls (frozen)

- `control_shuffled_r1`: permute `r1_P` across the batch, compose with
  `s1_A`, `r2_A`. For a unique-route Level-3 claim this must **not** be
  sufficient (`G < 0.50`).
- `g_control_r1_only` / `g_control_r2_only` are **aliases** of `G_res1` /
  `G_res2` and are recorded only. They are not extra controls.

## Gauge

Function-preserving hidden gauge (same 001–004 map). Function MSE
`<= 1e-8`. Re-run Stage A and Stage B. Require gauged restore `<= 0.05`,
gauged necessity `>= 0.10`, and **same path class** (architectural /
functional class, not literal unit IDs). Literal Jaccard recorded, not a
gate.

## Per-seed status (first matching)

1. competence fail → `MODEL_INCOMPETENT`
2. empty `V` → `LOCALIZATION_FAILED`
3. restore `Δvx > 0.05` → `SUFFICIENCY_FAILED`
4. not inclusion-minimal → `MINIMALITY_FAILED`
5. necessity `< 0.10` → `NECESSITY_FAILED`
6. independent specificity fail → `SPECIFICITY_FAILED`
7. random / act-random sufficient `> 0` → `INCONCLUSIVE`
8. `G_V < 0.50` → `INCONCLUSIVE` (no Stage B status)
9. gauge function / gauged mediator suff/nec fail → `GAUGE_FAILED`
10. MSRS ⊆ `{act_*}` → `INFORMATION_GATEWAY_ONLY` (Stage B class diagnostic)
11. Stage B class `INTERACTING` → `MEDIATOR_FOUND_PATH_UNRESOLVED`
12. Stage B `REDUNDANT_ROUTES` → `REDUNDANT_ROUTES` (Level 2)
13. Stage B `F2_CACHED_UNIDENTIFIED` → `MEDIATOR_FOUND_PATH_UNRESOLVED`
14. unique-route claim fails shuffled-`r1` control → `EDGE_CONTROL_FAILED`
15. gauged path class ≠ original → `PATH_CLASS_GAUGE_UNSTABLE`
16. Stage B class in `{DIRECT, DISTRIBUTED_F1}` and residual-inclusive `V`
    → corresponding `*_PATH_MECHANISM_PASSED`
17. Stage A passed, residual-inclusive `V`, no exclusive Level-3 class →
    `MEDIATOR_FOUND_PATH_UNRESOLVED`

## Experiment-level status

- any development seed incompetent → `MODEL_INCOMPETENT`
- all seeds the **same** Level-3 status → `PATH_MECHANISM_RECOVERY_PASSED`
  with that shared class
- all `REDUNDANT_ROUTES` → `REDUNDANT_ROUTES` (`all_seeds_passed=false`;
  evidence `None`; confirmation closed)
- else if all remaining interpreted seeds share one failure status → that
  status
- else `INCONCLUSIVE`

Confirmation opens **only** for development
`PATH_MECHANISM_RECOVERY_PASSED` with shared class in
`{DIRECT, DISTRIBUTED_F1}`. Confirmation
seeds must pass **that same** class. Mixed residual classes are
`INCONCLUSIVE`.

Evidence: experiment-level pass → `Causal effect`. Else `None`.

`functional_convergence`: ≥2 Level-3 passing seeds, same class, literal
Jaccard `< 1`. Not an experiment pass by itself.

## Pre-freeze instrument tests (not scientific evidence)

Deterministic plants must be distinguished **by `edge_factorial`**
(not by hand-written gap dicts) **before freeze**:

DIRECT-only; F1-message; F2-message; genuine skip1/F1 redundancy;
interacting (cancellation). Combined-only additive transmission is
`INTERACTING`. If these plants fail: **do not freeze**.

## 004 hybrid holds

004 `G_skip` / `G_res` may be recorded as diagnostics. They **must not**
assign 005 path class.

## Provenance

CLI refuses unless `execution_authorized: true` and status is not
`DRAFT_NOT_PREREGISTERED`. Do not rewrite 004 `git_dirty` sidecars.

## Explicit non-actions

No 004 mutation/rerun/5000 climb. No seed-97 Level 3. No JEPA, friction,
MiniPush, planning, stitching, Qwen 004, IBD-002, HARD-002.
