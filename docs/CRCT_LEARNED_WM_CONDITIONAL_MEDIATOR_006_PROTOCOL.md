# CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006 protocol (freeze candidate)

```text
STATUS:                 PREREGISTERED_NOT_RUN at freeze (this file is the freeze)
EXECUTION_AUTHORIZED:   true only on the freeze commit until the run is closed
PARENT:                 CRCT-LEARNED-WM-ACTION-DELTA-005 INCONCLUSIVE
PRIMARY TARGET:         Level 2B conditional downstream mediation
LEVEL 3:                not authorized by this protocol
```

This file is the freeze. Independent protocol and adversarial reviews
returned **FREEZE_ALLOWED** after repair of F2/`V_up` `b2_*` recompute
and adjudication-order alignment. No learned model is trained before
this commit.

Parent 005 remains `INCONCLUSIVE` (evidence `None`; confirmation closed;
selected rung 800; Stage A failed; Stage B never status-determining).
Do not mutate or rerun 001–005. Do not reinterpret 005 diagnostic path
scores. Seed 97 is not Level 3. IBD-003 remains synthetic
`MECHANISM_RECOVERY_PASSED`. HARD-002 remains `NEGATIVE_RESULT`.
IBD-002 was not executed.

Nomenclature: supervised residual-MLP PointMass predictor. **Not** a JEPA
objective.

## Question

After a valid upstream causal mediator `V_up` for learned `ax → Δvx` has
been identified in a competent residual world model, can conditional
interventions identify downstream **branch messages** `V_down` that are
necessary and sufficient for transmitting that mediator’s counterfactual
effect?

This is **not** another global MSRS experiment. Global restoration asks
where the effect can be reintroduced. Stage 2B asks what downstream
computation is required to transmit it.

## Methodological distinction (frozen; plants only)

1. **Recompute / `G_V` mediation** has an early-carrier property. Patching
   `V_up` from B into A and recomputing descendants leaves F1/F2 intact, so
   an early carrier can yield high `G_V` even when later residual maps do
   the readout computation.
2. **Mean-fill MSRS** is a different object. Mean-filling non-coalition
   hidden units disables downstream machinery. Do **not** claim that
   mean-fill MSRS has the same early-bottleneck bias.

Validated on planted instruments. Not a 005 reinterpretation.

## Mechanistic object

```text
M = (C, V_up, V_down, E, I)

L1  C      : information carrier (diagnostic; not a circuit claim)
L2A V_up   : upstream node coalition (Stage A)
L2B V_down : conditional downstream branch-message coalition ⊆ {r1, r2}
L3  E      : identifiable path/edge set (NOT a 006 pass criterion)
I          : intervention family below
```

Do not collapse these into one coalition. Do not require `b1_*` or `b2_*`
membership by fiat. Action-stem `V_up` is **allowed** at Level 2A/2B.

006 may establish Level 2B without establishing Level 3. `E` is recorded
only as a diagnostic label of the 2B class, not as an independently
identified edge factorization of the 005 kind.

## Residual architecture (frozen)

```text
s1 = h0
r1 = F1(h0)
h1 = s1 + r1
s2 = h1
r2 = F2(h1)
h2 = s2 + r2
y  = W_out(h2)
```

On the learned 001–005 topology, `F1` and `F2` are the residual MLP maps
(`b1_w2(tanh(b1_w1(h0)) @ q_b1)` and `b2_w2(tanh(b2_w1(h1)) @ q_b2)`).

Primary `V_down` candidates: `{r1, r2}`. Skip messages `s1`/`s2` are not
searchable Stage-2B units; `s1` is the carrier stream of `h0` after the
`V_up` patch.

## Finite competence ladder

Historical facts only (not this experiment’s outcome):

| Source | Steps | Result |
|---|---:|---|
| 001 | 200 | incompetent |
| 002 | 800 | competent on those seeds |
| 003 | 800 | seed 79 incompetent |
| 004 | 800 incompetent; 2000 competent |
| 005 | 800 competent on **those** new seeds |

800 is not assumed universal. 005 competence does not transfer to 006 seeds.

| Rung | Adam steps | Role |
|---|---:|---|
| A | 800 | first budget (003/005) |
| B | 2000 | 004 selected rung, new seeds only |
| C | 5000 | last cap |

Optimizer unchanged: Adam `lr=3e-3`, batch 64, pooled MSE, `H=6`.
No extra rungs. No architecture growth. No new rung after outcomes.

Stop at the first rung where **all** development seeds pass full-state
competence (all four Δ NMSE `<= 0.05` on `S*1000+67`). If C fails:
`MODEL_INCOMPETENT`, Stage A/2B **CLOSED**, confirmation **CLOSED**.

## Seeds (new; disjoint from 001–005)

Forbidden: 005 forbidden set plus 005 seeds `109,113,127,1103,1109,1117`.

- Development: `173, 179, 181`
- Confirmation: `1171, 1181, 1187`

```text
train:    S*1000+61   256 traj × 6 steps
dev:      S*1000+67   64 traj
confirm:  S*1000+71   64 traj  (confirmation stage only)
```

## Specificity (PointMass graph; unchanged)

Direct target: `Δvx`. Downstream of `ax`: `Δx` (not a negative control).
Independent controls: `Δvy`, `Δy`. Stage A ratios `>= 2` vs both
independents.

Stage 2B does not use `Δx` as a negative control. If an independent
channel’s counterfactual effect `median((y_B-y_A)^2) < 1e-8`, that
channel is inactive and does not fail Stage 2B specificity.

## Stage A — `V_up` (node-level; mean-fill restore)

Only on competent models. Label-blind greedy restore + inclusion-minimal
prune on `Δvx` (`max_coalition=4`, `min_step_nmse=0.02`, restore
`<= 0.05`). Encoder excluded. Residual stream `h0`/`h1` are not
searchable nodes.

Conjunction (same numeric bars as 004/005):

- nonempty `V_up`
- sufficiency `<= 0.05`
- inclusion-minimal
- necessity `>= 0.10`
- specificity vs `Δvy` and `Δy`
- random same-size: 0 sufficient of 32; if `V_up ⊆ {act_*}`, also 0 of 32
  (or remainder) same-size act-subsets
- `G_V >= 0.50` (patch `V_up` from B into A; descendants recompute; no
  path holds)

If Stage A fails: stop. Do **not** run Stage 2B as status-determining.
Record `downstream_class=null`, `stage_2b_ran=false`. Diagnostic
`N_down`/`S_down` must not rescue the seed.

Mean-fill restore that **proposes** `V_up` is not the same object as
`G_V`. Both are required. Residual-unit membership is not required.

## Reference forwards

Same pair generator as 002–005 (`n=64`, same state, same `ay`, `ax`
differs).

- **A:** factual `(state, action_A)`
- **B:** counterfactual `(state, action_B)` (for `y_B` and `V_up` activations)
- **P:** hybrid `(state, action_A)` with **only** `V_up` patched from B

```text
m_up = G_V = G(y_A, y_B, y_P; Δvx)
G = median( 1 - ||y_B - y||^2 / max(||y_B - y_A||^2, 1e-12) )
```

Cached messages from A and P: `h0_A/P`, `r1_A/P`, `r2_A/P`.
`r1_P` is the residual-1 message **on the P forward** (includes any `b1_*`
overrides in `V_up`). It is not required to equal `F1(h0_P)`.

## Stage 2B — conditional branch messages (only if Stage A passes)

Keep `V_up` patched (`h0 = h0_P`). Downstream interventions use:

```text
compose(h0, r1, r2) = W_out(h0 + r1 + r2)   # s2 := h1 = h0+r1

F2(h1 | V_up) =
    hid2 = tanh(b2_w1(h1)) @ q_b2
    hid2 := P's b2_* activations for every b2 unit in V_up
    r2   = b2_w2(hid2)
```

Descendant recompute **must not** drop `b2_*` members of `V_up`. Holding
the `r2` **message** at A still overrides that patch (`do(r2=r2_A)`).

### Necessity (Pearl hold; descendants of held r1 recompute)

```text
do(r1 = r1_A):  r1 = r1_A; r2 = F2(h0_P + r1_A | V_up)
do(r2 = r2_A):  r1 = r1_P; r2 = r2_A
do(both):       r1 = r1_A; r2 = r2_A

N_down(D) = 1 - G(hold D | V_up) / G_V
NaN if G_V <= 1e-8
```

Values outside `[0,1]` are allowed (anti-mediation). Do not clip.

Cached `r2_P` while holding `r1` is **recorded** (`g_hold_r1_cached_r2`)
and **does not assign class**. That is the 005 leakage mode.

### Sufficiency (factual-held damaged reference)

```text
damaged: h0 = h0_P, r1 = r1_A, r2 = r2_A
```

On-manifold A messages. Not mean-fill. Not an invented off-manifold
baseline.

Restore D:

```text
r1 in D: r1 := r1_P          # cached P message
r2 in D: r2 := F2(h0_P + r1 | V_up)  # recompute hid2, re-apply V_up b2_* from P
else:    r2 := r2_A          # unless descendant-recompute operator below

S_down(D) = (G_restore - G_damaged) / (G_V - G_damaged)
NaN if (G_V - G_damaged) <= 1e-8
```

`S_down^desc(r1)`: restore `r1_P` **and** recompute `r2 = F2(h0_P+r1_P)`.
Used only to separate sequential F1→F2 from additive interaction.

### Classification (`G_V >= 0.50`; else class is null)

Bars: `N_down >= 0.50`, `S_down >= 0.50`, `|I| >= 0.20`.
`I = median( y_PP - y_P,r2A - y_r1A,F2(hyb) + y_AA )` on `Δvx`.
`interaction_abs_min = 0.20` is frozen from planted sequential (`I≈0.7`)
vs additive interacting (`I≈0`) **before** learned outcomes.

A finite `S_down` at/above bar is sufficient. `NaN` is not sufficient.

| class | rule |
|---|---|
| `DIRECT` | not N(r1), not N(r2), `G_damaged >= 0.50` |
| `DOWNSTREAM_F1` | N(r1), not N(r2), S_hold(r1) |
| `DOWNSTREAM_F2` | N(r2), not N(r1), S_hold(r2) |
| `REDUNDANT_DOWNSTREAM` | not N(r1), not N(r2), S_hold(r1) and S_hold(r2) |
| `DOWNSTREAM_F1_F2` | N both, S_both, S_desc(r1), `|I| >= 0.20` |
| `INTERACTING_DOWNSTREAM` | N both, S_both, not S_hold(r1), not S_hold(r2), `|I| < 0.20` |
| `DOWNSTREAM_UNRESOLVED` | Stage 2B ran, none of the above |

`REDUNDANT_DOWNSTREAM` requires **independent** downstream sufficiency
(each singleton S_hold). `INTERACTING_DOWNSTREAM` requires that neither
singleton meets S_hold despite the joint restore.

Do not classify from score appearance without these conjunctions.

### Matched branch control

For unique-route `DOWNSTREAM_F1`: permute `r1_P` across the batch, restore
that shuffled message with `r2_A`. `S_down` of the shuffle must **not**
pass (`< 0.50` or `NaN`). Failure → `BRANCH_CONTROL_FAILED`.

`DOWNSTREAM_F2` uses shuffled `r2` analogously only if r2 restore is the
cached-P operator; under recomputed `F2(h1)` a batch permutation of `r2_P`
is not the restore operator, so the F1 shuffle is the frozen unique-route
control. F2 unique-route does not use shuffled `r1` as a pass gate.

## Gauge

Function-preserving hidden gauge (same 001–005 map). Function MSE
`<= 1e-8`. Re-run Stage A and Stage 2B. Require gauged restore `<= 0.05`,
gauged necessity `>= 0.10`, and **same downstream class** (role, not
literal unit IDs). Literal Jaccard recorded, not a gate.

Planted readout-coordinate rotation must keep `DOWNSTREAM_F1`. That is
instrument validation only. Do not claim learned-model gauge invariance
before confirmation.

## Per-seed status (first matching)

1. competence fail → `MODEL_INCOMPETENT`
2. empty `V_up` → `LOCALIZATION_FAILED`
3. restore `Δvx > 0.05` → `SUFFICIENCY_FAILED`
4. not inclusion-minimal → `MINIMALITY_FAILED`
5. necessity `< 0.10` → `NECESSITY_FAILED`
6. independent specificity fail → `SPECIFICITY_FAILED`
7. random / act-random sufficient `> 0` → `INCONCLUSIVE`
8. `G_V < 0.50` → `INCONCLUSIVE` (Stage 2B not opened)
9. gauge function / gauged mediator suff/nec fail → `GAUGE_FAILED`
10. Stage 2B not run after Stage A pass → `INCONCLUSIVE`
11. gauged class ≠ original → `DOWNSTREAM_CLASS_GAUGE_UNSTABLE`
12. `DOWNSTREAM_UNRESOLVED` → `DOWNSTREAM_UNRESOLVED` (Level 2A only)
13. `REDUNDANT_DOWNSTREAM` → `REDUNDANT_DOWNSTREAM` (Level 2A; not a pass)
14. `INTERACTING_DOWNSTREAM` → `INTERACTING_DOWNSTREAM` (Level 2A; not a pass)
15. unique F1 class fails shuffled-`r1` control → `BRANCH_CONTROL_FAILED`
16. class in `{DIRECT, DOWNSTREAM_F1, DOWNSTREAM_F2, DOWNSTREAM_F1_F2}`
    → corresponding `*_MEDIATION_PASSED` / `DIRECT_TRANSMISSION_PASSED`
    at hierarchy Level **2B** (not Level 3)

Action-stem `V_up` does **not** demote a 2B pass to a gateway. That 005
rule was a Level-3 path-identifiability guard, not a 2B mediation guard.

## Experiment-level status

- any development seed incompetent → `MODEL_INCOMPETENT`
- all seeds the **same** 2B pass status →
  `CONDITIONAL_DOWNSTREAM_MEDIATION_PASSED` with that shared class
- all `REDUNDANT_DOWNSTREAM` → `REDUNDANT_DOWNSTREAM`
  (`all_seeds_passed=false`; evidence `None`; confirmation closed)
- all `INTERACTING_DOWNSTREAM` → `INTERACTING_DOWNSTREAM` (same)
- else if all remaining interpreted seeds share one failure status → that
  status
- else `INCONCLUSIVE`

Confirmation opens **only** for development
`CONDITIONAL_DOWNSTREAM_MEDIATION_PASSED` with shared class in
`{DIRECT, DOWNSTREAM_F1, DOWNSTREAM_F2, DOWNSTREAM_F1_F2}`.
Confirmation seeds must pass **that same** class. Mixed 2B classes are
`INCONCLUSIVE`.

Evidence: experiment-level pass → `Causal effect`. Else `None`.

`functional_convergence`: ≥2 2B-passing seeds, same class, literal `V_up`
Jaccard `< 1`. Report **downstream-role convergence**, not neuron-level
convergence. Not an experiment pass by itself.

## Strongest supportable positive claim (if the frozen rule passes)

In competent independently trained tiny supervised residual world models,
CRCT identified valid upstream mediators for `ax → Δvx` and conditional
branch-message interventions identified a shared downstream computational
role transmitting their counterfactual effect.

If the shared class is `DIRECT`, the claim is skip-stream transmission of
`V_up`, **not** identification of F1/F2 residual computation.

Not claimed: generic mechanistic universality; Level-3 exclusive edges;
JEPA; learned gauge-invariant units; 005 path recovery.

## Pre-freeze instrument tests (not scientific evidence)

Real executable plant forwards, not dict-only outputs, must distinguish
before freeze:

DIRECT; DOWNSTREAM_F1; DOWNSTREAM_F2; DOWNSTREAM_F1_F2;
REDUNDANT_DOWNSTREAM; INTERACTING_DOWNSTREAM.

**KEY TEST (hard gate):** early-carrier / F1 plant:

- recompute `V_up={act_0}`: `G_V ~ 1`
- mean-fill act-only: fails
- `N_down(r1)` high, `N_down(r2)` low
- class `DOWNSTREAM_F1`

If KEY TEST fails: **NO FREEZE**.

## Provenance

CLI refuses unless `execution_authorized: true` and status is not
`DRAFT_NOT_PREREGISTERED`. Do not rewrite 005 `git_dirty` sidecars.

## Explicit non-actions

No 005 mutation/rerun. No residual-membership-by-fiat. No conflating
mean-fill MSRS with recompute `G_V`. No skipping Stage A. No extra
ladder rungs after outcomes. No JEPA, friction, MiniPush, planning,
stitching, Qwen 004, IBD-002, HARD-002.
