# CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006 — instrument validation (2026-08-27)

```text
STATUS:                 instrument / methodology only (not a learned-model result)
005 STATUS:             INCONCLUSIVE, evidence None, confirmation CLOSED (unchanged)
006 LEARNED STATUS:     INCONCLUSIVE; Stage A failed; Stage 2B not
                        status-determining; confirmation CLOSED
```

This file is **not** a learned-model result and **not** a freeze.
It does not reinterpret 005 diagnostic path scores.

## Methodological question

Does global restoration sufficiency find the computation, or only the
earliest place from which intact downstream machinery can reconstruct it?

## Two restoration semantics

On `h1 = h0 + F1(h0)`, `h2 = h1 + F2(h1)`:

1. **Mean-fill MSRS** (current Stage A greedy restore): non-coalition hidden
   units are replaced by their mean. Downstream F1/F2 are **disabled** unless
   they are in the coalition. This object is **not** biased toward early
   bottlenecks. It is biased toward units whose ablation breaks the function.

2. **Recompute / G_full** (hybrid patch of V into A, descendants recompute):
   downstream F1/F2 remain **intact**. Restoring an early carrier is then
   sufficient whenever those maps can still transform it. This object **is**
   biased toward early information bottlenecks.

005 Stage A used both: greedy mean-fill to propose a coalition, then G_full
as mediation. Action-stem G_full can look like a complete mediator even when
the computation that maps the carrier onto Δvx is later residual work.

The hypothesis therefore applies to **recompute mediation**, not to mean-fill
search. It is not licensed from 005 seeds; the plants below isolate it.

## Definitions (draft; not a learned freeze)

**Level 1 carrier C.** A representation that contains the counterfactual
information. Not automatically the computation.

**Level 2A upstream mediator V_up.** A node coalition whose B-patch into A,
with descendants recomputed, produces mediated target effect `m_up = G_V`
on `ax → Δvx`.

**Level 2B conditional downstream mediator V_down.** A **branch-message**
coalition `D ⊆ {r1, r2}` that is necessary and sufficient for transmitting
`m_up` once the V_up patch is held fixed.

**Level 3 identifiable edges E.** Only after V_up and V_down are validated.

Do **not** define success as “coalition must contain `b1_*` or `b2_*`”.

## Intervention equations

Paired runs: A (`ax = a`), B (`ax = a'`). Patch `V_up_B` into A → trajectory P.

```text
m_up(ch) = G(y_A, y_B, y_P; ch)
G = median( 1 - ||y_B - y||^2 / max(||y_B - y_A||^2, 1e-12) )
```

Stage 2B runs only if `m_up(Δvx) >= 0.50`.

Pearl hold of D (necessity), descendants of D recompute unless also held:

```text
do(r1 = r1_A) | V_up patch:
  h0 = h0_P
  r1 = r1_A
  r2 = F2(h0 + r1)
  y  = W(h0 + r1 + r2)

N_down(D) = 1 - G(y_{P, D←A}) / G_V
undefined if G_V <= 1e-8
```

`N_down` may exceed 1 when a hold is anti-mediating (`G` negative). Compare
against the bar; do not require `N_down <= 1`.

Factual-held sufficiency reference (on-manifold A messages, not mean-fill,
not an invented off-manifold baseline):

```text
damaged: r1 = r1_A, r2 = r2_A, h0 = h0_P
S_down(D) = (G_restore(D) - G_damaged) / (G_V - G_damaged)
undefined if G_V - G_damaged <= 1e-8
```

Restore/enable D by recomputing those messages from current parents. Other
residuals stay at A unless descendant-recompute is the named operator
`S_down^desc`.

Independent specificity controls remain `Δvy`, `Δy`. `Δx` is downstream of
`ax` and is not a negative control. If an independent channel’s
counterfactual effect `median((y_B-y_A)^2)` is below `1e-8`, it is inactive
and is not used to fail specificity (`G` on a zero channel is not
identifiable).

## Ontology

**A:** V_down = downstream node coalition (`b1_*`, `b2_*`).

**B:** V_down = branch-message coalition (`r1`, `r2`); E = branch/edge set.

On `h1 = h0 + F1(h0)`, unit intervention on `b1_*` overwrites hid1 and
therefore the residual message **and** skip2=`h1` together. A decoy hidden
unit that does not write the readout is not the message. Holding skip2 while
leaving `r2_P` intact is a different intervention from `do(r2)`.

**Verdict: B is identifiable; A is not.** This is an intervention fact, not
an aesthetic preference.

## Planted instrument (not learned evidence)

| Case | Plant | Recompute V_up={act_0} | Mean-fill act-only | Conditional class |
|---|---|---|---|---|
| A | F1 copies carrier onto readout | sufficient | fails | `DOWNSTREAM_F1` |
| B | F2 copies carrier onto readout | sufficient | fails | `DOWNSTREAM_F2` |
| C | F1 then F2 sequential | sufficient | fails | `DOWNSTREAM_F1_F2` |
| D | carrier is the readout (skip) | sufficient | sufficient | `DIRECT` |
| E | F1 and F2 independently copy | sufficient | fails | `REDUNDANT_DOWNSTREAM` |
| F | 3·F1 + (−2)·F2 | sufficient | fails | `INTERACTING_DOWNSTREAM` |

### Critical regression

Plant A: global recompute-MSRS identifies the **action stem**. Conditional
mediation additionally identifies **r1 / F1**. If that split failed, 006
would be abandoned rather than frozen.

Observed on the instrument: `critical_pass = true`.

Gauge: the same F1 role with a rotated readout coordinate still classifies
`DOWNSTREAM_F1`. Literal units change; branch role survives. This is a
planted-micro-model statement, not a learned gauge-invariant mechanism.

## What this does not establish

Learned PointMass mechanisms; 005 path class; JEPA; confirmation; Level 3
on any trained seed.
