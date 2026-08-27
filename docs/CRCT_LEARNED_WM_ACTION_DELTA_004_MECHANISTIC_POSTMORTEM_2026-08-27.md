# CRCT-LEARNED-WM-ACTION-DELTA-004 — mechanistic post-mortem (2026-08-27)

```text
PARENT STATUS:          INCONCLUSIVE  (unchanged)
EVIDENCE_LEVEL:         None          (unchanged)
CONFIRMATION:           CLOSED
THIS DOCUMENT:          post-mortem / successor design only
TRAINING:               none
FREEZE OF 005:          none
```

Authoritative 004 outcome remains the committed adjudication and post-run
review. This note does not retune gates, does not promote seed 97, does
not promote seed 101 `REDUNDANT_ROUTES`, and does not reinterpret `G_full`
as Level 3.

Adjudication: `docs/CRCT_LEARNED_WM_ACTION_DELTA_004_ADJUDICATION_2026-08-27.md`
(`d62f8fe`). Post-run review:
`docs/CRCT_LEARNED_WM_ACTION_DELTA_004_POST_RUN_REVIEW_2026-08-27.md`
(`d2a36e2`; independent review `48f0d4cd`; **AFFIRM INCONCLUSIVE**; P0 none).

Scientific access for this session:

- already opened: 004 development artifacts (rungs 800 and 2000);
  frozen 001/002/003 records as historical parents;
- still protected: confirmation seeds 1063/1069/1087; `test` / `paraphrase`;
- already frozen: 004 thresholds, seeds, path-hold equations, adjudication;
- still changeable: 005 draft only (`DRAFT_NOT_PREREGISTERED`).

Parents remain frozen: 001 `MODEL_INCOMPETENT`; 002 `INCONCLUSIVE` (seed 59
is not a pass); 003 `MODEL_INCOMPETENT`; IBD-003 `MECHANISM_RECOVERY_PASSED`
(synthetic IBD); HARD-002 `NEGATIVE_RESULT`.

## OBSERVED

Selected rung 2000 (first all-competent frozen rung). Rung 5000 was not
run. Confirmation was not opened. Substrate: supervised residual MLP
(`PathAwareActionDeltaPredictor`), not a JEPA objective.

Rung 800: all three development seeds `MODEL_INCOMPETENT` on `Δx`/`Δy`;
CRCT not run. Those rows have no localization, MSRS, or path metrics.

Rung 2000 CRCT (recorded; not re-derived):

| Seed | MSRS | action-stem | path_class (diag.) | G_full | G_skip | G_res | suff Δvx | nec Δvx | spec Δvy / Δy | status | level |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---|---:|
| 97 | `{act_0, act_3, act_1}` | yes | `DIRECT` | 0.992 | 0.998 | 0.197 | 0.0083 | 1.23 | 3.30 / 3.99 | `INFORMATION_GATEWAY_ONLY` | 2 |
| 101 | `{act_3, act_1, act_5, b1_0}` | no | `REDUNDANT_ROUTES` | 0.892 | 0.950 | 0.679 | **0.088** | 1.83 | 2.38 / **0.404** | `SUFFICIENCY_FAILED` | 0 |
| 107 | `{act_5, act_0, act_1}` | yes | `DIRECT` | 0.978 | 0.996 | −0.203 | 0.0264 | 0.771 | 2.23 / **1.941** | `SPECIFICITY_FAILED` | 0 |

Literal Jaccard across seeds: 97∩107 = 0.5; 97∩101 = 0.4; 101∩107 = 0.4.
Shared Level-3 class: none. Experiment `INCONCLUSIVE`, evidence `None`.

## Per-seed state machine (Q1)

Columns follow the reconstruction order requested for this post-mortem,
not the frozen `adjudicate_seed` sequence. Frozen first-matching order
is: nonempty MSRS → sufficiency → minimality → necessity → specificity
→ random → act-random → `G_full` → gauge function → gauged nec/suff →
gauged path class → action-stem veto → probe-uniqueness (non-split) →
`INTERACTING` / `REDUNDANT_ROUTES` / `DIRECT` / `DISTRIBUTED`.

Values are from
`artifacts/metrics/crct_learned_wm_action_delta_v4.rung2000.json`.
“Not used” means recorded but not status-determining because an earlier
*machine* gate already stopped. No missing cell is filled by inference.

### Seed 97 — stops at action-stem veto (Level 2)

| # | Gate | Recorded | Used? |
|---|---|---|---|
| 1 | competence (2000) | all four Δ NMSE ≤ 0.05 (`Δvx` 0.00162) | pass |
| 2 | information-carrier localization | train-split `ax` R²; top sites `b2_4` 0.816, `b1_3` 0.814, `act_0` 0.810; MSRS is not the probe top-k | diagnostic |
| 3 | CRCT MSRS | `{act_0, act_3, act_1}` (pre-prune identical) | pass (nonempty) |
| 4 | necessity Δvx | 1.230 ≥ 0.10 | pass |
| 5 | sufficiency Δvx | 0.0083 ≤ 0.05 | pass |
| 6 | minimality | drop NMSE `act_0` 0.358, `act_1` 0.147, `act_3` 0.274; none ≤ 0.05 | pass |
| 7 | specificity | Δvy 3.30, Δy 3.99; both ≥ 2 | pass |
| 8 | G_full | 0.992 ≥ 0.50 | pass |
| 9 | G_direct (= G_skip) | 0.998 | diagnostic |
| 10 | G_distributed (= G_res) | 0.197 | diagnostic |
| 11 | G_residual | 0.197 (alias of G_res) | diagnostic |
| 12 | G_skip | 0.998 | diagnostic |
| 13 | path classification | `DIRECT` (skip ≥ 0.50, res < 0.50) | recorded; not Level 3 |
| 14 | gauge re-search | `{act_5, act_2, act_3}`; Jaccard 0.20 | run |
| 15 | gauged necessity | 1.040 ≥ 0.10 | pass |
| 16 | gauged sufficiency | 0.0222 ≤ 0.05 | pass |
| 17 | gauged path class | `DIRECT` = original | pass |
| 18 | final | `INFORMATION_GATEWAY_ONLY` (action-stem) | Level 2 |

Also recorded: random / act-random / RMS-matched sufficient counts all 0;
gauge function MSE `1.90e-15`; probe top-k `{b2_4, b1_3, act_0}` failed
causal conjunction (suff 0.245, nec 0.474, G 0.866).

### Seed 101 — stops at sufficiency (Level 0)

| # | Gate | Recorded | Used? |
|---|---|---|---|
| 1 | competence | all four ≤ 0.05 (`Δvx` 0.00117, `Δy` 0.0305) | pass |
| 2 | information carrier | R² top `b1_2` 0.961, `b1_4` 0.940, `b2_2` 0.930, `b1_3` 0.876; MSRS `{act_3, act_1, act_5, b1_0}` | diagnostic |
| 3 | CRCT MSRS | `{act_3, act_1, act_5, b1_0}`; downstream `{b1_0}` | nonempty |
| 4 | necessity Δvx | 1.828 | recorded pass; later |
| 5 | sufficiency Δvx | **0.0883 > 0.05** | **STOP** |
| 6 | minimality | drop NMSE 0.460 / 0.684 / 0.268 / 0.164 | not used |
| 7 | specificity | Δvy 2.38; **Δy 0.404** | would fail if reached |
| 8 | G_full | 0.892 | not used for status |
| 9–12 | G_direct / G_distributed / G_residual / G_skip | 0.950 / 0.679 / 0.679 / 0.950 | diagnostic only |
| 13 | path class | `REDUNDANT_ROUTES` | **not a formal status** |
| 14 | gauge re-search | `{act_3, b1_2, b1_1, act_1}`; Jaccard 0.333 | not used |
| 15–16 | gauged nec / suff | 1.122 / **0.112** | gauged restore also fails |
| 17 | gauged path class | `DISTRIBUTED` ≠ original `REDUNDANT_ROUTES` | not used |
| 18 | final | `SUFFICIENCY_FAILED` | Level 0 |

### Seed 107 — stops at specificity (Level 0)

| # | Gate | Recorded | Used? |
|---|---|---|---|
| 1 | competence | all four ≤ 0.05 | pass |
| 2 | information carrier | R² top `b1_5` 0.869, `act_2` 0.818, `b2_3` 0.788; MSRS `{act_5, act_0, act_1}` | diagnostic |
| 3 | CRCT MSRS | `{act_5, act_0, act_1}` action-stem | nonempty |
| 4 | necessity Δvx | 0.771 | pass |
| 5 | sufficiency Δvx | 0.0264 | pass |
| 6 | minimality | drop 0.156 / 0.084 / 0.523; none ≤ 0.05 | pass |
| 7 | specificity | Δvy 2.23; **Δy 1.941 < 2** | **STOP** |
| 8 | G_full | 0.978 | not used for status |
| 9–12 | G_direct / G_distributed / G_residual / G_skip | 0.996 / −0.203 / −0.203 / 0.996 | diagnostic only |
| 13 | path class | `DIRECT` | not Level 3 |
| 14 | gauge re-search | `{act_0, act_1, act_2}`; Jaccard 0.50 | not used |
| 15–16 | gauged nec / suff | 0.790 / 0.0332 | not used |
| 17 | gauged path class | `DIRECT` | not used |
| 18 | final | `SPECIFICITY_FAILED` | Level 0 |

Also recorded: `act_random_control_sufficient_count` = 1. If specificity
had passed, the frozen machine would have stopped at `INCONCLUSIVE`
(act-random > 0) *before* the action-stem veto. That seed still could
not have been Level 3. The recorded status is `SPECIFICITY_FAILED`.

## LEVEL-1 EVIDENCE

Level 1 = an information carrier for `ax` (probe R²), not a causal
mediator.

On every competent seed the highest `ax` R² sites are **not** identical
to the CRCT MSRS:

- seed 97: probe `{b2_4, b1_3, act_0}` vs MSRS `{act_0, act_3, act_1}`;
  probe failed sufficiency (0.245) and necessity (0.474);
- seed 101: probe `{b1_2, b1_4, b2_2, b1_3}` vs MSRS `{act_3, act_1, act_5, b1_0}`;
  probe sufficiency 1.02 (failed);
- seed 107: probe `{b1_5, act_2, b2_3}` vs MSRS `{act_5, act_0, act_1}`;
  probe sufficiency 1.90, G 0.014 (failed).

Gradient / magnitude / act-grad top-k baselines also failed the causal
conjunction on every seed.

**Strongest Level-1 claim (development only):** linear `ax` probes locate
high-R² residual-block coordinates that are **not** the CRCT mediator.
Level 1 does not recover the object that passes necessity/sufficiency.
This is not an experiment-level information-localization pass.

## LEVEL-2 EVIDENCE

Level 2 = a causal mediator / MSRS under the frozen machine: necessity,
sufficiency, minimality, specificity vs independent controls, random and
(if action-stem) act-random controls at 0, `G_full` (hybrid patch of
MSRS activations from B into A), and gauge (function MSE, gauged restore,
gauged necessity, gauged path class equal to original). Action-stem plus
that conjunction is `INFORMATION_GATEWAY_ONLY`, not Level 3.

Only seed 97 cleared that conjunction. Its `G_full` 0.992 says: patching
`{act_0, act_3, act_1}` from a paired `ax` counterfactual closes almost
all of the factual–counterfactual `Δvx` gap while state and `ay` stay
fixed.

That is a **development observation** that label-blind CRCT found an
action-stem set which mediates `ax → Δvx` on seed 97. It is **not** an
experiment-level claim: confirmation never opened, evidence remains
`None`, and the other two seeds failed earlier gates.

Seed 101 is the opposite warning: `G_full` 0.892 with restore NMSE 0.088.
Activation-patch mediation and mean-fill sufficiency are different
interventions. High `G_full` without sufficiency is not a Level-2 pass.

Seed 107 is a near-miss on specificity vs `Δy` (1.941), with an additional
recorded act-random control hit. It is not a second Level-2 pass.

**Strongest Level-2 claim:** development-only, seed 97, action-stem MSRS
as a causal mediator of `ax → Δvx`. Not a scientific claim that “CRCT
recovers learned mediators” in general.

## LEVEL-3 FAILURE

Level 3 = a resolved computational pathway: residual-inclusive MSRS, a
split `DIRECT` xor `DISTRIBUTED`, gauge-stable class, and every prior
gate.

No seed reached Level 3.

Missing evidence between “causal information mediator” and “resolved
pathway”:

1. The only seed that passed Level-2 gates had MSRS ⊆ `{act_*}` — the
   input embedding, not a downstream computational edge.
2. The only residual-inclusive MSRS failed sufficiency, so it cannot
   ground a path claim.
3. No seed produced a residual-inclusive xor-split that also passed
   sufficiency, specificity, and gauge class equality.
4. The path holds used to name `DIRECT` / `DISTRIBUTED` do not isolate
   exclusive edges of one MSRS (see PATH IDENTIFIABILITY).

So 004 stopped at Level 2 because the object it could support is “the
channel through which `ax` enters,” and the object it asked for is “the
computation that uses that channel.” The gap is not “need more steps”
and not “need JEPA.”

## Why seed 97 is gateway-only (Q2)

Do not use the label as the explanation. The numerical/causal reason:

**Passed:** competence; nonempty MSRS; necessity 1.23; sufficiency
0.0083; inclusion-minimality; specificity vs `Δvy` and `Δy`; random /
act-random / RMS controls at 0; `G_full` 0.992; gauge function MSE
~1e-15; gauged restore 0.022; gauged necessity 1.04; gauged path class
`DIRECT`.

**Path-resolution:** diagnostic class is `DIRECT` because `G_skip`
0.998 ≥ 0.50 and `G_res` 0.197 < 0.50.

**What blocked `DIRECT_PATH_MECHANISM_PASSED`:** `action_embedding_only`
is true. Frozen machine: action-stem MSRS cannot be Level 3, including
when the diagnostic class is `DIRECT`. That is an ontology veto on the
*object* recovered (the action embedding), not a failure of `G_full`.

**What blocked `DISTRIBUTED_PATH_MECHANISM_PASSED`:** `G_res` 0.197 is
below 0.50, so the class is not `DISTRIBUTED`. Independently, action-stem
would still veto.

**Not the cause:**

- both routes weak? No (`G_skip` strong);
- both routes strong? No (`G_res` weak);
- `INTERACTING`? No (that requires both < 0.50);
- information probe matched CRCT? No (probe top-k ≠ MSRS; probe failed
  conjunction);
- tensor intervention ambiguity as the *status* reason? No: the status
  machine stopped on `action_only` after gauge. Ambiguity is why that
  veto exists: `G_skip` on an action-stem set tests whether patched `h0`
  rides the additive stream with residual internals frozen at A, which
  is the architecture default after the input channel, not a discovered
  residual-block computation.

Seed 97 therefore shows a **causal information mediator** whose
skip-route diagnostic is expected once the mediator is the mix input.
It does not show a resolved computational pathway.

## Seed 101 `REDUNDANT_ROUTES` (Q3)

The recorded class is both `G_skip` 0.950 ≥ 0.50 and `G_res` 0.679 ≥ 0.50
after `G_full` 0.892. Formal status is `SUFFICIENCY_FAILED`. The class
is diagnostic, not a pass.

Three readings:

**TRUE REDUNDANCY** would require each route independently sufficient
under a frozen intervention. Restore of the joint MSRS already failed
(0.088). No skip-only or residual-only restoration test was recorded.
Not established.

**ROUTE NON-IDENTIFIABILITY** fits the implemented holds. `G_skip`
overwrites `hid1`/`hid2` at A, so `b1_0` in the MSRS is unused by
construction. `G_skip` therefore scores the action-carrier subset on
the skip stream, not “the same four-unit mechanism via skip.” `G_res`
holds `skip1 = h0_A` while residual block 1 still reads patched
`h0_mix` and then applies `b1_0` from B. The two scores are overlapping
hybrid forwards, not two independently sufficient pathways.

**INTERACTION** (neither route alone, combination does) is also not
established: both scores are above 0.50, and the joint set is not
mean-fill sufficient.

**Conclusion:** overlapping mediation scores that look redundant.
Do not promote to mechanistic evidence of two sufficient routes.

Gauge on this seed is consistent with non-identifiability: re-search
yields a different residual-inclusive set, gauged sufficiency still
fails (0.112), and gauged class is `DISTRIBUTED` rather than
`REDUNDANT_ROUTES`.

## PATH IDENTIFIABILITY

Architecture (searchable units in parentheses):

```text
z  = enc(state)                         # not searchable
e  = act_enc(ax, ay)                    # (act_*)
h0 = tanh(mix([z, e]))                  # path site only; not in CRCT V
hid1 = tanh(b1_w1(h0_branch)) @ q_b1    # (b1_*)
h1 = skip1 + b1_w2(hid1)
hid2 = tanh(b2_w1(h1)) @ q_b2           # (b2_*)
h2 = skip2 + b2_w2(hid2)
y  = out(h2)                            # Δ head input = h2
```

Defaults: `h0_branch = h0`, `skip1 = h0`, `skip2 = h1`.

Paired A/B: same `state`, same `ay`, different `ax`. Patch = MSRS
activations from B into factual A.

### G_full

- factual tensors: A forward;
- counterfactual tensors: B used only to supply MSRS activations;
- overwritten: MSRS coordinates in `e` / `hid1` / `hid2`;
- preserved: non-MSRS units, encoder `z`, unpatched path defaults;
- residual / skip: both recompute from patched mix unless held;
- output-head input: unheld `h2`.

Isolates the **MSRS as a mediator**, not a single edge.

### G_skip (`G_direct`)

```text
overrides = patch(MSRS from B)
then hid1 := hid1_A
     hid2 := hid2_A
h1 = h0_mix_patched + F1(hid1_A)
h2 = h1 + F2(hid2_A)     # skip2 defaults to this h1
```

- overwritten: MSRS patch, then residual hidden states replaced by A;
- residual MSRS members: **unused**;
- skip branch: `skip1 = h0_mix_patched` (not held);
- residual branch state: `F1`, `F2` frozen at A;
- output-head input: `h0_mix_patched + F1_A + F2_A`.

This alters **two** things at once whenever the MSRS includes `b1_*` /
`b2_*`: it patches residual members and then discards them. Even for
action-stem MSRS it freezes residual internals while letting the patched
mix ride the additive identity. It is not a pure “skip-edge only”
intervention on the MSRS as a set.

### G_res (`G_distributed` / `G_residual`)

```text
overrides = patch(MSRS from B)
skip1 := h0_A
h0_branch defaults to h0_mix_patched
hid1 = F1_pre(h0_mix_patched); apply b1 overrides
h1 = h0_A + b1_w2(hid1)
hid2 from this h1; skip2 defaults to this h1
h2 = h1 + F2(hid2)
```

- skip1: factual mix (held);
- residual block 1 input: **patched mix**, not `h0_A`;
- skip2: **not** held at `h1_A` (intentional 003 P0 repair);
- residual MSRS members: used if present.

This is not “residual edges only.” Residual `F1` still sees the patched
stream. Direct and distributed holds therefore share the patched mix as
an input to residual block 1 unless additional holds freeze `h0_branch`.

### G_skip vs residual interpretation

Yes: `G_skip` confounds residual interpretation. The protocol already
names this (`g_skip_semantics`). Empirically, seed 101’s residual member
`b1_0` cannot appear in `G_skip` at all.

## Are DIRECT xor DISTRIBUTED identifiable? (Q7)

Write the default residual map as

```text
h1 = h0 + F1(h0)
h2 = h1 + F2(h1)
```

Implemented holds are **not** the exclusive edge interventions

```text
DIRECT:      y = out(h0_patched + F1(h0_A) + F2(h1_A))     with residual MSRS unused
DISTRIBUTED: y = out(h0_A + F1(h0_patched))                with skip identity unused
```

`G_skip` matches the first line only after discarding residual MSRS
members. `G_res` does **not** match the second: `F1` is evaluated at
`h0_patched` while `skip1` is `h0_A`, so both the additive stream and
the residual branch see a mixture of A and B.

Therefore the frozen xor is **not identifiable** as

- DIRECT = effect primarily carried through skip/direct route of the
  recovered mechanism, versus
- DISTRIBUTED = residual transformations of that mechanism are
  causally required.

What *is* identifiable for an **action-stem** MSRS is a weaker
architecture-route diagnostic: after patching `e`, does `Δvx` ride
`h0_patched + F1_A + F2_A` more than `h0_A + F1(h0_patched)`? Seed 97
answers yes (0.998 vs 0.197). That is expected if the mediator is the
mix input and the residual stream is additive. It is not Level-3 path
recovery.

For a **residual-inclusive** MSRS the xor is confounded. Seed 101’s
both-high scores are the symptom, not a redundancy finding.

**Methodological blocker:** intervention design, not model size, step
count, or JEPA.

## GAUGE

Recorded, all seeds: function MSE ~1e-15 (≤ 1e-8). Literal Jaccard vs
re-search: 0.20 / 0.33 / 0.50. Units are not invariant under the
compensated hidden-basis change.

Seed 97 (the only Level-2 pass): gauged MSRS is a **different**
action-stem `{act_5, act_2, act_3}`; gauged restore and necessity pass;
diagnostic class remains `DIRECT`. That is development-only
**mediator-class** recurrence (action-stem + skip diagnostic), not
literal-unit stability and not path-mechanism invariance (Level 3 did
not pass).

Seed 101: gauged class `DISTRIBUTED` ≠ `REDUNDANT_ROUTES`; gauged
sufficiency fails. Not path-class stability.

Seed 107: gauged class `DIRECT` on another action-stem, but the seed
failed specificity first.

**Gauge currently validates:** function-preserving reparameterization
(representation map). It does **not** validate path invariance. It does
not license a “mechanistic invariance” claim. Literal unit identity is
unstable. Mediator-class stability is a seed-97 development observation
only.

## CROSS-SEED

Do not claim mechanistic convergence.

- **Literal overlap:** moderate (Jaccard 0.4–0.5); shared `act_1` on all
  three MSRS; 97 and 107 share `{act_0, act_1}`.
- **Mediator-level similarity:** 97 and 107 are both action-stem; 101 is
  residual-inclusive and not sufficient. Only 97 is a Level-2 mediator
  under the frozen gates. Not enough for a shared mediator claim.
- **Path-level similarity:** diagnostic `DIRECT` on 97 and 107; 101
  diagnostic `REDUNDANT_ROUTES`. No Level-3 class exists to compare.

Closest honest pattern: **A-ish** (same broad action-stem mediation with
different literal units) on 97 vs a failed-spec 107, plus **D**
(insufficient evidence) once 101 and the identifiability limit are
included. Not B (demonstrably different mechanisms) and not a resolved
shared path class (C as a finding). Path classes remain **unresolved**.

## IMPLICATION FOR CRCT

Node coalitions `act_*` / `b1_*` / `b2_*` can localize a Level-2
mediator in the searchable set (seed 97). They cannot name the residual
**stream** (`h0`, `h1`), which is where skip-route causality would live.
`G_skip` tries to test that stream by holding residual hidden units, and
in doing so overwrites residual members of the coalition.

The ontology is therefore **sufficient for Level 2** and **insufficient
for Level 3** on this residual architecture. Expanding searchable units
without repairing holds would not fix `G_skip` overwrite. The justified
next primitive is **edge/path interventions with identifiable semantics**
(`I` in `M = (V, E, I)`), not a larger coalition budget and not a
subspace ontology (gauge did not show a preserved subspace object, only
literal instability plus one seed’s action-stem recurrence).

## IMPLICATION FOR META-INTERPRETERS

Design hypothesis, **not** an experimentally established fact:

A meta-interpreter should not emit only `{unit_3, unit_9, unit_14}`.
004’s seed 97 object that survived causal gates was structured:

- input variable: `ax`;
- carrier (Level 1): high-R² sites, which **disagreed** with the mediator;
- mediator (Level 2): coalition `C ⊆ act_*`;
- path class: diagnostic `DIRECT`, not a resolved `E`;
- target: `Δvx`;
- causal evidence: necessity, sufficiency, mediation, specificity, gauge
  function preservation.

What 004 teaches: emitting `C` without `E` and `I` invites promoting an
information gateway to a pathway. The preferred mechanistic object is

```text
M = (V, E, I)
V = relevant internal components (mediator / carrier distinguished)
E = causal computational edges / routes
I = intervention semantics that identify those edges
```

adopted here as a **successor design hypothesis** because the post-mortem
shows `V` without identifiable `I` cannot support Level 3.

## NEXT HYPOTHESIS

004 did not fail because the model was too small or too undertrained at
the selected rung. It failed because CRCT recovered a causal **node set**
and then applied hybrid forwards that cannot identify the **edges** of
that set.

Successor (draft only):
`docs/research/CRCT_LEARNED_WM_ACTION_DELTA_005_DRAFT.md`.

Target: path-identifiability repair, conditioned on a Level-2 mediator
where one exists, using edge holds that do not overwrite the tested
edge. Not 5000 steps. Not gate retuning. Not JEPA.

Independent post-mortem review (`095a1172`): **AFFIRM_POSTMORTEM**. P0
none. P1 table-order / seed-107 counterfactual-status / Level-2
conjunction wording were repaired in this file; 004 adjudication is
unchanged.

**STOP** pending authorization of a new prospective mechanistic protocol.
