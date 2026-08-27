# CRCT-LEARNED-WM-ACTION-DELTA-004 protocol

Status at freeze: `PREREGISTERED_NOT_RUN`.
Execution is authorized by the freeze commit that follows written
independent freeze-allowed verdicts on the repaired candidate.

Parent 003: `MODEL_INCOMPETENT` (evidence `None`; CRCT not run).
Parent 002: `INCONCLUSIVE`. Seed 59 is **not** a pass.
Parent 001: `MODEL_INCOMPETENT`. IBD-003 remains synthetic
`MECHANISM_RECOVERY_PASSED`. HARD-002 remains `NEGATIVE_RESULT`.
IBD-002 was not executed.

The draft `docs/research/CRCT_LEARNED_WM_ACTION_DELTA_004_DRAFT.md` is not
this freeze.

## Nomenclature

Supervised residual-MLP PointMass predictor. **Not** a JEPA objective.

## Question

Did label-blind CRCT identify the **computational pathway** implementing
learned `ax → Δvx`, or only the **channel** through which `ax` enters?

| Level | Name | Meaning |
|---|---|---|
| 1 | information carrier | `ax` is decodable |
| 2 | causal mediator | interventions on an MSRS mediate `Δvx` |
| 3 | computational path mechanism | a **resolved** skip vs residual route class |

Level 3 requires a **split** path class **and** a residual-inclusive MSRS.
`G_full` alone is Level 2. Action-stem MSRS is Level 2
(`INFORMATION_GATEWAY_ONLY`) even if `G_skip` vs `G_res` splits.

## Finite competence ladder (frozen before any 004 training)

Historical evidence only (no 004 seeds):

| Source | Steps | Result |
|---|---:|---|
| 001 | 200 | all three seeds incompetent (Δx/Δy) |
| 002 | 200 | incompetent; 800 competent on 59/71/73 |
| 002 | 2000 | never run (stop rule) |
| 003 | 800 | seed 79 Δy NMSE 0.139; 83/89 competent; CRCT closed |

Ladder on **new** seeds:

| Rung | Adam steps | Role |
|---|---:|---|
| A | 800 | 003 budget; expected mixed; control |
| B | 2000 | 002’s unused third rung, new seeds only |
| C | 5000 | a-priori last cap (`~6×800`); **not** historically attested |

Rung C must not be described as 002’s unused 2000 rung.

Optimizer unchanged: Adam `lr=3e-3`, batch 64, pooled MSE, `H=6`.
No extra rungs. No architecture/optimizer/threshold changes after outcomes.

Climb development only. Competence: all four Δ NMSE `<= 0.05` on that seed’s
development eval (`S*1000+67`). Persist per channel variance, energy, MSE,
NMSE, pred_variance; `train_loss_curve` (10 points, not a stopping rule);
`checkpoint_sha256`.

- If **all** development seeds pass at rung R: select R, **stop**, then CRCT.
- If a rung fails: only the next frozen rung.
- If C fails: `MODEL_INCOMPETENT`, confirmation **CLOSED**, CRCT **CLOSED**.

## Seeds (new; disjoint)

Forbidden: 003 forbidden set plus 003 seeds `79,83,89,1049,1051,1061`.

- Development: `97, 101, 107`
- Confirmation: `1063, 1069, 1087`

```text
train:    S*1000+61   256 traj × 6 steps
dev:      S*1000+67   64 traj
confirm:  S*1000+71   64 traj  (confirmation stage only)
```

## Specificity (from physics graph)

Direct target: `Δvx`. Downstream of `ax`: `Δx` (diagnostic, **not** a
negative control). Independent controls: `Δvy`, `Δy`.

Gates: `nec_Δvx/nec_Δvy >= 2` and `nec_Δvx/nec_Δy >= 2`.
Do **not** require a ratio vs `Δx`.

## MSRS vs MCP

**MSRS:** pruned greedy restore set on the model’s own `Δvx` (label-blind
units; `max_coalition=4`, `min_step_nmse=0.02`, restore NMSE `<= 0.05`).

**MCP:** MSRS plus a resolved path class. MSRS ≠ MCP by default.

Selector: 001/002 greedy + inclusion-minimal prune. Mean-fill = coordinatewise
baseline. Encoder excluded. Residual stream `h0`/`h1` are path sites only.

## Tensor-level path holds (paired A/B: same state, same `ay`, `ax` differs)

Factual A and counterfactual B forwards. Patch = MSRS activations from B
into A. `G_*` = median fraction of `Δvx` gap closed (same 002 pair generator,
`n=64`, bar `0.50`).

Let `e` be post-tanh action embedding after optional unit overrides.
`h0_mix = tanh(mix([z_A, e]))`.

**G_full** (Level 2):
```text
overrides = patch(MSRS from B)
path_holds = {}
```

**G_skip** (`G_direct`; **architecture-route test**):
```text
overrides = patch(MSRS from B)
then hid1 := hid1_A
     hid2 := hid2_A
h1 = h0_mix + b1_w2(hid1_A)
h2 = h1 + b2_w2(hid2_A)
```
Residual MSRS members are **unused by construction**. This tests whether
the skip/stream from the **action-carrier subset** transmits `ax→Δvx`.
It is **not** proof that residual units in the MSRS are idle in the
unheld network.

**G_res** (`G_distributed`):
```text
overrides = patch(MSRS from B)
skip1 := h0_A
hid1 = tanh(b1_w1(h0_mix)); apply b1 overrides
h1 = h0_A + b1_w2(hid1)
hid2 = tanh(b2_w1(h1)); apply b2 overrides
h2 = h1 + b2_w2(hid2)     # skip2 defaults to this h1; do not hold skip2=h1_A
```

`G_direct := G_skip`. `G_distributed := G_res`. `G_residual := G_res`.

## Path class (only if `G_full >= 0.50`)

| class | rule | level / per-seed status |
|---|---|---|
| `DIRECT` | `G_skip >= 0.50` and `G_res < 0.50` | 3 / `DIRECT_PATH_MECHANISM_PASSED` **only if** MSRS is residual-inclusive and other gates pass |
| `DISTRIBUTED` | `G_res >= 0.50` and `G_skip < 0.50` | 3 / `DISTRIBUTED_PATH_MECHANISM_PASSED` **only if** MSRS is residual-inclusive and other gates pass |
| `REDUNDANT_ROUTES` | both `>= 0.50` | 2 / `REDUNDANT_ROUTES`; not a split; confirmation **closed** |
| `INTERACTING` | both `< 0.50` | 2 / `MEDIATOR_FOUND_PATH_UNRESOLVED` unless the gateway override (action-stem, or non-split probe conjunction) fires first |

Status order: action-stem → `INFORMATION_GATEWAY_ONLY` **before** the
`INTERACTING` / `REDUNDANT_ROUTES` / split labels. Residual-inclusive
`INTERACTING` is `MEDIATOR_FOUND_PATH_UNRESOLVED`.

`INTERACTING` **cannot** be `PATH_MECHANISM_RECOVERY_PASSED`.

`REDUNDANT_ROUTES` is option A: unresolved unique route. Experiment-level
status may be `REDUNDANT_ROUTES`. `all_seeds_passed` is **false**. Evidence
`None`. Confirmation **does not open**. It is **not** renamed `H_DIRECT`
and is **not** Level 3.

## Information gateway

Train-split linear probe `ax ~ site`; report `R^2`; top-`k` with `k=|MSRS|`.
Same necessity, sufficiency, `G_full` tests. Diagnostic for uniqueness.

`INFORMATION_GATEWAY_ONLY` if `G_full >= 0.50` and **either**:

- MSRS ⊆ `{act_*}` (action-stem / channel), **including** when the
  diagnostic path class is `DIRECT` or `DISTRIBUTED` (`G_skip` on an
  action-stem MSRS is an architecture-route test of the residual stream,
  not Level-3 pathway recovery); or
- path class is **not** a split and probe top-k itself meets sufficiency
  `<= 0.05`, necessity `>= 0.10`, and `G_full >= 0.50`.

A residual-inclusive MSRS with a **split** class may still be Level 3.
Probe overlap on that residual-inclusive split is recorded
(`probe_uniqueness_failed`); it does not override the split.

This does **not** relabel 002 seed 59. It prevents the same *pattern*
from counting as a 004 Level-3 pass.

No `R^2` numeric floor. Relative comparison and causal tests only.

## Other gates (copied from 001/002; in `threshold_digest`)

sufficiency `<= 0.05`; necessity `>= 0.10`; spec ratio `>= 2`; random
same-size 0 sufficient of 32; if MSRS ⊆ `{act_*}`, also 0 sufficient of 32
same-size subsets of `{act_*}` excluding the MSRS (**up to 32 or the full
remainder**; `|act_*|=6` so `C(6,k)-1` may be `< 32`); RMS-matched same-size
subsets of the random pool are **recorded** (`rms_matched_sufficient_count`),
not a uniqueness gate (same as 001/002); cancellation may be
`NO_MEANINGFUL_CANCELLATION_DETECTED`.

Magnitude / gradient / act×grad top-k: same nec/suff/`G_full` conjunction
is **recorded** (`causal_conjunction`); not a uniqueness veto.

## Gauge

Post-tanh `h' = hQ`, `W' = WQ`. Function MSE `<= 1e-8` or `GAUGE_FAILED`.
Re-run greedy+prune. Gauged restore NMSE `<= 0.05` **and** gauged
necessity `>= 0.10` or `GAUGE_FAILED`. Gauged path class **must equal**
original class or `PATH_CLASS_GAUGE_UNSTABLE`. Literal Jaccard is recorded,
not a gate.

## Per-seed status (first matching)

1. competence fail → `MODEL_INCOMPETENT`
2. empty MSRS → `LOCALIZATION_FAILED`
3. restore `Δvx` `> 0.05` → `SUFFICIENCY_FAILED`
4. not inclusion-minimal → `MINIMALITY_FAILED`
5. necessity `Δvx` `< 0.10` → `NECESSITY_FAILED`
6. independent specificity fail → `SPECIFICITY_FAILED`
7. random (and act-restricted if applicable) sufficient `> 0` → `INCONCLUSIVE`
8. `G_full < 0.50` → `INCONCLUSIVE`
9. gauge function MSE, gauged sufficiency, or gauged necessity fail →
    `GAUGE_FAILED`
10. gauged path class ≠ original → `PATH_CLASS_GAUGE_UNSTABLE`
11. MSRS ⊆ `{act_*}` → `INFORMATION_GATEWAY_ONLY` (path class diagnostic)
12. probe uniqueness fail **without** a split class → `INFORMATION_GATEWAY_ONLY`
13. class `INTERACTING` → `MEDIATOR_FOUND_PATH_UNRESOLVED`
14. class `REDUNDANT_ROUTES` → `REDUNDANT_ROUTES` (Level 2; not a pass)
15. class `DIRECT` and residual-inclusive → `DIRECT_PATH_MECHANISM_PASSED`
16. class `DISTRIBUTED` and residual-inclusive → `DISTRIBUTED_PATH_MECHANISM_PASSED`

Action-stem MSRS is a **Level-3 veto**. The 002 status name
`ARCHITECTURE_CUTSET` is not emitted (002 remains frozen).

## Experiment-level status

- any development seed incompetent → `MODEL_INCOMPETENT`
- all seeds `DIRECT_PATH_MECHANISM_PASSED` → `PATH_MECHANISM_RECOVERY_PASSED`
  (`shared_path_class=DIRECT`)
- all seeds `DISTRIBUTED_PATH_MECHANISM_PASSED` → `PATH_MECHANISM_RECOVERY_PASSED`
  (`shared_path_class=DISTRIBUTED`)
- all seeds `REDUNDANT_ROUTES` → `REDUNDANT_ROUTES` (`all_seeds_passed=false`;
  evidence `None`; confirmation closed)
- else if all remaining interpreted seeds share one failure status → that status
- else `INCONCLUSIVE`

Confirmation opens **only** for development `PATH_MECHANISM_RECOVERY_PASSED`
with shared class `DIRECT` or `DISTRIBUTED`. All three confirmation seeds
must pass **the same** shared split class as development. Train exactly the
selected rung. Incompetent confirmation seed: no CRCT;
`MODEL_INCOMPETENT_CONFIRMATION`. If all confirmation seeds pass a
*different* shared class than development: `CONFIRMATION_PATH_CLASS_MISMATCH`
(not a pass). `REDUNDANT_ROUTES` does **not** open confirmation.

`H_EQUIVALENT`: ≥2 Level-3 passing seeds, Jaccard `< 1`, **same** split
class. `functional_convergence` is this flag, not any experiment pass.

Evidence: experiment-level `PATH_MECHANISM_RECOVERY_PASSED` →
`Causal effect`. Else `None`.

## Provenance

CLI and `run_development_rung` / `run_confirmation` refuse unless the
config has `execution_authorized: true` and is not `DRAFT_NOT_PREREGISTERED`.

```text
python -m causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_004 --stage development --rung 800 --output artifacts/metrics/crct_learned_wm_action_delta_v4.rung800.json
python -m ... --stage development --rung 2000 --require-previous ...rung800.json --output ...rung2000.json
python -m ... --stage development --rung 5000 --require-previous ...rung2000.json --output ...rung5000.json
python -m ... --stage confirmation --require-development <selected-rung.json> --output artifacts/metrics/crct_learned_wm_action_delta_v4.json
```

Climbing requires previous this ID, development, `MODEL_INCOMPETENT`, matching
digests, immediately prior rung.

Confirmation requires development status
`PATH_MECHANISM_RECOVERY_PASSED`, `all_seeds_passed`, shared class in
`{DIRECT, DISTRIBUTED}`, matching seeds/digests/selected rung.

## Explicit non-actions

Do not mutate 001/002/003. Do not reuse their seeds. Do not interpret 83/89.
Do not execute JEPA, friction, MiniPush, planning, stitching, Qwen 004,
or IBD-002.
