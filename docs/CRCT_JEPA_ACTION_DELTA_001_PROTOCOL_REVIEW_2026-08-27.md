# Independent protocol review — CRCT-JEPA-ACTION-DELTA-001

Reviewer role: protocol design, before freeze, before any trained-seed outcome.
Draft `docs/research/CRCT_JEPA_ACTION_DELTA_001_DRAFT.md` is **not** this freeze.

## Verdict

**Ready to freeze** after the repairs recorded below. No remaining P0.
IBD-002 stays not executed. HARD-002 stays `NEGATIVE_RESULT`.

## P0 (resolved before freeze)

1. **Draft was not an executable protocol.** Missing architecture dims, site
   ontology, thresholds, selector, competence, provenance, and confirmation
   authorization. **Repair:** dedicated protocol + config; do not promote the
   draft verbatim.
2. **TinyJEPA ridge/identity is not a learned mechanism substrate.** **Repair:**
   residual MLP `H=6`, trained on PointMass transitions; encoder trained but
   not searched.
3. **Frozen-downstream overwrite would fake necessity.** Caching `b1`/`b2`
   while ablating `act` leaves action information in stored MLP units.
   **Repair:** causal re-forward; downstream sites recompute unless overridden.
4. **Gauge through `tanh` is not function-preserving.** **Repair:** mix
   post-nonlinearity coordinates `h' = h Q` and compensate the next
   `nn.Linear` as `W' = W Q`. Full-map MSE gate `<= 1e-8`.
5. **Fused CLI provenance (IBD-003 P1) must not be copied.** **Repair:**
   per-stage `stage_cli_command`, collect provenance *before* writing metrics,
   confirmation sidecar seed `1013`, `--require-development` gate, regression
   tests. Do not rewrite IBD-003 history.

## P1 (resolved or accepted as frozen limitations)

1. Residual stream not independently searchable — frozen omission, documented.
2. Greedy (not exhaustive) search — alternate forbidden-set greedy is reported.
3. Specificity ratio `2.0` vs `Δvy` may fail if `ax`/`ay` share embedding
   dimensions. That is a scientific outcome, not a retune license.
4. Confirmation data generated only after circuit freeze, confirmation stage
   only.

## Checks requested by the session brief

| Question | Finding |
|---|---|
| Circular “mechanism”? | No. Competence is vs physics; localization explains the *network’s* `Δvx`; pass requires necessity, sufficiency, specificity, matched controls, and counterfactual patch. |
| Physics → internal labels? | No. Selector sees unnamed sites and the model’s `Δvx` channel. |
| Post-hoc circuit size? | No. `max_size=6`, `min_step=0.02`, `ε=0.05` frozen. |
| Confirmation leakage? | Circuit frozen on `S*1000+67`; confirmation metrics on `S*1000+71` after freeze; CLI refuses unless development passed. |
| Invalid / off-manifold patching? | Mean-ablate and CF patch use in-support activations. Gaussian fills are not pass tokens. |
| Specificity definition? | Frozen `nec_dvx / nec_dvy >= 2`. |
| Gauge validity? | Invertible hidden mix + compensating next linear; not an output-energy tautology. |
| Baselines trivial? | Magnitude / gradient / act×grad are compared, not fail-gated. Honest overlap is allowed. |

## Independent second pass (pre-freeze)

Reviewers [protocol](76235ae6-d065-4bd3-9b1a-fcb051915d3e) and
[adversarial](16a3beaa-044f-44ee-840b-3dd3b2b66675) issued **NO-FREEZE** on the
first design. Prospective repairs (same ID; no outcomes seen):

| P0 | Repair |
|---|---|
| Minimality claimed but not gated | Selector now inclusion-minimal prunes; drop-one sufficiency is a `MINIMALITY_FAILED` gate |
| Full 6-unit action layer selectable at max size 6 | `max_coalition=4`; exact `{act_*}` is `INCONCLUSIVE`; full action cut-set is a reported baseline |
| Gauge recovery of all 6 rotated act units | Same size cap |
| Mean-fill labeled in-support | Relabeled `coordinatewise_mean_fill`; CF is `hybrid_activation_patch` |
| Specificity ignored Δy; Δx is physically coupled | Gate vs `Δvy` **and** `Δy`; protocol records Euler `Δx` coupling |
| Confirmation not source-bound | `threshold_digest` + `source_digest` + `all_seeds_passed`; sidecar stage check |
| min_step used shrinking `current_error` | Absolute `min_step_nmse=0.02` |
| plus-one p impossible for size-1 | Gate is zero sufficient same-size controls |
| RMS mismatch | RMS of activations |
| JEPA name vs supervised MLP | Claim boundary states supervised residual MLP |

Remaining accepted P1: encoder/residual-stream omitted; gauge pass is sufficiency-only plus map identity; greedy not exhaustive.

**Revised freeze verdict:** freeze allowed after these repairs. Do not retune after outcomes.

## Claim boundary (must survive freeze)

Pass would be **Causal effect** on a **learned tiny supervised PointMass MLP**.
It would not be Qwen, workspace, Platonic physics, planning, a JEPA-objective
result, or a HARD-002 rescue.

