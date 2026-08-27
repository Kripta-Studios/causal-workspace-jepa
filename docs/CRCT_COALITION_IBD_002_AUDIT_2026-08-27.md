# CRCT-COALITION-IBD-002 — protocol audit (2026-08-27)

Status of IBD-002 remains **`PREREGISTERED_NOT_RUN`**.

This document does **not** mutate IBD-002 gates, does **not** execute IBD-002,
and does **not** relabel IBD-001 smoke or HARD-002.

Independent protocol review: `fa8fe000-1a02-4796-ae60-9e863b3f3f87`.
Independent adversarial review: `dc9cd773-f572-4761-88a7-1e51021dd055`.

## Verdict

**Not adequate to execute as frozen.** The 22-line protocol is a seed-and-gauge
stub on the IBD-001 runner. It does not test whether CRCT can *recover*
functionally correct mechanisms under redundancy, cancellation, or equivalence.

## HARD-002 lesson (preserved)

HARD-002 is `NEGATIVE_RESULT` on seeds 1009/2027/4093. Seeds 1009/4093 fail
residual eligibility after T2. Seed 2027 reconstructs >99% of residual effect
with node recall 0.400. Literal planted-node recovery is the wrong primary
object when the plant has redundant, cancelling, and substitutable paths.
That negative is informative. It is not retuned.

## Material defects (do not “fix” inside IBD-002)

1. **No executable 002 runner/config.** The only module still emits
   `CRCT-COALITION-IBD-001` and accepts only seeds 11/13/17 and 811/823/829.
2. **No selector.** `selected = ["known_a", "unknown", "residual"]` is hardcoded.
3. **False equivalence.** `known_a` and `known_b` are the same map
   `features @ known_w @ known_r`. Pairwise NMSE is 0. `known_b` is not in the
   forward.
4. **No interventions.** Sufficiency is an oracle additive sum of named tensors.
5. **Gauge Spearman is tautological** on output energies that the compensated
   `known_w *= s`, `known_r /= s` leaves invariant.
6. **`decoy_causal = 0.0` is assigned**, not measured; decoy is stripped from
   the evaluator maps.
7. **Matched control is the exact cancel pair**, which is insufficient by
   construction.

Swapping in IBD-002 seeds without a new ID would mutate the frozen IBD-001
runner and still not ask a recovery question.

## Successor

A new ID is required: `CRCT-COALITION-IBD-003`.
See `docs/CRCT_COALITION_IBD_003_PROTOCOL.md`.
