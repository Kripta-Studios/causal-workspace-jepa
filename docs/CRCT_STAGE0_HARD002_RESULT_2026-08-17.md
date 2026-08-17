# CRCT-STAGE0-HARD-002 — adjudicated result (2026-08-17)

Status: `NEGATIVE_RESULT` (scientific), infrastructure clean.

This document freezes the independent post-run adjudication of
`CRCT-STAGE0-HARD-002`. Thresholds are not retuned and the primary seeds are not
rerun. The experiment is a planted synthetic falsification benchmark only; it
does not establish a Qwen circuit, a JEPA mechanism, a workspace, or SOTA.

## Reproducibility identity

- Base commit: `f69cc28f00faf9d5382e3a47a551410785ae9374`.
- Primary seeds: `1009`, `2027`, `4093`.
- Profile: `full`.
- Runtime: Python 3.14.2, PyTorch 2.10.0+cu128, RTX 5070 Ti Laptop GPU (SM120).
- Uploaded diagnostic bundle SHA-256:
  `49eacf9861419f194781ceadef4079b48d6d062334ce654a652f129cf1f6118b`.
- Per-seed result SHA-256:
  - 1009: `092fb81cbe1dc5b44a25906ad40308c1b85ba65ca68656077adb1053e5c8eb80`
  - 2027: `b87cfc1b4acc54a10470b53276af825ec3950096bb82ed1b77053995e426dcf2`
  - 4093: `1a5928e304f99efd19d71837d03036a8c79fe5c7c3cc4e9a89d89af577eb2d4f`

All targeted tests, the CUDA preflight, provenance guards, and all three model
runs completed without infrastructure failures. Validation discovery was frozen
and hashed before IID/OOD confirmation data were generated.

## Primary result

| seed | residual power after T2 | IID recovery | OOD recovery | node P/R | QK-edge P/R | matched-control p | selected - control p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1009 | 0.0547 | 0.9956 | 0.9978 | 1.000 / 0.800 | 1.000 / 1.000 | 0.003891 | 0.9997 |
| 2027 | 0.1290 | 0.9928 | 0.9924 | 1.000 / 0.400 | 1.000 / 1.000 | 0.003891 | 0.9728 |
| 4093 | 0.0420 | 0.9969 | 0.9967 | 1.000 / 0.600 | 1.000 / 1.000 | 0.003891 | 0.9822 |

The aggregate result is negative for two distinct preregistered reasons:

1. Seeds 1009 and 4093 fail the residual-eligibility gate
   `residual_power_fraction >= 0.08`. Their second-order differential model
   already explains about 94.5% and 95.8% of finite-effect energy respectively.
   Under the frozen rule, a learned residual route is not justified in these
   plants.
2. Seed 2027 has enough residual power (`0.1290`) but fails the full planted-node
   recall gate (`0.400 < 0.600`). It nevertheless reconstructs 99.28% of IID and
   99.24% of OOD residual effect with precision 1.0 and all three planted QK-like
   edges recovered.

These are scientific negative results, not software failures.

## What survived the harder benchmark

The validation-only selected circuits remain extremely faithful under both IID
and OOD confirmation. All three seeds have circuit recovery above 0.992, edge
precision/recall 1.0/1.0, decoy rejection 1.0, and gauge-safe causal rank
Spearman 1.0. The activation-magnitude ranking is coordinate-sensitive
(Spearman approximately 0.386, 0.623, and 0.501 after the compensated gauge
change), while the causal score is invariant.

Unlike CRCT-STAGE0-001, matched controls are frozen before confirmation.
All 256 controls per seed are generated from validation-only kind/activation/
finite-effect matching. No control beats the selected circuit on IID
confirmation, giving the finite-sample floor `1/257 = 0.003891`; margins above
the control p95 are 0.9997, 0.9728, and 0.9822. This supports bounded
specificity of the frozen selection procedure on the planted system.

The QK-like edge result is a useful engineering positive control: all three
true bilinear routing edges are recovered in 3/3 seeds despite active
cancelling and state-only QK decoys. It validates the synthetic edge-accounting
machinery, not attention-circuit discovery in a transformer.

## Why node recall and functional recovery disagree

The planted `truth_nodes` set contains redundant and cancellation pathways by
construction. In seed 2027, the frozen selector chooses four of ten planted
nodes plus all three QK edges, yet reconstructs more than 99% of the residual.
Several omitted planted nodes have tiny, opposing, or redundant validation
contributions. A greedy single-component selector is optimized for sparse
signed reconstruction, not exhaustive recovery of every edge in a redundant
generative graph.

Therefore HARD-002 exposes an ontology mismatch rather than licensing a
post-hoc threshold change. Future synthetic work must report separately:

- full planted-graph recall;
- epsilon-functional completeness/sufficiency;
- individually necessary mechanisms;
- redundancy/cancellation group coverage; and
- equivalence classes of functionally substitutable circuits.

A future group/coalition-aware selector must be frozen on new calibration
plants and evaluated on fresh primary seeds. HARD-002 must not be rerun to
validate such a redesign.

## Differential and learned-predictor diagnostics

First-order and second-order component rankings are already highly correlated
with exact finite patching. `Screen-Flag-Fix` flags zero components in seeds
1009/4093 and one in seed 2027; it does not produce a decisive additional
benefit on this plant. This is retained as a negative/neutral diagnostic, not
evidence against HVP corrections in real networks.

The equal-capacity direct-delta MLP beats the residual MLP in normalized error
on every primary seed, both IID and OOD:

| seed | residual/direct IID NMSE | residual/direct OOD NMSE |
|---:|---:|---:|
| 1009 | 0.00942 / 0.00118 | 0.04161 / 0.00908 |
| 2027 | 0.00389 / 0.00072 | 0.02269 / 0.01468 |
| 4093 | 0.00564 / 0.00072 | 0.03211 / 0.00935 |

Thus HARD-002 provides no evidence that learning only the T2 residual is a
better predictive target than learning the full finite delta. The
differential-plus-residual architecture remains conditional on independent
real-model evidence that a substantial, predictable residual exists.

## Adjudication of the generated summary

The generated `summary.md` section titled `Frozen gates` lists the frozen
*required gate values* from configuration, not the observed per-seed gate
outcomes. It must not be read as saying every gate passed. The authoritative
outcomes are the per-seed JSON `gates` fields and the aggregate status.

Observed failed gates:
- seed 1009: `finite_residual_power_ge_0_08 = false`;
- seed 2027: `node_recall_ge_0_60 = false`;
- seed 4093: `finite_residual_power_ge_0_08 = false`.

No rerun is required; this wording bug does not affect any metric or status.

## Scientific continuation

Do not run a larger version of this same synthetic benchmark and do not lower
the failed thresholds.

1. Freeze HARD-002 as the negative falsification above.
2. Implement coalition/group-aware circuit metrics on development-only plants,
   distinguishing exhaustive graph recovery from minimal functional circuits.
   Use new untouched primary seeds only after the metric/selector is frozen.
3. Treat already-open capital-patch data as development-only for integrating
   exact patching, QK/head paths, HVP reliability diagnostics, and sparse
   substrates. It cannot provide a new confirmatory claim.
4. For Qwen, keep the existing competence-first binding/permutation protocol.
   Phase 0 must first measure exact finite residual beyond strong JVP/T2/
   relinearized baselines. A learned residual model is eligible only if that
   signal survives the frozen gate.
5. Compare sparse substrates rather than assuming a privileged basis:
   Qwen-Scope SAEs, cross-layer transcoder/attribution-graph features, raw
   heads/MLPs, and Sparse Weight Decomposition where feasible. Final claims
   require direct intervention on the original model.
6. For EB-JEPA, finish the frozen multi-seed training/competence gate before
   recurrent mechanism analysis. If competence passes, trace
   action -> GRU gates/state -> predicted latent -> cost -> selected action ->
   closed-loop success with direct path/gate interventions.

The workspace hypothesis remains secondary. A workspace claim is not reopened
until a real model first yields a validated mechanism with necessity,
sufficiency, specificity, faithfulness, and held-out generalization.
