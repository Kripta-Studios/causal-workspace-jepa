# CRCT-COALITION-IBD-003 — adjudication (2026-08-27)

Registered outcome: **`MECHANISM_RECOVERY_PASSED`**.
Evidence level: **Causal effect** (synthetic IBD only).

Freeze commit (before outcomes): `fbaec9ccec922ae1a4e0054055f68b3b1e402e96`.
Confirmation seeds: `971, 977, 983`. Development `31, 37, 41` passed first.
IBD-002 was **not** executed. HARD-002 remains `NEGATIVE_RESULT`.

Freeze `fbaec9c` before outcomes. Confirmation provenance sidecars record a
single fused development+confirmation invocation (seed field 31, shared
timestamp); they are not a cleaner per-CLI story. Metrics JSON remain
`run_stage()` output. Gates were not retuned.

## Primary result

A label-blind restore-only search recovered `{h2,h7,h3,h5}` on every
confirmation seed and also discovered the Q-rotated equivalent
`{h4,h9,h3,h5}`. Literal recall of the full mechanistic set is `0.5`
(diagnostic). IID/OOD restoration NMSE is `≈ 8e-4 / 8e-4` to `1.3e-3`.

| Seed | recovered | IID NMSE | OOD NMSE | literal recall | random plus-one p |
|---:|---|---:|---:|---:|---:|
| 971 | `{h2,h3,h5,h7}` | 0.00078 | 0.00077 | 0.5 | 0.0303 |
| 977 | `{h2,h3,h5,h7}` | 0.00111 | 0.00129 | 0.5 | 0.0303 |
| 983 | `{h2,h3,h5,h7}` | 0.00078 | 0.00078 | 0.5 | 0.0303 |

Activation-RMS top-4 never matched `C_hat` (decoy `h6` ranked first).
Gauge activation-rank Spearman `0.43–0.55` (moved). Executed out-of-support
`h2 *= 8` steer NMSE `≈ 19–21`.

## What this establishes

On this interpretable-by-design plant, coalition restore/ablate interventions
can recover a minimal sufficient member of a planted functional equivalence
class, with necessity, minimality, matched-control rejection, signed
cancellation (member ablation), specificity on a decoy-driven head, and a
gauge that moves activation ranking while magnitude selection fails.

## What this does not establish

- Recovery of a *learned* world-model mechanism
- Qwen circuits, workspace, Platonic physics, or planning competence
- That HARD-002 would pass under this ontology (it remains negative)
- That IBD-001/002 smoke is confirmatory
