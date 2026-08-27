# CRCT-COALITION-IBD-003 protocol

Status at freeze: `PREREGISTERED_NOT_RUN`.
After the freeze commit, execution of **this ID only** is authorized.
`execution_authorized: true` applies to IBD-003 confirmation seeds listed here,
not to IBD-002, Qwen 004, stitching, or planning tracks.

HARD-002 remains `NEGATIVE_RESULT`. IBD-001 remains smoke. IBD-002 remains
`PREREGISTERED_NOT_RUN` and is **not** executed. See
`docs/CRCT_COALITION_IBD_002_AUDIT_2026-08-27.md`.

This is a synthetic interpretable-by-design **recovery** test. It is not a
Qwen, JEPA, workspace, Platonic, or planning claim.

## Question

Can a label-blind coalition selector, using **actual restore/ablate
interventions**, recover at least one member of the planted epsilon-functional
equivalence class, while satisfying frozen necessity, sufficiency, minimality,
specificity, redundancy/substitution, cancellation, matched-control, and
non-tautological gauge criteria?

Success is **not** `recovered_sites == all planted sites`.

## Seeds (chosen before generation)

Forbidden (constructor-blocked): HARD-002 `1009, 2027, 4093`; IBD-001
`11, 13, 17, 811, 823, 829`; IBD-002 `21, 23, 29, 941, 947, 953`.

- Development: `31, 37, 41`
- Confirmation: `971, 977, 983`

Temporal order: freeze code/config/thresholds → commit → run development →
if development passes every seed, generate confirmation splits **after**
validation discovery is frozen on that seed → adjudicate once. Do not retune
thresholds after any confirmation outcome.

## Plant (interpretable-by-design)

Ten unlabeled intervention sites `h0…h9`. Evaluation metadata (not passed to
the selector):

| Sites | Role |
|---|---|
| `h2`, `h7` | Known path A: complementary masks of map `K` |
| `h4`, `h9` | Known path B: a **different** partition of the same `K` |
| `h3` | Unknown / MLP-like |
| `h5` | Residual tanh product |
| `h0`, `h1` | Cancelling `+u`, `-u` (in the forward; net zero) |
| `h6` | High-activation decoy; **measured** causal energy on the target; drives a separate specificity head |
| `h8` | Tiny nuisance write in the forward |

Forward target (path B computed but **not** added by default):

```text
y = write(h2)+write(h7)+write(h3)+write(h5)+write(h0)+write(h1)+write(h8)
```

Restore-only `C` sums **writes of sites in C**, including path-B writes. So restoring
`C_equiv` recovers `y` without an oracle rewire. Restoring a coalition that includes
the decoy write `h6` is not sufficient.

Planted equivalence class of minimal sufficient circuits:

```text
C_forward = {h2, h7, h3, h5}
C_equiv     = {h4, h9, h3, h5}
```

Path B uses a distinct Q-rotated partition of the same known map, so `h2 ≠ h4`
and `h7 ≠ h9` while `h2+h7 ≈ h4+h9`.

## Selector (label-blind)

On validation only: exhaustive inclusion-minimal **restore-only** search over
unlabeled `h0…h9` (max size 6). Truth labels are evaluation metadata.

Pass recovery iff `C_hat` is a size-minimal sufficient set **and** both
`C_forward` and `C_equiv` are discovered.

Then freeze `C_hat` and matched controls. Only then generate IID/OOD
confirmation samples.

## Interventions (executed, not labels only)

Record for each: `operation`, `site`, `magnitude`, `source`, `target`,
`combination`, `support_status`.

- restore-only `C_hat` and restore-only `C_equiv` (real site names)
- ablate each member of `C_hat` on the default forward
- ablate `{h0}`, `{h1}`, `{h0,h1}`
- ablate `h6` on the specificity head (`h6 + 0.2 h3`)
- executed out-of-support `steer`: uncompensated `h2 *= 8`, record NMSE

## Frozen thresholds (a priori; not taken from IBD-003 outcomes)

Primary conjunction (all required):

| Quantity | Gate |
|---|---|
| epsilon | `0.02` |
| recovered set is a planted class member | true |
| both class members discovered | true |
| restoration NMSE of `C_hat` (IID and OOD) | `<= 0.02` |
| drop-one member restore NMSE | `> 0.02` |
| ablation of each `C_hat` member: delta NMSE | `>= 0.05` |
| cancel pair: each single ablation NMSE | `>= 1e-4` |
| decoy activation energy | `> 1.0` |
| random same-size controls `n=32` plus-one p | `<= 0.05` |
| distinct RMS leave-one/next-nearest controls | `count >= 2` and `sufficient_count = 0` |
| decoy-containing control sufficient | `false` |
| spec: ablate `h6` / ablate `C_hat` on spec head | `>= 2.0` |
| gauge activation-rank Spearman | `<= 0.95` (must **move**) |
| activation-RMS top-`k` ≠ `C_hat` | true (and after gauge) |

Diagnostics (recorded, **not** in the primary conjunction): compensated
function MSE, causal-rank Spearman, uncompensated `h2` energy ratio, literal
recall, joint cancel ablation, default-forward decoy ablation NMSE.

RMS controls: for each member of `C_hat`, replace that member with the
nearest unused site by activation RMS, then with the next-nearest, skipping
duplicates, up to 8 distinct coalitions.

Compensated gauge (frozen scales):

```text
h2 *= 25, h7 *= 0.04, h3 *= 8, h5 *= 0.125, h0 *= 5
write[site] *= 1/gauge_scale[site]
```

## Matched controls (frozen on validation)

1. 32 random same-size coalitions excluding `C_hat` **and** other discovered
   size-minimal class members (seeded).
2. Distinct leave-one / next-nearest RMS replacements (`count >= 2`, none sufficient).
3. Cancel pair `{h0,h1}` recorded as a cancellation diagnostic, not the sole control.
4. A decoy-containing coalition (must fail sufficiency).

## Primary adjudication

A confirmation seed is `MECHANISM_RECOVERY_PASSED` only if every **primary**
gate above holds on frozen `C_hat` after confirmation generation.

Aggregate pass: all three confirmation seeds pass.

Any scientific gate fail: `NEGATIVE_RESULT`. Do not retune.

Evidence level on pass: `Causal effect` (synthetic IBD). Claim boundary: not
Qwen/JEPA/workspace/Platonic/planning.

## Commands

```powershell
$env:PYTHONPATH = "src"
python -m causal_workspace_jepa.experiments.cross_domain.crct_coalition_ibd_003 --stage development --output artifacts/metrics/crct_coalition_ibd_v3.dev.json
python -m causal_workspace_jepa.experiments.cross_domain.crct_coalition_ibd_003 --stage confirmation --output artifacts/metrics/crct_coalition_ibd_v3.json
```

Run confirmation only after the freeze commit, and only if development passed.

## Explicit non-actions

Do not execute IBD-002. Do not rerun HARD-002. Do not execute Qwen 004.
Do not open stitching, Reachable-003, Rectified Flow, DINO-WM, or LeWM.
Do not draft/run `CRCT-JEPA-ACTION-DELTA-001` unless this ID’s frozen
primary gate passes.
