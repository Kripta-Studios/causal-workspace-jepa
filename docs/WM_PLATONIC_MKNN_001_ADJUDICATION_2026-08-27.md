# WM-PLATONIC-MKNN-001 adjudication

Registered outcome: **`TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED`**.

Evidence level: **Availability**. This is not circuit evidence, not CRCT,
not workspace, not shared coordinates, not platonic physics, and not
permission to open stitching or to relabel `CRCT-STAGE0-HARD-002`.

## Protocol identity

- Protocol: `docs/WM_PLATONIC_MKNN_001_PROTOCOL.md`
- Config: `configs/experiments/wm_platonic_mknn_v1.json`
- Preregistration commit: `7392ab5`
- Implementation/provenance-fix commit used for the retained run: `01f93ab`
- Confirmation seeds: `131 / 137 / 139`
- Primary metric: `predictor_mknn(A,B)`
- Frozen floor: `2 * 5/127 ≈ 0.07874`
- Splits accessed: `train`, `confirmation`
- Downloads: none
- Stitching: not executed

A first execution under `e25f7b5` was discarded because provenance was
collected after writing the metrics file (`git_dirty: true`). Gates were
not changed. The retained run is the clean-tree rerun under `01f93ab`.

## Primary outcome

| seed | encoder A,B | predictor A,B | predictor vs shuffle | predictor vs random-map | seed gate |
|---:|---:|---:|---:|---:|---|
| 131 | 0.8906 | **0.8797** | 0.6969 | 0.0484 | pass |
| 137 | 0.9125 | **0.9156** | 0.8031 | 0.0156 | pass |
| 139 | 0.9266 | **0.9219** | 0.7375 | 0.0438 | pass |

Every confirmation seed satisfies the frozen conjunction: predictor A,B
beats shuffled-action, random-map, and `2×chance`.

Random-map controls sit near chance (`≈0.02–0.05`). Matched trajectory
hashes are identical within each seed for maps A and B. Observation-map
SHA-256 values are identical across seeds (maps are frozen).

## Required caveats (do not drop)

1. **Encoder geometry is already above the chance floor** in every seed
   (`encoder_geometry_already_above_chance_floor: true`). Predictor
   alignment is therefore largely inherited from two frozen linear maps of
   the same 4-d PointMass state, not from a newly discovered shared
   coordinate system.
2. Shuffled-action predictor m-kNN is **high**, not near chance
   (`0.70–0.80`). The pair still beats it, but this control does not show
   that action identity is the main source of neighborhood overlap.
3. Seed **137** confirmation **open-loop multi-step** latent MSE exploded
   (`≈1.38e16`) while one-step neighborhood overlap remained high. That
   quantity is `evaluate_latent_mse` (predictor unrolled from t=0 over the
   full action sequence), not one-step predictor MSE and not a decoder-only
   diagnostic. It does not falsify the frozen one-step `predictor_mknn`
   gate and is **not** a reason to retune k, maps, or gates.
4. Action-conditioned (frozen probe action) A,B overlap is also high and
   close to the shuffled probe scores. Report it separately; do not
   collapse it into the primary score.

## What this does not authorize

- `WM-PLATONIC-STITCH-001`
- `WM-LEFLOW-TRANSFER-001`
- `WM-CRCT-PLATONIC-COMPUTE-001`
- Relabeling HARD-002, V3, IBD-001, or Qwen confirmation
- Executing `LLM-QWEN-BINDING-ALGEBRA-004`

It does authorize continuing to `WM-LEFLOW-AMORTIZE-001` on integrity
grounds: `integrity_blockers` is empty. The scientific caveats above
remain in force.
