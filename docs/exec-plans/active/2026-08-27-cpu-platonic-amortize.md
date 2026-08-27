# Execution plan — 2026-08-27 Track A CPU Platonic/amortize

Authoritative continuation of `1a1340a` on `crct-stage0-001`.

## Verified baseline

- Branch: `crct-stage0-001`
- HEAD at continuation start: `1a1340a76b22614b5c2e652d28989bd8ea168e2f`
- Origin contains `1a1340a`
- GitHub Actions `cpu-ci` on `1a1340a`: **failure** (collection `yaml` / `h5py`) — not green
- Local CI collection fix committed as `807b735`, not yet on origin, CI not re-observed
- Model routing: UNVERIFIED for Codex Sol High/Luna

## Frozen invariants (untouched)

- HARD-002 `NEGATIVE_RESULT` seeds 1009/2027/4093
- V3 `INELIGIBLE_TASK_PHASE0`
- Confirmation passed; 004 `execution_authorized: false` (not run)
- IBD-001 smoke; IBD-002 not run
- No DINO-WM / LeWM download
- No stitching (`WM-PLATONIC-STITCH-001` not opened)

## Decisions

### D-CI-001

Repair CPU collection with PyYAML in `[dev]`, `h5py` importorskip, Qwen tokenizer `@pytest.mark.gpu`, offline hub env. Qwen forwards remain CUDA-only.

### D-T0-001 — substrate

PointMass2D (existing tiny JEPA dynamics). MiniPush pixels are not required for this CPU control.

### D-T0-002 — m-kNN gate (pre-outcome)

`k=5`, `n_eval=128`. Chance reference `k/(n_eval-1)`. Pass iff every confirmation seed has predictor m-kNN(A,B) strictly greater than shuffled-action, random-map, and `2 * chance`. Evidence level Availability. Fail closed: `NEGATIVE_RESULT`.

### D-T0-003 — amortize gate (pre-outcome)

H=5 primary. Success = true position MSE `< 0.15`. Amortized latent N=64 must be no worse than random-shooting by 0.05 success and faster in wall-clock. H=10 is diagnostic; collapse is an allowed outcome, not a retune trigger.

## Checklist

- [x] Verify git/invariants
- [x] Observe CI failure on `1a1340a`
- [x] Commit CI collection/GPU gate (`807b735`)
- [x] Preregister T1/T2 (no results) — commit `7392ab5`
- [ ] Implement + semantic tests
- [x] Implement + semantic tests (`e25f7b5`, provenance fix `01f93ab`)
- [x] Execute T1
- [x] Adjudicate T1 (`TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED`; encoder caveat)
- [x] Execute T2 if T1 adjudication is integrity-clean
- [x] Adjudicate T2 (`NEGATIVE_RESULT`; evidence_level None)
- [ ] Review, suite, docs
