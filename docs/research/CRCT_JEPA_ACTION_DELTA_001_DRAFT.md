# CRCT-JEPA-ACTION-DELTA-001 — DRAFT ONLY

```text
STATUS:                 DRAFT_NOT_PREREGISTERED
EXECUTION_AUTHORIZED:   false
CONFIRMATION:           CLOSED
TRAINING:               NOT TO BE RUN UNDER THIS DOCUMENT
DOWNLOADS:              none
```

Parent: `CRCT-COALITION-IBD-003` `MECHANISM_RECOVERY_PASSED` (synthetic IBD).
This file is **not** a protocol freeze and is **not** scientific evidence.

HARD-002 remains `NEGATIVE_RESULT`. IBD-002 remains not executed.
Qwen 004, stitching, Reachable-003, Rectified Flow, DINO-WM, and LeWM
remain unauthorized.

## Question (for a future freeze)

Can CRCT discover and causally verify the internal mechanism by which a
**trained** tiny neural world-model predictor implements

```text
f_theta(z_t, a_t) -> Delta z_t   (or z_{t+1})
```

on known linear PointMass physics? Decoding `ax` from activations is not
the claim. The claim is which components **causally implement** each map.

## Proposed substrate

Small learned CPU model, not ridge/identity TinyJEPA:

- encoder (tiny MLP)
- predictor with at least two residual blocks
- identifiable sites: block MLPs and, if present, attention heads

External physics (known, not planted named routes):

```text
state = (x, y, vx, vy)
action = (ax, ay)
```

Mechanistic targets (separate circuits, not one global ablate):

```text
ax -> Delta vx
ay -> Delta vy
vx -> Delta x
vy -> Delta y
```

## Required tests per hypothesized mechanism M (future freeze)

1. Localization of candidate sites (label-blind).
2. Necessity: ablating M damages the target map more than a matched control
   map (e.g. `ax` circuit vs `Delta vy`).
3. Sufficiency: restoring/patching M recovers the target effect.
4. Counterfactual action: `ax -> ax'` holding other state fixed; M must
   change in the direction that explains `Delta vx(ax') - Delta vx(ax)`.
5. Specificity: not global network destruction.
6. Redundancy: search alternate coalitions.
7. Cancellation: signed opposing routes.
8. Gauge: functional claim survives allowed reparameterization.

Planning, if used at all, is a **downstream readout** after a contact
mechanism exists (later tier), not an optimization target here.

## Explicit non-actions

Do not train, do not freeze this ID, do not download weights, do not open
confirmation. Next authorized step after this draft is a **new** freeze
commit, not this file.
