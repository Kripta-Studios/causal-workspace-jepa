# 003 pre-freeze graphs — PointMass physics and residual/skip MLP

Parent 002 is not mutated. These graphs are derived from already-frozen
code: `step_pointmass` and `ActionDeltaPredictor.forward_intervene`.

## PointMass Euler (`dt=0.1`, `drag=0.05`, `mass=1`, `force_scale=1`)

Actions in the generator already lie in `[-1, 1]`, so the clip is inactive
on the training/eval distribution.

```text
a_x' = clip(ax, -1, 1)
a_y' = clip(ay, -1, 1)
vx'  = (1 - drag·dt)·vx + a_x'·dt
vy'  = (1 - drag·dt)·vy + a_y'·dt
x'   = x + vx'·dt
y'   = y + vy'·dt
```

Therefore one-step targets:

```text
Δvx = -drag·dt·vx + a_x'·dt          # direct in ax, drag term in vx
Δvy = -drag·dt·vy + a_y'·dt
Δx  = vx'·dt = (vx + Δvx)·dt         # downstream of Δvx (hence of ax)
Δy  = (vy + Δvy)·dt                  # downstream of Δvy (hence of ay)
```

Frozen causal dependency of Δ-outputs on inputs (`D` = depends, `0` = not):

```text
          ax   ay   vx   vy   x   y
Δx         D    0    D    0   0   0
Δy         0    D    0    D   0   0
Δvx        D    0    D    0   0   0
Δvy        0    D    0    D   0   0
```

For primary map **`ax → Δvx`**:

| class | channels |
|---|---|
| direct target | `Δvx` |
| downstream of ax | `Δx` (not a negative control) |
| independent negative controls | `Δvy`, `Δy` |

`Δy` is independent of `ax` under this integrator. 002’s seed-73
`SPECIFICITY_FAILED` (`nec_Δvx / nec_Δy < 2`) therefore remains a valid
specificity failure under this ontology; 003 does not excuse it.

A priori scale: `|a_x·dt| ~ 0.1` vs `|drag·dt·vx| ~ 0.001`. Action
dominates `Δvx`. A skip-heavy `ax → Δvx` path would be physically
unsurprising. That is a result to report, not a reason to drop residual
tests.

## Residual MLP computational graph (`H=6`)

```text
state  → enc1 → tanh → enc2 → tanh → z          (encoder; not searchable)
action → act1 → tanh → e  (× q_act)             searchable: act_0..5

h0 = tanh(mix([z, e]))                          residual stream start
                                                (not independently searchable)

hid1 = tanh(b1_w1(h0))  (× q_b1)                searchable: b1_0..5
h1   = h0 + b1_w2(hid1)                         skip = h0; branch = b1_w2(hid1)

hid2 = tanh(b2_w1(h1))  (× q_b2)                searchable: b2_0..5
h2   = h1 + b2_w2(hid2)                         skip = h1; branch = b2_w2(hid2)

Δ = Linear(h2)                                  (Δx, Δy, Δvx, Δvy)
```

Path interventions (not added to the CRCT search set):

- **skip_only:** patch candidate sites from B into A; hold `hid1` and `hid2`
  at A. New mix/skip can flow; residual branches stay at A.
- **residual_only:** patch candidate sites; hold only `skip1=h0_A`. Residual
  branches recompute; block 2 sees residual-updated `h1`. Do not freeze
  `skip2` to `h1_A` while `hid2` still depends on a different `h1`.

CRCT search remains `act_*`, `b1_*`, `b2_*` only. Encoder remains excluded.
Residual-stream coordinates are path-intervention sites, not localization
sites.
