# 004 pre-freeze graphs — PointMass physics and residual/skip MLP

Derived from frozen `step_pointmass` and `ActionDeltaPredictor`. 001/002/003
are not mutated.

## PointMass Euler (`dt=0.1`, `drag=0.05`, `mass=1`, `force_scale=1`)

Generator actions lie in `[-1, 1]`; clip is inactive on train/eval.

```text
vx'  = (1 - drag·dt)·vx + clip(ax)·dt
vy'  = (1 - drag·dt)·vy + clip(ay)·dt
x'   = x + vx'·dt
y'   = y + vy'·dt

Δvx = -drag·dt·vx + clip(ax)·dt
Δvy = -drag·dt·vy + clip(ay)·dt
Δx  = (vx + Δvx)·dt
Δy  = (vy + Δvy)·dt
```

Frozen causal dependency (`D` / `0`):

```text
          ax   ay   vx   vy
Δx         D    0    D    0
Δy         0    D    0    D
Δvx        D    0    D    0
Δvy        0    D    0    D
```

For primary **`ax → Δvx`**:

| class | channels |
|---|---|
| direct target | `Δvx` |
| downstream of ax (not a negative control) | `Δx` |
| independent negative controls | `Δvy`, `Δy` |

`Δy` and `Δvy` are the same orthogonal axis (`ay`/`vy`), not two independent
physics worlds. Both ratios are still required (conservative). 002 seed 73’s
specificity failure vs `Δy` remains valid under this ontology and is not
excused.

## Residual MLP (`H=6`)

```text
z  = tanh(enc2(tanh(enc1(state))))     encoder; not searchable
e  = tanh(act1(action)) @ q_act        searchable act_0..5

h0 = tanh(mix([z, e]))                 residual-stream start; path site only

hid1 = tanh(b1_w1(h0)) @ q_b1          searchable b1_0..5
F1   = b1_w2(hid1)                     residual branch 1
h1   = skip1 + F1                      skip1 defaults to h0

hid2 = tanh(b2_w1(h1)) @ q_b2          searchable b2_0..5
F2   = b2_w2(hid2)                     residual branch 2
h2   = skip2 + F2                      skip2 defaults to h1

Δ = Linear(h2)
```

Searchable CRCT sites: `act_*`, `b1_*`, `b2_*` only.
