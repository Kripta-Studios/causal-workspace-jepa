# 005 graphs — PointMass physics and identifiable residual edges

004 physics graph is unchanged. 001–004 are not mutated.

## PointMass Euler (same as 004)

Primary `ax → Δvx`. Independent controls: `Δvy`, `Δy`. `Δx` is downstream
of `ax`, not a negative control.

## Residual MLP and edge ontology (`H=6`)

```text
z  = tanh(enc2(tanh(enc1(state))))     encoder; not searchable
e  = tanh(act1(action)) @ q_act        searchable act_*  (may enter V)

h0 = tanh(mix([z, e]))                 s1 default; not a searchable node

r1 = b1_w2(tanh(b1_w1(h0)) @ q_b1)     F1 message; hid1 searchable as b1_*
h1 = s1 + r1                           s1 defaults to h0

r2 = b2_w2(tanh(b2_w1(h1)) @ q_b2)     F2 message; hid2 searchable as b2_*
h2 = s2 + r2                           s2 defaults to h1

Δ = Linear(h2)
```

Stage A searchable nodes: `act_*`, `b1_*`, `b2_*`.

Stage B edges (cached messages from A and V-patched P):

```text
E_skip1 : s1_P, r1_A, r2_A     (skip1 CF; residuals factual)
E_F1    : s1_A, r1_P, r2_A     (F1 message then additive stream)
r2_P    : diagnostic only; not a Level-3 F2 edge
```

`r1_P` / `r2_P` are **cached from the P forward**, not `F2(hybrid h1)`.
