# CRCT-COALITION-IBD-002 protocol

Status: `PREREGISTERED_NOT_RUN`.

`CRCT-COALITION-IBD-001` remains a synthetic evaluator smoke. Independent
review found its recorded gauge Spearman used an untransformed copy of the
plant. That caveat is frozen; IBD-001 confirmation is **not** re-run or
relabeled.

IBD-002 is the prospective successor:

1. Development seeds: `21, 23, 29` (not HARD-002, not IBD-001).
2. Freeze code/thresholds/gauge implementation in a commit.
3. Confirmation seeds, chosen before generation: `941, 947, 953`.
4. Commit the freeze.
5. Generate confirmation plants once and adjudicate.

Required gauge: apply a compensated `known_w *= s`, `known_r /= s`
reparameterization, recompute contributions, and require the same equivalent
minimal circuits. An uncompensated `known_w *= s` must change known energy.

Do not reuse seeds `1009, 2027, 4093, 11, 13, 17, 811, 823, 829`.
