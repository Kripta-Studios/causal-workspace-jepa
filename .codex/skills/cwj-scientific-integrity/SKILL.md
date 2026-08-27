---
name: cwj-scientific-integrity
description: Enforce prospective scientific integrity for causal-workspace-jepa. Use when touching experiments, datasets, splits, thresholds, seeds, Qwen forwards, CRCT metrics, result adjudications, provenance, baselines, or scientific documentation.
---
Read `../../../docs/codex-handoff/02_SCIENTIFIC_INVARIANTS.md`.

Before any outcome-sensitive action, state:
- data already opened;
- data still protected;
- choices already frozen;
- choices still legally changeable;
- required commit/hash boundary.

Refuse internally to:
- tune on confirmation/test outcomes;
- relabel a frozen negative result;
- fabricate provenance;
- drop an unfavorable fair baseline;
- use candidate-only accuracy as competence eligibility;
- rerun HARD-002 primary seeds to validate a redesigned metric.

Record experiment access and adjudication mechanically where the repo supports it.
