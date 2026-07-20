# SUMMARY

## 2026-07-20

### Done

- Read `AGENTS.md` and `VPS_RUNBOOK.md` fully.
- Confirmed current machine is the CPU VPS path: no GPU, 4 CPU cores, about 7 GB free at start.
- Added package metadata, resource profiles, package tree, resource guard, doctor CLI, typed interfaces, provenance helpers, and docs skeleton.
- Added standard-library unit/integration tests for config parsing, resource guards, CLI output, intervention serialization, and reproducibility scaffolding.
- Validated Milestone 0 with `doctor`, `python -m unittest discover -s tests -p 'test_*.py'`, `scripts/audit_reproducibility.py`, and `git diff --check`.

### Discovered

- `README.md` initially only contained the project title.
- `pytest` is not installed, but NumPy is available system-wide.
- The runbook contains a GPT-2/GPT2-medium note that conflicts with the user instruction not to download weights or install Transformers. For this run it is treated as `BLOCKED_RESOURCE`.

### Next Ideas

- Milestone 1 should add deterministic Tier 0 generators and tiny NumPy JEPA smoke training.
- Milestone 2 should add intervention/probe/circuit primitives and the mock Qwen pipeline.
