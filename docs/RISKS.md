# Risks

Status: `SCAFFOLDED`.

- `BLOCKED_RESOURCE`: CPU VPS has no GPU and limited free disk; real Qwen and published JEPA experiments must wait.
- `BLOCKED_RESOURCE`: Do not install Transformers or download GPT/Qwen weights in this run, despite contradictory runbook notes.
- `BLOCKED_EXTERNAL`: SkyJEPA remains blocked until official implementation assets are verified.
- `ACTIVE`: The repository currently depends on system NumPy for local no-install smoke code; an editable install will install NumPy in a venv when allowed.
- `ACTIVE`: All literature entries are prompt-derived until source verification occurs.
