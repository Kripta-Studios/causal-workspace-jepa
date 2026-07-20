# Risks

Status: `SCAFFOLDED`.

- `BLOCKED_RESOURCE`: CPU VPS has no GPU and limited free disk; real Qwen and published JEPA experiments must wait.
- `ACTIVE`: GPT-2 Medium was downloaded and run after explicit user override; weights are under `.cache/` and ignored by Git.
- `BLOCKED_RESOURCE`: Do not download Qwen weights in this run.
- `BLOCKED_EXTERNAL`: SkyJEPA remains blocked until official implementation assets are verified.
- `ACTIVE`: The repository currently depends on system NumPy for local no-install smoke code; an editable install will install NumPy in a venv when allowed.
- `ACTIVE`: All literature entries are prompt-derived until source verification occurs.
- `ACTIVE`: `uv` is not installed on this VPS. `scripts/bootstrap_cpu.sh` reports the install command and exits with `SKIPPED_RESOURCE` rather than modifying the toolchain automatically.
- `ACTIVE`: `/root/.cache/pip` is about 4.4 GB but was not modified because it may predate this task. Free disk remains above the 4 GB guard.
- `ACTIVE`: GPT-2 Medium CPU runs are slow; the committed smoke uses 4 prompts, 2 residual coordinates, and 16 direct interventions.
- `ACTIVE`: adversarial review found that the original `WM-T0-002` action patch assigned the donor
  target directly. Its artifact is preserved, but the Specificity interpretation is withdrawn until
  the repaired intervention run is executed from committed code.
- `ACTIVE`: a five-consumer subspace in the tiny model may simply recover its four-dimensional
  physical-state manifold. High-variance PCA is therefore a mandatory matched control.
- `ACTIVE`: `WM-T0-003` confirmed this concern: PCA ablation was more damaging than the proposed
  subspace. The current OOD-uncertainty readout also failed held-out prediction.
- `REJECTED`: arbitrary random latent projection as a multistep control on the current linear model.
  Some controls cause extreme off-manifold rollout divergence. Future controls must match activation
  density or use in-manifold donor/resampling interventions.
