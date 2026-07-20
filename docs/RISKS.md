# Risks

Status: `ACTIVE`.

- `BLOCKED_RESOURCE`: CPU VPS has no GPU and limited free disk; real Qwen and published JEPA experiments must wait.
- `ACTIVE`: GPT-2 Medium was downloaded and run after explicit user override; weights are under `.cache/` and ignored by Git.
- `BLOCKED_RESOURCE`: Do not download Qwen weights in this run.
- `BLOCKED_EXTERNAL`: SkyJEPA remains blocked until official implementation assets are verified.
- `ACTIVE`: The repository currently depends on system NumPy for local no-install smoke code; an editable install will install NumPy in a venv when allowed.
- `ACTIVE`: Four central literature entries have now been checked against their primary web/arXiv
  sources; the remaining registry entries are still prompt-derived.
- `ACTIVE`: `uv` is not installed on this VPS. `scripts/bootstrap_cpu.sh` reports the install command and exits with `SKIPPED_RESOURCE` rather than modifying the toolchain automatically.
- `RESOLVED`: an earlier handoff reported `/root/.cache/pip` at about 4.4 GB. The 2026-07-20
  resource re-audit measured all of `/root/.cache` at about 233 MB; no cache deletion was needed.
- `ACTIVE`: GPT-2 Medium CPU runs are slow; the committed smoke uses 4 prompts, 2 residual coordinates, and 16 direct interventions.
- `RESOLVED`: adversarial review found that the original `WM-T0-002` action patch assigned the donor
  target directly. The corrected intervention was executed from clean commit `315d8cf`; the original
  metric is superseded.
- `ACTIVE`: a five-consumer subspace in the tiny model may simply recover its four-dimensional
  physical-state manifold. High-variance PCA is therefore a mandatory matched control.
- `ACTIVE`: `WM-T0-003` confirmed this concern: PCA ablation was more damaging than the proposed
  subspace. The current OOD-uncertainty readout also failed held-out prediction.
- `REJECTED`: arbitrary random latent projection as a multistep control on the current linear model.
  Some controls cause extreme off-manifold rollout divergence. Future controls must match activation
  density or use in-manifold donor/resampling interventions.
- `RESOLVED`: conditional donor candidate/random patches in `WM-T0-004` stayed near the empirical
  activation bank and yielded 63/64 and 64/64 matched controls. PCA patches were correctly rejected
  as unmatched because their perturbations were much larger.
- `ACTIVE`: `WM-T0-004` is one seed on simple PointMass physics. Its null blocks claims for this run;
  it does not prove that stronger goal-conditioned or multi-task JEPAs lack workspace-like structure.
- `RESOLVED_NULL`: `WM-T0-005` tested richer synthetic task context on three seeds. Its post-hoc
  consumers did not generalize to the held-out composition and task-counterfactual recovery was
  negative, so no shared state/task manifold is promoted to a workspace candidate.
- `ACTIVE`: `LLM-GPT2-002` is larger than the original smoke but still has one seed, eight local
  prompts, coordinate interventions, and selected outputs. Its local Jacobian uses extra direct
  small-magnitude executions and must be compared on fidelity, not runtime.
- `ACTIVE`: `LLM-GPT2-002` took `647.85` seconds, exceeding the CPU profile's 10-minute expectation
  by `47.85` seconds. The result is retained, but this configuration must not be described as
  within the runtime guard.
- `ACTIVE`: `LLM-GPT2-003` calls its directions semantic only by contrast construction. Lexical,
  syntax, and final-token differences remain confounds, so labels cannot support feature claims.
  The run is limited to 12 batched interventions and must abort at the 600-second CPU guard.
