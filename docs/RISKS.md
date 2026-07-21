# Risks

Status: `ACTIVE`.

- `RESOLVED_RESOURCE`: the current host has an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM and about
  370 GB free; the historical CPU-VPS blocker no longer applies.
- `ACTIVE`: GPT-2 Medium was downloaded and run after explicit user override; weights are under `.cache/` and ignored by Git.
- `ACTIVE`: bounded Qwen3-0.6B/4B downloads are permitted by `gpu_12gb`, but storage and VRAM must be
  estimated first and all-layer/all-token capture remains prohibited.
- `MITIGATED`: the host has an invalid implicit Hugging Face OAuth token for public API calls. Qwen
  loaders explicitly use `token=False`; no token value was printed, changed, stored, or committed.
- `RESOLVED`: `LLM-QWEN-001` validates real pinned Qwen3-0.6B hooks/interventions/autograd from clean
  code. The result remains instrumentation smoke and cannot substitute for held-out meta-model or
  circuit evidence.
- `RESOLVED`: the first Qwen run recorded `git_dirty=true` because provenance was collected after
  creating its output. Those two regenerable files were discarded; ordering was fixed in `0d6a37b`,
  and only the clean rerun is reported.
- `ACTIVE`: the Qwen dataset generator accesses adapter donor/statistic registries to compute the
  exact source edit without using downstream targets. This is tested but should become a public
  adapter method before broader reuse.
- `ACTIVE`: 432 outcomes cover two small prompt families and selected residual coordinates/sites;
  even a clean run is smoke-scale and cannot establish broad Qwen mechanism generalization.
- `ACTIVE`: local-linear MSE `139.83` is aggregated across projected hidden/logit targets and very
  different edit classes. It indicates a hard/off-local regime but is not itself a nonlinear-model win.
- `BLOCKED_EXTERNAL`: SkyJEPA remains blocked until official implementation assets are verified.
- `ACTIVE`: the current Windows system Python is 3.14 with CUDA PyTorch 2.10 and Transformers 5.3.
  A project-local reproducible environment/lock must be repaired before this host is a clean install target.
- `ACTIVE`: Four central literature entries have now been checked against their primary web/arXiv
  sources; the remaining registry entries are still prompt-derived.
- `RESOLVED`: `uv` is installed and the previously empty `uv.lock` has been regenerated from the
  declared dependency groups. Actual experiment provenance must still record the running versions.
- `RESOLVED`: the reproducibility audit compared POSIX provenance strings to Windows-rendered paths;
  normalized relative paths now pass on both platforms.
- `ACTIVE`: ignored GPT-2 data shards from the Linux host are absent on this machine. Their manifest
  records are audited as skipped, not verified; this is not a checksum pass for the missing bytes.
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
- `RESOLVED_NULL`: `LLM-GPT2-003` finished in `392.85` seconds. Lexical, syntax, and final-token
  confounds remain, so its constructed direction labels do not support semantic feature claims.
  Large compositions were additive and all learned held-out-prompt predictions failed.
