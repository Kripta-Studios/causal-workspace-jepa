# Decisions

## 2026-07-21

- Activate `gpu_12gb` after detecting an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM, CUDA-enabled
  PyTorch, approximately 370 GB free, and a passing GPU doctor check.
- Treat the old VPS resource blocks as historical. Qwen and published-JEPA work is now `ACTIVE`, but
  remains unvalidated until real direct runs produce committed metrics from clean code.
- Begin Qwen with Qwen3-0.6B to validate hooks, storage, interventions, and Windows portability before
  Qwen3-4B. Keep Qwen3-30B-A3B restricted to `gpu_cluster`.
- Normalize recorded relative metric paths in the reproducibility audit so Linux provenance can be
  checked on Windows. A fresh clone may skip ignored shard checksums, but any available shard must
  still match exactly.
- Pin the first Qwen target to public revision `c1899de289a04d12100db370d81485cdf75e47ca`
  and pass `token=False`; an invalid implicit local OAuth token returned HTTP 401 for a public model,
  and repository experiments must neither depend on nor modify user credentials.
- Require explicit training-split statistics or registered donors for Torch mean/resample operations.
  Batch-one self-means and self-resampling are invalid controls.
- Store Qwen intervention data as checksummed HDF5 shards, not monolithic NPZ. Keep prompt/donor
  split metadata in every record and choose output-logit coordinates using training prompts only.
- Attach a direct 5-percent finite-difference execution to every outcome so learned predictors must
  compete with a prompt-local baseline in the later evaluation.
- Freeze the observed Qwen dataset and test meta-models on three incompatible holdouts: unseen
  prompts, an unseen steering coordinate, and an unseen operation. Require two-of-three seed
  replication for numbered hypothesis decisions.
- Treat the direct-verification graph as a ranked-coordinate candidate, not a circuit, unless later
  necessity/sufficiency/faithfulness/minimality tests pass.
- Retain the unchanged `LLM-IJEPA-001` preregistered decisions: H-LLM-01/02/03 pass all three seeds,
  but reject the graph because H-LLM-06 precision@1 fails. Do not revise the gate after seeing that
  nearest-neighbor narrowly beats the model on resampling-holdout MSE.
- Integrate LeWorldModel first as a faithful small reproduction at official revision `8edfeb3...`
  instead of claiming benchmark equivalence without installing the larger external framework and
  datasets. Preserve its two-loss recipe and record every deliberate scaling difference.
- Do not checkpoint-select LeWM by prediction MSE alone: engineering checks showed that this rewards
  collapsed embeddings despite a high SIGReg penalty. Use the preregistered final training step.
- Reconstruct only the restricted action-embedding to final-predictor-site to planner graph. Since
  AdaLN action conditioning enters every predictor block, earlier blocks are alternatives rather
  than falsely asserted serial nodes.

## 2026-07-20

- Use only the `cpu_vps` resource path for this run.
- Treat all new model-weight downloads, external datasets, CUDA, Docker, Conda, and large simulation
  suites as unavailable.
- Implement CPU smoke code with NumPy and standard-library infrastructure.
- Keep real Qwen and published JEPA checkpoints `BLOCKED_RESOURCE` until `gpu_12gb`.
- A later explicit user override permitted cached GPT-2 Medium and the minimal Transformers runtime
  under the final `AGENTS.md` CPU limits. It does not permit Qwen downloads or GPU claims.
- Use `SUMMARY.md` as the handoff log after each milestone.
- Preserve the original `WM-T0-002` artifact but withdraw its specificity interpretation: its
  “patch” result was assigned directly to the donor target.
- Validate workspace discovery against known shared and disjoint controls before applying it to a
  learned representation.
- Treat a compact five-consumer sensitivity subspace as a candidate only. It is not a workspace
  without direct ablation, PCA/random controls, controllability, selective necessity, and
  generalization.
- For `LLM-GPT2-002`, use direct held-out outcomes and compare against a prompt-local Jacobian. Do
  not infer a nonlinear advantage from beating only no-change or mean-effect.
- For `WM-T0-004`, freeze the deep predictor before fitting consumers, calibrate ensemble uncertainty
  on validation only, and replace projection-out controls with conditional donor resampling that
  explicitly measures activation-manifold distance.
- Do not call a shared Jacobian eigensubspace J-space. Anthropic defines J-space as a sparse
  nonnegative token-aligned frame and validates report, modulation, reasoning, reuse, and selectivity;
  the JEPA detector currently tests only a narrower functional analogue.
- Preserve the `WM-T0-004` null without threshold tuning. Conditional donor controls are retained as
  a useful method; the observed architecture is rejected as a workspace-discovery target because
  action dependence and uncertainty consumers fail before the workspace decision.
- For `WM-T0-005`, cross goals with dynamics modes and hold out one full composition. Require two of
  three seeds, random and local-tangent specificity, at least 50 percent task-counterfactual
  recovery, and a selective-necessity ratio of at least 1.25. Keep `workspace_found` false even if
  the narrower candidate passes.
- Preserve the `WM-T0-005` null without threshold or seed tuning. A future CPU JEPA study must change
  the architecture materially by jointly training task consumers or a planner; adding more post-hoc
  probes to the same representation is not a meaningful continuation.
- For `LLM-GPT2-003`, derive directions only from disjoint calibration prompts, orthogonalize them,
  train predictors on singles only, and reserve all composed targets for evaluation. Require direct
  large-single addition and prompt-local finite differences so a learned model cannot win against
  weak baselines alone.
- Preserve the `LLM-GPT2-003` null. Seen-prompt bilinear success is memorization evidence, not
  composition generalization. Do not add prompts or tune this observed grid; a continuation must
  change intervention class to replacement, resampling, patching, or feature clamping.
