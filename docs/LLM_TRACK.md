# LLM Track

Status: `SMOKE_VALIDATED`.

The inherited CPU implementation uses a mock transformer with known activation dependencies. It is valid for interface, leakage, and intervention-pipeline tests only.

Real Hugging Face Qwen instrumentation and bounded Qwen3-0.6B experiments are `SMOKE_VALIDATED` on
the RTX 5070 Ti host. Tests cover stable selected sites, deterministic capture, Torch interventions,
donor/statistic registration, and autograd. Bounded selected-site Qwen3-4B capture also passes.
Ollama is not a hidden-state or autograd source.

The primary-scale capture script is now implemented for `Qwen/Qwen3-4B` at immutable revision
`1cfa9a7...`. Its repository estimate is 8,060,926,626 bytes; it captures five residual sites and
three selected positions over 12 fixed prompts into resumable checksummed HDF5 under a 64 MB output
budget. It ran from clean commit `55087ea`: exact revision, 180 rows, 574,308 stored bytes, and all
budget/checksum gates pass. This establishes selected-site Availability only.

Validated Qwen instrumentation smoke:

- `LLM-QWEN-001` ran at pinned revision `c1899de...` from clean commit `0d6a37b`.
- Hooked replay was exact; a selected-logit gradient with respect to layer-14 residual state had norm
  `0.944`; zero, mean, donor patch/resample, and steering all changed final hidden state and logits.
- This meets the real-Hugging-Face-instrumentation boundary only. It does not validate Intervention-
  JEPA, circuit ranking, behavior change, feature meaning, or a workspace.

Dataset continuation:

- `LLM-INTDATA-001` is preregistered for 432 direct Qwen outcomes across three sites, five operation
  classes, disjoint 8/2/2 prompt splits, split-local donors, and training-only target selection.
- Every example includes a separately executed 5-percent local-linear approximation. Shards are
  ignored HDF5; the checksum manifest and summarized causal-data metrics are committed after a run.
- Status is `SMOKE_VALIDATED` from clean commit `0aa80ac`: 432 outcomes, 17 top-token changes, one
  verified 412 KB shard, and mean local-linear MSE `139.83`.

Meta-model continuation:

- `LLM-IJEPA-001` fixes prompt, coordinate, and operation holdouts and trains three neural
  Intervention-JEPA seeds against nine baselines.
- The runner saves/reloads checkpoints, fits a sparse dictionary on training context only, and
  directly re-executes all 16 coordinate predictions on four new prompts before writing a candidate
  graph.
- Status is `SMOKE_VALIDATED` from clean commit `a54f2ed`. The registered H-LLM-01/02/03 gates pass
  on all three seeds. The nearest-neighbor baseline narrowly wins resample-holdout MSE, and direct
  precision@1 is zero; H-LLM-06 fails and the candidate graph is `REJECTED`.

Current mock implementation:

- exposes `resid_pre`, `attn_out`, `mlp_out`, `resid_post`, and `logits`;
- supports direct replayable interventions through the shared `InterventionSpec`;
- trains a small intervention-conditioned JEPA-style ridge predictor against no-change, mean-effect, and linear-context baselines.

Validated CPU mock smoke:

- `LLM-MOCK-001` ran with code commit `85c1dbfbe9c824bcca415af13f4a6f34acc95267`.
- Intervention-JEPA MSE: `1.32e-07`; no-change: `0.002113`; mean-effect: `0.002113`; linear-context: `0.002113`.
- Effect correlation: `0.99997`.
- This is not Qwen evidence.

Validated GPT-2 Medium smoke:

- `LLM-GPT2-001` loaded `gpt2-medium` through Hugging Face Transformers in `.venv`.
- Direct residual steering at `transformer.h.12.resid_post` changed downstream logits with mean absolute logit delta `0.0797`.
- Tiny intervention-JEPA MSE was `0.00220` vs no-change `0.0114` on a 4-example held-out smoke split.
- This is causal-mediation smoke evidence for a directly edited residual coordinate, not a J-space or workspace discovery.

Strengthened GPT-2 Medium study:

- `LLM-GPT2-002` ran from clean commit `8fbab8c` and generated a checksummed 288-outcome dataset.
- It batches 288 direct residual interventions across eight prompts, layers `6/12/18`, two
  coordinates, and six magnitudes while storing only selected positions and outputs.
- Prompt, magnitude, and layer holdouts are explicit; training uses six prompts, layers `6/12`, and
  magnitudes through `0.5`, while evaluation uses two unseen prompts, magnitude `1.0`, and a separate
  layer-18 stress split.
- Required baselines include a prompt-local finite-difference Jacobian, corpus-averaged Jacobian,
  linear and bilinear regressions, trained MLP, nearest neighbor, and sparse-context transport.
- On held-out prompts/magnitude, local-Jacobian MSE was `7.79e-7`, bilinear Intervention-JEPA
  `0.003499`, and no-change `0.006461`. The bilinear model captured some context-dependent effect but
  did not beat the strong local baseline.
- On held-out layer 18, bilinear MSE `0.01009` was worse than no-change `0.006228`; cross-layer
  transfer failed. The MLP also failed and no intervention changed the top token.
- The model loaded from the existing local cache only. This remains selected-output GPT-2 causal
  smoke evidence, not Qwen, behavior-level, feature-semantic, or workspace evidence.

Semantic-composition follow-up:

- `LLM-GPT2-003` constructs two orthogonal residual contrast directions from eight calibration
  prompts that never appear in train or test evaluation.
- It executes 72 outcomes at block 12: eight single-direction settings and four signed compositions
  for each of six prompts. Predictors train on singles from four prompts and are tested on
  compositions from two unseen prompts.
- Prompt-local finite difference, corpus transport, and direct addition of large single effects are
  mandatory baselines. Construction labels are not treated as validated feature semantics.
- The clean run finished in `392.85` seconds with a 24,933-byte checksummed shard. Composition
  interaction was only `0.043%` of effect power on held-out prompts. Prompt-local Jacobian MSE was
  `0.000990`, but MLP/bilinear MSE was `0.725`/`1.346`, both worse than no-change `0.418`.
- Bilinear prediction appeared excellent on seen-prompt compositions (`0.00110` MSE, `0.9985`
  correlation) and then failed with negative correlation on unseen prompts. All registered
  hypotheses failed. Two of 72 interventions changed the top token; this is sparse output mediation,
  not evidence that the constructed directions have the proposed semantics.
