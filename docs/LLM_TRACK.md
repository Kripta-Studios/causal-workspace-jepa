# LLM Track

Status: `COMPLETED_NEGATIVE` for the corrected Qwen nonlinear-advantage audit; the broader research
track remains active.

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
- Original status was `SMOKE_VALIDATED` from clean commit `a54f2ed`. The registered gates passed
  against the then-specified comparators, but H-LLM-01 is now `WITHDRAWN`: the so-called local
  Jacobian was a BF16 one-sided secant and can be a numerical-noise floor. The nearest-neighbor
  baseline narrowly wins resample-holdout MSE, and direct
  precision@1 is zero; H-LLM-06 fails and the candidate graph is `REJECTED`.
- The neural model used there is a supervised two-branch conditional bottleneck. It lacks a target
  encoder, stop-gradient/EMA target, and JEPA anti-collapse objective, so current documentation no
  longer treats the class name as architectural evidence that it is a genuine JEPA.

Exact-derivative corrective milestone:

- `LLM-QWEN-JVP-AUDIT-001` is implemented and preregistered but not yet executed. It replays the
  immutable 432-record grid in float32/eager attention, reconstructs every semantic edit as a dense
  residual direction, computes exact autograd directional JVPs, and checks them against six
  symmetric central-difference scales plus a quadratic Taylor baseline.
- The old H-LLM-01 result is retained only if the numerical audit passes and at least two of three
  refitted conditional-bottleneck seeds beat exact JVP and quadratic controls under the frozen raw
  and semantic-deduplicated gates. A negative result will withdraw the old claim.
- V1 executed from clean commit `686368e` and is `REJECTED` on a numerical gate: all JVP/central convergence
  gates passed, but semantic downstream endpoint max error was `1.335e-4` versus `1e-5`. Its
  preliminary exact-JVP MSE `0.6143` beat the conditional bottleneck `3.1899`; quadratic Taylor was
  `0.07870`, and zero learned seeds beat exact JVP. H-LLM-01 remains unresolved until a separately
  registered source-semantic check passes.
- V2 is implemented/preregistered as a post-diagnostic confirmation. It adds source captures and
  replaces only v1's invalid downstream absolute identity test with exact direct-source equality
  plus a two-rounding float32 bound. All scientific score and disposition gates are unchanged.
- V2 completed from clean commit `a779ff6`: every numerical gate passed, finite-amplitude
  nonlinearity failed the registered floor, and zero of three learned seeds beat exact JVP or
  quadratic Taylor. Exact JVP/quadratic/bottleneck raw MSE was `0.6143`/`0.07870`/`3.1899`.
  Restricted H-LLM-01 is withdrawn. The next dataset must target actual semantic behavior changes,
  and the next learned model must implement a genuine target-encoder JEPA objective.

Behavior-changing continuation:

- `LLM-CAPITAL-PATCH-001` fixes 36 one-token capital facts and entity-disjoint 24/6/6 splits. It
  applies every ordered within-split full residual donor patch at layer 21 and records direct
  full-vocabulary answer transfer plus exact JVP and quadratic controls.
- Layer selection used four excluded calibration countries only. The final roster was inspected for
  tokenizer token count, never model success. The clean `95018cb` run passed all eligibility gates:
  93.6% top-token change and 50% test donor-answer transfer. Exact-JVP/quadratic candidate agreement
  was 35.1%/74.3%. The next study may therefore fit a genuine target-encoder JEPA, but this dataset
  result alone is only direct causal mediation.

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
