# Baselines

Status: `SMOKE_VALIDATED` for implemented CPU/Qwen methods; a published-model small reproduction is
`IMPLEMENTED_UNVALIDATED`.

## World Model

- oracle/raw-state dynamics;
- simple autoencoder dynamics;
- no-action JEPA;
- shuffled-action JEPA;
- frozen random encoder;
- linear latent dynamics;
- LeWorldModel;
- JEPA-WMs;
- Delta-JEPA objective;
- C-JEPA;
- local Jacobian;
- probe-only interpretation;
- matched random-subspace intervention.

`WM-T0-003` implements equal-dimensional random subspaces, high-variance PCA, a planted shared
subspace positive control, and disjoint-consumer negative control. The local hardware blocker for
published baselines is removed.

`WM-LEWM-001` evaluated the official LeWorldModel two-loss recipe at small scale. Its strong controls are
shuffled/no-action prediction, paired donor action swaps, equal-dimensional perturbation-norm-matched
random subspaces, action-module suppression, layerwise linear/nonlinear probes, and clean versus
intervened latent planning/closed-loop execution. Reproduction passes all seeds; planner specificity
passes two; the full circuit passes only one and is rejected at the replicated level.

## LLM Meta-Model

- no-change predictor;
- mean intervention effect;
- linear regression;
- bilinear predictor;
- local Jacobian;
- corpus-averaged Jacobian Lens;
- MLP predictor;
- autoregressive layer predictor;
- nearest-neighbor intervention retrieval;
- sparse-feature linear transport.

Success must not be declared from beating only a trivial baseline.

`LLM-GPT2-002` implements all listed CPU-feasible baselines except an autoregressive layer predictor
and a learned SAE transport. Its `sparse_context_linear` baseline uses top-k compressed context and
must not be described as an SAE. The prompt-local finite-difference Jacobian is the primary strong
baseline.

Measured `LLM-GPT2-002` result: the prompt-local Jacobian won by a large margin on both held-out
splits. Bilinear Intervention-JEPA ranked second on unseen prompts/magnitude but last on the held-out
layer. The MLP did not outperform linear/no-change. Keep the local Jacobian as a mandatory baseline;
nonlinear models are not justified for unit coordinate steering in this tested range.

`LLM-GPT2-003` adds direct composition of matching magnitude-6 single effects. This is stronger
than a finite-difference Jacobian because it isolates only the interaction residual at the tested
magnitude. Held-out interaction was `0.043%` of effect power; prompt-local direct addition and
finite differences dominated all learned models. Seen-prompt bilinear success did not transfer.

`LLM-IJEPA-001` evaluated the complete registered Qwen comparison: no-change, mean-effect, linear,
dual-solved bilinear, trained MLP, nearest neighbor, per-example direct local Jacobian, corpus local
transport, and sparse-dictionary feature transport. Intervention-JEPA won the primary fixed split,
but nearest-neighbor narrowly won resampling-holdout MSE and the ranked coordinate failed direct
precision@1.
