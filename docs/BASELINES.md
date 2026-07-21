# Baselines

Status: `SMOKE_VALIDATED` for the corrected Qwen exact-derivative comparison; its nonlinear-
advantage result is negative. Other methods retain their recorded statuses.

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
- exact autograd directional JVP;
- symmetric central-difference convergence sweep;
- second-order directional Taylor approximation;
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

Correction: its per-example “local Jacobian” was actually a one-sided 5-percent bfloat16 secant.
It is retained as a historical baseline, but it is not a valid strong Jacobian comparator until
checked against float32 direct effects and exact autograd JVPs. `LLM-QWEN-JVP-AUDIT-001` performs
that audit with central-difference convergence, quadratic Taylor, raw/deduplicated scoring, and
BF16/FP32 drift reporting. The learned model in the old run is a supervised conditional bottleneck,
not a target-encoder/stop-gradient JEPA; a genuine JEPA objective is a separate milestone.

V2 passed every numerical gate and established the corrected ranking on the fixed primary split:
quadratic Taylor MSE `0.07870`, exact JVP `0.6143`, conditional bottleneck `3.1899`, nearest neighbor
`4.9026`, and no-change `6.8654`. The historical BF16 secant scored `120.8994`. Exact and quadratic
transport are mandatory strong baselines for all future Qwen meta-model claims.

`LLM-TARGET-IJEPA-001` implements the genuine target-encoder comparison on behavior-changing
capital patches. It retains no-change, mean, raw linear, PCA-bilinear, supervised MLP, legacy
conditional bottleneck, PCA nearest-neighbor, corpus-average source-delta transport, sparse-
dictionary linear transport, exact JVP, and quadratic Taylor. It reports hidden-space and logit-
space normalized MSE separately because the frozen dataset showed that full-vector MSE and answer
behavior can rank exact JVP and quadratic Taylor differently.

Measured result: no genuine-JEPA seed passed. Exact JVP minimized full-vector normalized MSE
(`0.599`), raw linear ridge minimized logit normalized MSE (`0.329`), and quadratic Taylor maximized
answer-candidate agreement (`0.700`). The target-JEPA ensemble scored `0.930`, `0.587`, and `0.200`
on those endpoints. The PCA-bilinear control reached `2.04e11` normalized MSE under entity shift;
this is retained as a failed extrapolative baseline, not clipped or omitted.
