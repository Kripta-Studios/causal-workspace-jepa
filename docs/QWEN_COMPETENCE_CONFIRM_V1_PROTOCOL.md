# Qwen competence confirmation V1 protocol

Experiment: `QWEN-BINDING-COMPETENCE-CONFIRM-001`

This file **is** the preregistration narrative. Config:
`configs/experiments/qwen_competence_confirm_v1.json`.

Parent development result: `QWEN-BINDING-COMPETENCE-RECOVERY-001` selected
`qwen_chat_prefill_v1` on calibration only. V3 remains `INELIGIBLE_TASK_PHASE0`.

## Question

Does the already-selected frozen renderer make frozen Qwen3-0.6B
(`c1899de289a04d12100db370d81485cdf75e47ca`) solve the same lookup task on
genuinely fresh confirmation examples?

## Freeze

- Renderer: `qwen_chat_prefill_v1` only. No prompt search.
- Split name: `confirmation`. Seed `701`. Count `32`.
- Keys: maple, quartz, ridge, frost (tokenizer-verified spaced single tokens).
- Values: teal, ivory, coral, peach (same contract).
- These tokens are disjoint from parent calibration/train/validation/test pools.
- Forbidden model-forward splits: train, validation, test, paraphrase, calibration.
- Gates: clean and direct-permuted **full-vocabulary** accuracy `>= 0.90`.
- Candidate-only accuracy is diagnostic only.
- Ledger is mandatory.
- Failure closes this prompt/task path negative without threshold changes.
- Pass authorizes only a separately preregistered mechanistic successor
  (`LLM-QWEN-BINDING-ALGEBRA-004` / `CRCT-QWEN-BRIDGE-003` drafts). It does not
  change V3.

## Command

After this protocol is committed:

```powershell
$env:PYTHONPATH = "src"
C:\Program Files\Python314\python.exe scripts/run_qwen_competence_confirm.py --device cuda
```

Use the CUDA Python that has `torch 2.10.0+cu128`. The repo `.venv` is CPU-only.
