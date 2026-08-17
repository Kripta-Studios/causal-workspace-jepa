# Qwen Binding Algebra V2 token-contract adjudication — 2026-08-17

## Disposition

`LLM-QWEN-BINDING-ALGEBRA-002` / `QWEN-BINDING-ALGEBRA-CR-V1` is retained unchanged and is
**ineligible for Phase-0 model-outcome interpretation because its registered answer alphabet
violates the pinned tokenizer contract**.

This is not a competence failure and not a scientific negative.  In the first cache-complete
Bridge-001 execution, `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` loaded successfully.  The runtime stopped in the
pre-forward token-contract check before `_batch_accuracy` or any other competence/model-output
forward was executed.

## Tokenizer-only audit

The frozen tokenizer-only audit contains 88 registered key/value strings.  Exactly 23 **values**
fail the answer-token requirement under the pinned tokenizer:

- calibration: `cobalt`, `saffron` (2)
- train: `Exeter`, `Hobart`, `Izmir`, `Jaipur`, `Kigali`, `Quito`, `Riga`, `Suva`, `Utrecht`,
  `Wuhan`, `Yangon` (11)
- validation: `Accra`, `Cusco`, `Doha`, `Fez`, `Haifa` (5)
- test: `Juba`, `Malmo`, `Nuuk`, `Odesa`, `Pisa` (5)

Total: **23**.

The prompt templates end immediately after `->` or `:`.  Therefore the candidate next-token
contract is the tokenization of the literal string `" " + value`.  A bare-only one-token encoding
is not sufficient.  Keys need not be single tokens because they are neither permuted values nor
candidate answer logits.

## Amendment boundary

The repair is allowed to use only tokenizer metadata.  It may not use model weights, logits,
accuracy, hidden states, validation behavior, test behavior, or any protected outcome.  It must:

1. preserve every V2 answer value already satisfying the strict leading-space single-token
   contract;
2. replace only failed slots, in their original order;
3. preserve colors as colors for calibration and city names as city names for all other splits;
4. preserve split counts, split seeds, key pools, action classes, templates, treatment/capture
   sites, B0/B1 thresholds, training seeds, and protected-access policy;
5. freeze/hash the resolved V3/CR-V2 configuration before a model forward;
6. leave V2 and CR-V1 byte-for-byte unchanged.

`CRCT-QWEN-BRIDGE-002` is the first bridge authorized to consume the resolved V3/CR-V2 parent,
and still authorizes only B0/B1 on calibration/train/validation.
