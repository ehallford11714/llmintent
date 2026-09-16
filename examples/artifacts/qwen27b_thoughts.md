# Latent thoughts

**Model:** `Qwen/Qwen3.8-27B`
**Prompt:** I hear a song because a dark shape is looming. What should I do?
**Compile prior:** vision, causal_logic, auditory

## Layer logit lens

- L0: `vision` — (, ., ,, -
- L4: `vision` — ., (, ,, ?
- L8: `vision` — ,, ., ?, (
- L12: `vision` — ..., ?, ., !
- L16: `vision` — ..., ?, !, .
- L20: `vision` — �, ance, ?, arning
- L24: `causal_logic` — ?, ..., 加斯, beck
- L28: `vision` — ..., �, 抽, est
- L32: `vision` — ..., 抽, 灵, ?
- L36: `vision` — TA, 危险, A, tauf
- L40: `causal_logic` — TA, 呼, 缓, 威胁
- L44: `auditory` — ..., 扑, A, -
- L48: `gustatory` — 这句话, （, <think>, --[
- L52: `gustatory` — <think>, Additionally, Additionally, 此外
- L56: `gustatory` — Additionally, <think>, 此外, 这句话
- L60: `gustatory` — Additionally, -, <think>
- L63: `vision` — -, (

## Occupancy

- `gustatory`: 0.454
- `vision`: 0.228
- `causal_logic`: 0.176
- `auditory`: 0.142

## Caveats
- Inspects model-internal correlates / probes / heuristics — not human thoughts.
- Probe and SAE labels are hypothesized correlates, not proven goals or beliefs.
- Verbalized chain-of-thought may be unfaithful to internal computation.
- Layer saliency in MVP is heuristic unless causal patching evidence is provided.
- Do not claim to 'read the mind' of the model beyond the stated evidence class.
- Logit-lens tokens are next-token correlates at that residual, not inner speech.
