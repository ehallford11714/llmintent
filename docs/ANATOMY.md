# Fly-connectome anatomy of an LLM (1.3.0)

The fly brain is the **guiding engine**, not a language model we pretend the transformer is. MaleCNS / fly wiring gives LLMIntent a closed region catalogue, an identification prior for instrumental variables, and a test that activating region A vs B must change the output.

This is an anatomical *prior*. It is not a claim that Qwen grew an optic lobe.

## How the mapping works

Literature-core fly types collapse onto eleven LLM regions. Each region records three things:

1. **What it handles** — vision (looming, luminance), auditory (song, pulse), causal_logic (because / if-then), descending/motor, and so on.
2. **How it integrates** — edges from the collapsed connectome (sensory → associative / workspace → descending). Sensory→motor short-pipes (vision→descending giant fibre) are flagged as **exclusion violations** for IV.
3. **Ablation** — driving region A against region B must move the next-token (or a linear readout). If it does not, the map is not doing causal work.

English is **compiled** onto that closed catalogue by alias hits **and** cosine against region intent documents. Unmatched English is dropped. We do not hash leftover words onto a region.

SVD then places residuals or FFN components onto the same documents. Occupancy is “how much of this prompt/model lives in vision vs auditory,” not a neuropil count.

```text
English
  → compile (aliases + intent docs; drop unmatched)
  → region occupancy
  → connectome paths as IV instruments (sensory Z ⊥ motor Y | path)
  → SVD map of residuals / FFN
  → ablate A vs B (output must change)
```

## Qwen 27B test

We ran that pipeline against **Qwen/Qwen3.8-27B** (`qwen:27b`) in NF4 on a 24 GB-class GPU (RTX 5090 Laptop). Thinking was **off** on the chat template so the residual stream was not mixed with verbalized `<think>` tokens.

Prompt:

> I hear a song because a dark shape is looming. What should I do?

**Compile prior** (closed catalogue, the trustworthy read):

- `vision` — dark shape / looming
- `auditory` — hear / song
- `causal_logic` — because

**Residual logit-lens** (last-token unembed, stride 4 of 64 layers), mapped onto the same atlas:

| Depth | What showed up | Read |
|-------|----------------|------|
| L0–L16 | punctuation | Early unembed is not lexical yet |
| L20 | `ance` / `arning` | warning fragment |
| L36–L40 | 危险, 威胁 | danger / threat — looming as collision, not as song |
| L48–L60 | `<think>`, Additionally, 此外 | thinking gate still in the residual |
| L63 | punctuation | motor punctuation |

Region cosine on those token strings stayed **below the compile floor (0.18)**. Residual occupancy is therefore **not identified**. The fly map’s catalogue answer is the compile prior, not a gustatory bar from weak matches.

These tokens are next-token correlates at that residual. They are not inner speech, not mind-reading, and not a claim the model contains fly neuropils.

```bash
python -m llmintent compile --text "I hear a song because a dark shape is looming."
python -m llmintent anatomy --text "I hear a song because a dark shape is looming."
python -m llmintent latent --backend hf --model qwen:27b --4bit --format markdown \
  --text "I hear a song because a dark shape is looming. What should I do?"
```

`qwen:27b` is a special-case size (not a sixth suite tier). FP16 is ~54 GB; NF4 is required on ~24 GB GPUs. `[models]` extra now includes `bitsandbytes`.
