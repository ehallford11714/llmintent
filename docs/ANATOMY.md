# Fly-connectome anatomy of an LLM (1.4.0)

The fly brain is the **guiding engine**, not a language model we pretend the transformer is. MaleCNS / fly wiring gives LLMIntent a closed region catalogue, an identification prior for instrumental variables, and a test that activating region A vs B must change the output.

This is an anatomical *prior*. It is not a claim that Qwen grew an optic lobe.

## How the mapping works

Literature-core fly types collapse onto eleven LLM regions. Each region records four things:

1. **What it does** — a job sentence (vision reads looming/luminance; causal_logic runs if-then / because; descending commits a command). Band and fly analogue sit beside that job, not instead of it.
2. **How it varies through the prompt** — each clause is compiled on its own. Occupancy is a series: silent → on → peak → drop. `trace_prompt` keeps causal conjunctions (`because`, `so`) on the following span so they still compile.
3. **How it integrates** — edges from the collapsed connectome (sensory → associative / workspace → descending). Sensory→motor short-pipes (vision→descending giant fibre) are flagged as **exclusion violations** for IV.
4. **Ablation** — driving region A against region B must move the next-token (or a linear readout). If it does not, the map is not doing causal work.

English is **compiled** onto that closed catalogue by alias hits **and** cosine against region intent documents. Unmatched English is dropped. We do not hash leftover words onto a region.

SVD then places residuals or FFN components onto the same documents. Occupancy is “how much of this prompt/model lives in vision vs auditory,” not a neuropil count.

## What was taken from the fly connectome

Three things only. Nothing else is copied from the fly into the transformer.

| Taken from the fly | How it is used on the LLM | What it is *not* |
|--------------------|---------------------------|------------------|
| Literature-core **cell types** collapsed into 11 territories | Closed region catalogue (`vision` … `motor`) | A claim that Qwen has photoreceptors |
| **Neuropil jobs** (optic lobe sees looming; AMMC hears song; CX heads; DNp01 commits escape) | `what_region_does` — the job sentence on each region card | A neuropil count inside residual space |
| **Type→type edges** (ORN→PN→KC/LH; LPLC2→DNp01 giant fibre; EPG/PEN ring; PFL→DNa02) | Connectome graph as IV prior: sensory Z may instrument central X only if a path exists. Giant-fibre **vision→descending** is an exclusion violation | Proof of synaptic weights in the LM |

Depth bands follow the same cascade: sensory early (0–0.34 of residual blocks), central mid (≈0.28–0.80), motor late (0.62–1.0). That is a *depth prior*, not a layer-name from the fly.

## How each region works

Prompt used below: *I hear a song because a dark shape is looming. What should I do?*

| Region | Fly analogue (taken) | What it does on an LLM prompt | Example on that prompt |
|--------|----------------------|-------------------------------|-------------------------|
| `vision` | Optic lobe: R1–R6, L1/L2, T4/T5, LPLC2, LC4, LC10, HS | Reads luminance, motion, looming, spatial layout | Span *because a dark shape is looming* — aliases `dark` / `looming` |
| `auditory` | JO-A/B → AMMC / WED | Reads pulse, song, sequential tone | Span *I hear a song* — aliases `hear` / `song` |
| `olfactory` | Antennal lobe ORN/PN | Reads chemical identity / naming cues | Silent here (no smell/pheromone) |
| `gustatory` | GNG / SEZ, GRNs | Reads hunger, taste, ingestive drive | Silent here |
| `somatosensory` | Peripheral / ascending | Reads touch and body contact | Silent here |
| `associative` | Mushroom body KC / DAN / MBON | Binds earlier cues to remembered outcomes | Silent unless the prompt asks to remember/bind |
| `valence` | Lateral horn | Innate approach / avoid before a motor program | Silent unless afraid/hate/avoid is said |
| `causal_logic` | Central complex EPG / PEN / PFN / PFL | Runs because / if-then / heading | `because` kept on the looming span |
| `workspace` | LAL / P1 | Holds mixed cues in one buffer | Silent unless integrate/together is said |
| `descending` | DNp01, DNa02, MDN, pIP10 | Selects a command (escape, turn, stop) | Weak unless *escape / decide / jump* appears (*What should I do?* is a request, not a command) |
| `motor` | VNC motor neurons | Emits the next tokens | Late residual punctuation on 27B; not compiled from English |

`trace_prompt` splits on sentence punctuation and keeps `because` / `so` / `then` on the **following** span. So occupancy **varies**: auditory is on for *I hear a song* and off for the looming clause; vision and causal_logic do the reverse.

## How we decipher LLM intent inside 27B

Two reads, ranked. Do not mix them.

**1. Compile (trustworthy catalogue).** English → alias hits **and** cosine vs region intent documents. Floor 0.18. Unmatched clauses are dropped. For the looming prompt this recovers **vision + auditory + causal_logic**. That is the intent *of the prompt* on the atlas, not a claim about hidden states.

**2. Residual logit-lens (correlate).** Load `Qwen/Qwen3.8-27B` (`qwen:27b`) NF4, thinking **off**, last-token unembed every 4 of 64 layers, map those token strings onto the same atlas. On this prompt:

| Depth | Unembedded tokens | What we take it to mean |
|-------|-------------------|-------------------------|
| L0–L16 | punctuation | Early unembed is not lexical yet |
| L20 | `ance` / `arning` | warning fragment |
| L36–L40 | 危险, 威胁 | danger / threat — looming as **collision**, not as song |
| L48–L60 | `<think>`, Additionally, 此外 | thinking gate still in the residual |
| L63 | punctuation | motor punctuation |

Cosine of those lens strings onto intent docs **never cleared 0.18**. Residual occupancy is therefore **not identified**. The atlas answer stays the compile prior. A gustatory bar from a weak match is an artifact; we do not report it as intent.

So “deciphering 27B intent” here means: (a) compile the prompt onto fly territories, (b) look at what the residual would say next at each depth, (c) only accept a residual→region assignment if it clears the same floor as compile. We did (a) and (b). (c) failed — which is a result, not a miss.

These tokens are next-token correlates. They are not inner speech and not a fly neuropil inside Qwen.

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
python -m llmintent anatomy --text "I hear a song because a dark shape is looming." --draft
python -m llmintent guide --text "I hear a song because a dark shape is looming."
python -m llmintent latent --backend hf --model qwen:27b --4bit --format markdown \
  --text "I hear a song because a dark shape is looming. What should I do?"
python -m llmintent mcp --install
```

`--draft` appends a guided prose report. Default is a deterministic template. Pass `--slm gpt2` (or any live suite key), `--endpoint` (OpenAI-compatible chat URL), env `LLMINTENT_GUIDE_URL`, or a Python `agent=lambda system, user: ...` if you already have a writer.

MCP: [`docs/MCP.md`](MCP.md). Agents call `li_compile` → `li_anatomy` → `li_draft`. `li_latent` stays on the rule backend unless you ask for `hf`.

`qwen:27b` is a special-case size (not a sixth suite tier). FP16 is ~54 GB; NF4 is required on ~24 GB GPUs. `[models]` extra now includes `bitsandbytes`.
