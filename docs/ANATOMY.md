# Fly-connectome anatomy of an LLM (1.7.0)

The fly brain is the **guiding engine**, not a language model we pretend the transformer is. MaleCNS / fly wiring gives LLMIntent a closed region catalogue, an identification prior for instrumental variables, and a test that activating region A vs B must change the output.

This is an anatomical *prior*. It is not a claim that Qwen grew an optic lobe.

## v1.6 function-first atlas

The 1.6 path is **not** another catalogue of fly-inspired labels. Sequence:

1. Fly functional assay (LPLC2 → DNp01 looming/escape rate model; proposed simulation reproducing published *direction* on held-out speeds).
2. Weight SVD of `W_down` with signed logit profiles and FFN unit localization.
3. Matched LLM tasks (approaching-object language → action; causal consequence).
4. Analogous-region test in that shared operation space.
5. Cascade interventions with no-op / random / rescue controls.

```bash
python -m llmintent atlas --no-model --out artifacts/atlas_v16
python -m llmintent atlas --model gpt2 --out artifacts/atlas_v16
```

See [`ATLAS.md`](ATLAS.md) for evidence modes, SVD convention, and what was *not* retrieved (MaleCNS volumes, Lappalainen 2024 network, living-fly recordings).

Depth bands remain an **optional prior**, not a discovered organization. Token projections are **associations**, not “responsible for” until a function test lands. Missing SVD axes return **unidentified**, not random fills.

Latent inspection still resolves as `latentintent` / `latentintentinspect` if installed, else `llmintent.latent_vendor`.

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

So “deciphering 27B intent” here means: (a) compile the prompt onto fly territories, (b) look at what the residual would say next at each depth, (c) only accept a residual→region assignment if it clears the same floor as compile. We did (a) and (b). (c) failed on logit-lens — which is a result, not a miss.

## Latent intent through each layer (1.7.0)

Logit-lens needs the residual dim to match the unembedding. On Qwen 3.8-27B it does not. **Residual-probe tracking** does not use the unembed: at every layer, the prompt's last-token hidden state is compared (cosine) to last-token hidden states of a fixed bank of intent-prototype prompts run through the *same* layer.

Three channels, never mixed: **compile** (prompt prior), **residual probe** (latent ranking), **logit-lens** (optional; unidentified on dim mismatch). Depth bands are labels, not a discovered sensory→motor map.

```bash
python -m llmintent intent-track --text "I hear a song because a dark shape is looming."
python -m llmintent intent-track --battery --model qwen:27b --no-lens
```

On 27B with a shared 32-probe bank, ranking (not the raw 0.7 floor) is the signal: many short English probes sit near each other in last-token space. Hear→see→ask on the mixed looming prompt; stop-action vs parked-static dissociate; dopamine-the-word is not value-gain.

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

MCP: [`docs/MCP.md`](MCP.md). Agents call `li_compile` → `li_anatomy` → `li_draft` / `li_trajectory`. `li_latent` stays on the rule backend unless you ask for `hf`.

## Anatomy of any weighted model (1.5.0)

`Anatomy` is the entry point. Given **any Hugging Face model with weights**, it SVD-maps each block's FFN, unembeds top tokens, scores them on the **full intent catalogue** (fly atlas plus inquire/plan/harm/deception/autonomy/… — not fly-only), and says **what each layer is responsible for**. It then builds a **complete graph**: fly-connectome prior ∪ residual stream Lᵢ→Lᵢ₊₁ ∪ layer→intent responsibility ∪ co-responsibility ∪ cascade.

```python
from llmintent.anatomy import Anatomy, trajectory

# Structural map from weights (GPT-2, Qwen, Mistral, GLM, …)
anat = Anatomy.from_pretrained("gpt2")
print(anat.layer(7).responsible_for())
print(anat.graph.mermaid())

# Offline prior (no download)
anat = Anatomy.offline("I hear a song because a dark shape is looming.")

# All layers × all intents, negative-intent loci, MisAlign Flag
traj = trajectory("I hear a song because a dark shape is looming.", print_flag=True)
```

```bash
python -m llmintent anatomy --text "I hear a song because a dark shape is looming." --model gpt2
python -m llmintent trajectory --text "I hear a song because a dark shape is looming." --no-flag --format json
```

`trajectory` imputes how the model is reasoning from correlates. JSON includes **every intent on every layer** (zeros too). Markdown shows active intents. If negative intent emerges (harm, deception, giant-fibre short-pipe, residual heading that is not in the prompt), **MisAlign Flag** prints to stderr.

`qwen:27b` is a special-case size (not a sixth suite tier). FP16 is ~54 GB; NF4 is required on ~24 GB GPUs. `[models]` extra now includes `bitsandbytes`.
