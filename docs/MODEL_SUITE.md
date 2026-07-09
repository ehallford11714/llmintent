# LLMIntent Model Suite — Qwen / Mistral / MiniMax / GLM

Curated Hugging Face (and API) targets for intent analysis beyond the legacy
GPT-2 / DistilGPT-2 CI defaults. Research snapshot: **July 2026**.

## How LLMIntent selects models

Resolution order for `resolve_model_id()` / `load_suite_model()`:

1. Explicit `--model` / `model=` argument  
   - Raw HF id: `Qwen/Qwen2.5-7B-Instruct`  
   - Suite key: `qwen:medium`
2. Explicit `--family` + `--size` (default size `medium`)
3. Environment:
   - `LLMINTENT_MODEL` — HF id or `family:size`
   - `LLMINTENT_FAMILY` + `LLMINTENT_SIZE`
   - `LLMINTENT_DEVICE` — `cpu`, `cuda`, `cuda:0`, …
4. Fallback default: `gpt2` (legacy offline path)

Heavy weights are **never** downloaded at import time. Loading goes through
`transformers` only when `load_suite_model()` / `LLMIntentAnalyzer` / Live
session actually runs.

| Env var | Purpose |
|---------|---------|
| `LLMINTENT_MODEL` | Override HF id or `family:size` |
| `LLMINTENT_FAMILY` | `qwen` \| `mistral` \| `minimax` \| `glm` \| `legacy` |
| `LLMINTENT_SIZE` | `tiny` \| `small` \| `medium` \| `large` \| `xl` |
| `LLMINTENT_DEVICE` | Torch device string |
| `LLMINTENT_LOAD_TEST` | Set to `1` to enable optional weight-download tests |

## Size tiers

| Tier | Intent |
|------|--------|
| `tiny` | Edge / CI / lowest VRAM in family |
| `small` | Laptop / 8–16GB class |
| `medium` | Default research target |
| `large` | Workstation / multi-GPU |
| `xl` | Cluster / API-preferred frontier |

VRAM figures below are **rough FP16/BF16 weight footprints** (no KV cache).
Quantization (AWQ/GPTQ/GGUF) and MoE activation can cut requirements sharply.

---

## Qwen (Alibaba Cloud — `Qwen` on HF)

| Size | Primary HF id | ~Params | ~VRAM FP16 | Notes |
|------|---------------|---------|------------|-------|
| tiny | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~1.5 GB | Live default; alt `Qwen/Qwen3-0.6B` |
| small | `Qwen/Qwen2.5-3B-Instruct` | 3B | ~8 GB | alt `Qwen/Qwen3-4B-Instruct-2507` |
| medium | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | ~16 GB | alt `Qwen/Qwen3-8B` |
| large | `Qwen/Qwen2.5-32B-Instruct` | 32B | ~64 GB | alt MoE `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| xl | `Qwen/Qwen2.5-72B-Instruct` | 72B | ~144 GB | alt `Qwen/Qwen3-235B-A22B-Instruct-2507` |

**API alternatives:** DashScope (Alibaba), OpenRouter, Together, Fireworks.  
**Mirrors:** [ModelScope](https://modelscope.cn/organization/Qwen) if HF is slow/gated in-region.  
**License:** Apache 2.0 for most open Qwen2.5 / Qwen3 weights.

---

## Mistral (`mistralai` on HF)

| Size | Primary HF id | ~Params | ~VRAM FP16 | Notes |
|------|---------------|---------|------------|-------|
| tiny | `mistralai/Ministral-3-3B-Instruct-2512` | 3B | ~8 GB | BF16 alt `…-BF16` |
| small | `mistralai/Ministral-3-8B-Instruct-2512` | 8B | ~18 GB | alt classic `Mistral-7B-Instruct-v0.3` |
| medium | `mistralai/Ministral-3-14B-Instruct-2512` | 14B | ~30 GB | |
| large | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B | ~55 GB | alt `Mixtral-8x7B-Instruct-v0.1` |
| xl | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 141B MoE | multi-GPU | API: `mistral-large-latest` / Large 3 |

**API alternatives:** [Mistral AI Studio](https://mistral.ai/), Azure AI Foundry, Amazon Bedrock, NVIDIA NIM.  
Some newer Large 3 checkpoints may be gated or format-specific — use API if HF access fails.

---

## MiniMax (`MiniMaxAI` on HF)

MiniMax open text weights are **large MoE** models; there is no public sub-billion dense instruct line comparable to Qwen-0.5B. Suite tiers therefore map tiny→medium onto **M2**, large onto **Text-01**, xl onto **M3**.

| Size | Primary HF id | ~Total / active | Guidance |
|------|---------------|-----------------|----------|
| tiny | `MiniMaxAI/MiniMax-M2` | 230B / ~10B | Prefer API on consumer GPUs |
| small | `MiniMaxAI/MiniMax-M2` | 230B / ~10B | vLLM / SGLang recommended |
| medium | `MiniMaxAI/MiniMax-M2` | 230B / ~10B | Default local MiniMax |
| large | `MiniMaxAI/MiniMax-Text-01` | 456B / ~46B | Multi-GPU; `trust_remote_code` |
| xl | `MiniMaxAI/MiniMax-M3` | 428B / ~23B | 1M context; multimodal |

**API alternatives:** `https://api.minimax.io` — models `MiniMax-M3`, `MiniMax-M2.7`, `MiniMax-M2`, etc.  
**Load tip:** set `trust_remote_code=True`; prefer vLLM/SGLang over naive `pipeline` for MoE.

---

## GLM / ChatGLM (Zhipu / Z.ai — `zai-org`, `THUDM`)

| Size | Primary HF id | ~Params | Guidance |
|------|---------------|---------|----------|
| tiny | `THUDM/chatglm3-6b` | 6B | Legacy ChatGLM; OK for small VRAM |
| small | `zai-org/GLM-4-9B-0414` | 9B | Modern instruct / tools (`THUDM/…` mirror) |
| medium | `zai-org/GLM-4.7-Flash` | 30B-A3B MoE | Efficient local; alt dense `GLM-4-32B-0414` |
| large | `zai-org/GLM-4.5-Air` | 106B-A12B | Multi-GPU / FP8 |
| xl | `zai-org/GLM-4.7` | 355B-A32B | Cluster; alts GLM-4.5 / 4.6 |

**API alternatives:** [Z.ai](https://z.ai/) / Zhipu Open Platform (`glm-4.7`, `glm-4.7-flash`, `glm-4.5-air`, …).  
**Mirrors:** ModelScope `ZhipuAI/…` if HF gated.

---

## Legacy (offline CI)

| Size | HF id | Role |
|------|-------|------|
| tiny | `distilgpt2` | Fastest CI |
| small | `gpt2` | Default analyzer fallback |
| medium | `gpt2-medium` | |
| large | `gpt2-large` | |
| xl | `gpt2-xl` | |

These remain the **safe default** when no family is requested and for AutoCausal soft imports that must not pull multi-GB weights.

---

## Python API

```python
from llmintent.suite import list_models, get_model_spec, resolve_model_id, load_suite_model

list_models(family="qwen")
spec = get_model_spec("mistral", "small")
print(spec.hf_id, spec.vram_gb_fp16)

# Resolve without loading
hf_id = resolve_model_id(family="glm", size="medium")

# Load (downloads on first use)
bundle = load_suite_model(family="qwen", size="tiny", device="cpu")
```

Wire into the analyzer:

```python
from llmintent import LLMIntentAnalyzer
from llmintent.suite import resolve_model_id

analyzer = LLMIntentAnalyzer(resolve_model_id(family="qwen", size="tiny"), load_glove=False)
# or:
analyzer = LLMIntentAnalyzer.from_suite(family="qwen", size="tiny", load_glove=False)
```

## CLI

```powershell
python -m llmintent models list
python -m llmintent models list --family qwen
python -m llmintent models info qwen medium
python -m llmintent run --family qwen --size small --text "The capital of France is"
python -m llmintent analyze --family mistral --size tiny --prompt "Hello"
```

## Install extras

```powershell
pip install "llmintent[models]"   # transformers, torch, accelerate (explicit suite stack)
pip install "llmintent[hf]"       # alias of [models]
pip install "llmintent[slm]"      # alias — small/legacy friendly pin set
```

Core `llmintent` already depends on `torch` + `transformers`; the extras document the
recommended stack for larger suite models (`accelerate` for `device_map="auto"`).
