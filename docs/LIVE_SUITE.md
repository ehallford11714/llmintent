# LLMIntent Live Suite — Architecture

Real-time application layer for applying focused reasoning, retracement, and steering on loaded SLMs (Phi-3, Qwen 0.5B, GPT-2, etc.).

## Goals

- **One hot model** per session — load once, reuse for analyze / heighten / generate
- **Sub-second to few-second latency** on SLMs — avoid full research pipeline unless requested
- **Same primitives as research** — `heighten`, `retracement`, activation steering, not a parallel stack

## Layer diagram

```text
┌─────────────────────────────────────────────────────────────┐
│  App surfaces                                               │
│  Streamlit UI · FastAPI REST · CLI `llmintent live`         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  live/pipeline.py — LiveIntentPipeline                      │
│  analyze() · heighten() · generate() · probe_next_tokens()  │
│  visualize tab → live/viz_panel.py (VisualizationSuite)     │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────────────┐
│ live/session  │  │ live/generate  │  │ llmintent core       │
│ Model cache   │  │ Chat template  │  │ activation, heighten │
│ Retracement   │  │ Token loop     │  │ retracement, focus │
└───────────────┘  └────────────────┘  └──────────────────────┘
        │
        ▼
┌───────────────┐
│ live/registry │  Phi-3, Qwen2.5-0.5B, TinyLlama, GPT-2, …
└───────────────┘
```

## Module map

| Module | Role |
|--------|------|
| `live/registry.py` | Curated HF model catalog + custom HF id fallback |
| `live/session.py` | `LiveSession` — bundle cache, retracement wrapper, unload |
| `live/pipeline.py` | `LiveIntentPipeline` — orchestration API for apps |
| `live/generate.py` | Chat formatting, greedy/sampled decode, steering hooks |
| `live/api.py` | FastAPI app (`create_app`, `serve`) |
| `live/ui.py` | Streamlit interactive app |
| `live/viz_panel.py` | Visualization suite embed (maps, correlations, GIFs, **real-time layer stream**) |
| `live/activation_stream.py` | Per-token layer activation snapshots during decode |
| `live/types.py` | `LiveAnalyzeResult`, `LiveHeightenResult`, etc. |

## Real-time vs research paths

| Operation | Live path | Full research path |
|-----------|-----------|-------------------|
| Activation pivots | `identify_activation_layers` | + full layer map |
| Focus | `build_trajectory_mapping(include_cognitive=False)` | + cognitive kernels |
| Heighten | retrace string + 2 lightweight mappings | `heighten_reasoning()` loop |
| Generate | `RetracementTransformer.forward` per token | N/A |
| Visualize | `VisualizationSuite` via `live/viz_panel.py` | `llmintent viz` CLI |
| Analyze report | Not exposed | `LLMIntentAnalyzer.analyze_prompt()` |

## Registered models

| Key | HF id | Chat |
|-----|-------|------|
| `qwen-0.5b` | Qwen/Qwen2.5-0.5B-Instruct | yes |
| `qwen-0.5b-base` | Qwen/Qwen2.5-0.5B | no |
| `phi3-mini` | microsoft/Phi-3-mini-4k-instruct | yes |
| `phi2` | microsoft/phi-2 | no |
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | yes |
| `gpt2` | gpt2 | no |
| `distilgpt2` | distilgpt2 | no |

## CLI

```powershell
pip install -e ".[live]"

llmintent live models
llmintent live run --model gpt2 --prompt "..." --action analyze
llmintent live serve --model qwen-0.5b --port 8765
llmintent live ui
```

## API endpoints (FastAPI)

- `GET /models` — registry + loaded model
- `POST /load` — switch model
- `POST /analyze` — pivots + focus
- `POST /heighten` — retrace scaffold + focus delta
- `POST /generate` — completion with retracement / steer
- `POST /probe` — top-k next tokens

## Future extensions

- WebSocket streaming generate
- GPU batch probe for multi-prompt dashboards
- CorrSteer layer ranking before steer (`steering_intent` roadmap)
- Session persistence + retrace store integration for A/B in UI
