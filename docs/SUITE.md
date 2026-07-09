# LLMIntent suite architecture

**LLMIntent** is the suite home for semantic extraction **and** intent-structure tooling. Sibling repos (`intent-isolates`, future LatentIntentInspect) remain extractable libraries; this package re-exports and vendors the offline path so one install covers the suite.

## Suite diagram

```mermaid
flowchart TB
  subgraph suite [LLMIntent suite]
    M[models / suite registry<br/>Qwen · Mistral · MiniMax · GLM]
    A[analyzer · jspace · heighten · live · viz]
    I[isolates + typology]
    Mo[motifs + reasoning trajectories]
    L[latent inspect hooks]
    IV[iv_motifs / causal_layers<br/>indication vs IV causation]
  end

  ExtII[intentisolates<br/>optional extractable]
  ExtLI[LatentIntentInspect<br/>optional / soft]
  ExtIV[causaliv / autocausal<br/>soft IV backends]

  ExtII -.->|prefer if installed| I
  I --> Mo
  Mo --> IV
  ExtLI -.-> L
  ExtIV -.-> IV
  M --> A
```

## Module map

| Import | Role |
|--------|------|
| `llmintent` / `llmintent.suite` | Curated model registry + analyzer entrypoints |
| `llmintent.isolates` | Identify isolates, typology, layers, reports |
| `llmintent.motifs` | Alias: `form_motifs`, `trajectory_from_motifs` |
| `llmintent.iv_motifs` / `llmintent.causal_layers` | `LayerCausalSuite` — indication vs IV |
| `llmintent.latent` | Soft LatentIntentInspect / stub |
| `llmintent.isolates._core` | Vendored offline IntentIsolates (always present) |

Resolution for isolates: **prefer** installed `intentisolates` ≥0.3; else use vendored `_core`. Torch / HF / `causaliv` remain soft.

## One install

```bash
pip install llmintent
# optional: prefer external extractable package
pip install "llmintent[isolates]"   # pulls intentisolates
pip install "llmintent[suite]"      # alias of [isolates]
pip install "llmintent[models]"     # accelerate stack for large HF models
```

```python
from llmintent.isolates import identify_isolates, form_motifs, trajectory_from_motifs
from llmintent.iv_motifs import LayerCausalSuite

isos = identify_isolates(text="I want X. I cannot Y. I will do Z.")
motifs = form_motifs(isos)
traj = trajectory_from_motifs(motifs, isos)
result = LayerCausalSuite.from_text("I want X. I will do Z.").run(outcome_hint="Z")
```

## CLI umbrella

```bash
python -m llmintent isolates --text "..."
python -m llmintent motifs --text "..."
python -m llmintent reasoning-trajectory --text "..."
python -m llmintent trajectory --text "..."          # same as reasoning-trajectory
python -m llmintent trajectory --prompt "..."        # activation trajectory (model)
python -m llmintent iv-motifs --text "..." --mock-iv
python -m llmintent models list
```

## Epistemic notes

- Motifs / trajectories are **structural hypotheses**, not proven cognition.
- Abstract L0–L4 layers are a scaffold unless bound to residual indices.
- **Indication ≠ causation** — see IntentIsolates `LAYER_CAUSAL_IV.md` and IV reports.

## Related packages

| Package | Role vs suite |
|---------|----------------|
| [intent-isolates](https://github.com/ehallford11714/intent-isolates) | Extractable lib; also vendored here |
| LatentIntentInspect | Soft hook via `llmintent.latent` (when published) |
| AutoCausalLib | Soft IV / `isolates-causal` bridge |
| CausalIVSuite | Preferred `causaliv` 2SLS when installed |
