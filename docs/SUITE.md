# LLMIntent suite architecture

**LLMIntent** is the suite home for semantic extraction **and** intent-structure tooling. Sibling repos (`intent-isolates`, `latent-intent-inspect`) remain extractable libraries; this package re-exports and vendors the offline path so one install covers the suite.

## Suite diagram

```mermaid
flowchart TB
  subgraph suite [LLMIntent suite]
    M[models / suite registry<br/>Qwen · Mistral · MiniMax · GLM]
    A[analyzer · jspace · heighten · live · viz]
    I[isolates + typology]
    Mo[motifs + reasoning trajectories]
    L[latent thought inspect]
    IV[iv_motifs / causal_layers<br/>indication vs IV causation]
    An[anatomy<br/>fly connectome IV · SVD · ablation]
  end

  ExtII[intentisolates<br/>optional extractable]
  ExtLI[latentintent<br/>optional extractable]
  ExtIV[causaliv / autocausal<br/>soft IV backends]

  ExtII -.->|prefer if installed| I
  I --> Mo
  Mo --> IV
  ExtLI -.->|prefer if installed| L
  ExtIV -.-> IV
  An --> IV
  M --> A
```

## Module map

| Import | Role |
|--------|------|
| `llmintent` / `llmintent.suite` | Curated model registry + analyzer entrypoints |
| `llmintent.isolates` | Identify isolates, typology, layers, reports |
| `llmintent.motifs` | Alias: `form_motifs`, `trajectory_from_motifs` |
| `llmintent.iv_motifs` / `llmintent.causal_layers` | `LayerCausalSuite` — indication vs IV |
| `llmintent.latent` | Latent thought inspection (`ThoughtReport`) |
| `llmintent.latent_vendor` | Vendored offline latent API (always present) |
| `llmintent.anatomy` | Fly-connectome atlas → compile → SVD map → IV → region A vs B ablation |
| `llmintent.isolates._core` | Vendored offline IntentIsolates (always present) |

Resolution for isolates: **prefer** installed `intentisolates` ≥0.3; else use vendored `_core`.  
Resolution for latent: **prefer** installed `latentintent` ≥0.1; else use vendored `latent_vendor`. Torch / HF / `causaliv` remain soft for advanced paths.

## Latent thought inspection (1.2.0+)

`llmintent.latent` ships an offline **ThoughtReport** builder:

- Rule/heuristic intent tags
- Synthetic linear probe demo metrics
- SAE-lite sparse codes (sklearn)
- Logit-lens stub
- Mandatory epistemic caveats (correlates ≠ mind-reading)

```python
from llmintent import latent

print(latent.describe())
report = latent.inspect_text("I want X but cannot Y. What should I do?")
print(report.to_json())
```

```bash
python -m llmintent latent --text "I want X but cannot Y."
python -m llmintent latent --status
```

Optional HF residual capture: install extractable package

```bash
pip install "llmintent[latent]"   # pulls latentintent when published
# or: pip install -e ../LatentIntentInspect
python -m llmintent latent --text "..." --backend hf --model distilgpt2
python -m llmintent latent --text "I hear a song because a dark shape is looming. What should I do?" --backend hf --model qwen:27b --4bit --format markdown
```

Qwen 27B (`qwen:27b` → `Qwen/Qwen3.8-27B`) is a special-case size, not a 6th suite tier. On a 24 GB GPU it loads NF4. Residuals are logit-lens decoded onto the fly atlas; that is a correlate, not mind-reading. Thinking is disabled on the prompt so the stream is not mixed with verbalized `<think>` tokens.

SOTA research map (sibling tree): `research/docs/SOTA_LATENT_THOUGHT_INSPECTION.md`.

Full write-up (how the fly map works, and the Qwen 27B residual test): [`docs/ANATOMY.md`](ANATOMY.md).

## Fly-connectome anatomy (1.3.0+)

`llmintent.anatomy` maps a transformer as if it had fly-like territories. Each region records (1) what it handles, (2) how literature-core wiring integrates it, (3) whether activating region A vs B changes the next-token (or linear) output.

```python
from llmintent.anatomy import compile_regions, map_anatomy

plan = compile_regions("a dark shape rushing toward me")
report = map_anatomy("I hear a song because a dark shape is looming.")
print(report.to_markdown())
```

- **Compile** — intent documents, not catalogue wording; unmatched English is dropped.
- **IV** — sensory regions instrument central regions only if the collapsed connectome has a path. Vision→descending is flagged as an exclusion violation (giant-fibre shortcut).
- **SVD** — FFN/activation components matched onto region intent docs. Offline tests plant orthogonal axes.
- **Ablation** — drive region A against region B; `changed` is true when the top token (or KL) moves.

```bash
python -m llmintent compile --text "a dark shape rushing toward me"
python -m llmintent anatomy --text "I hear a song because a dark shape is looming."
python -m llmintent anatomy --text "..." --model gpt2 --region-a vision --region-b auditory
```

## One install

```bash
pip install llmintent
# optional: prefer external extractable packages
pip install "llmintent[isolates]"   # pulls intentisolates
pip install "llmintent[latent]"     # pulls latentintent when on PyPI
pip install "llmintent[suite]"      # isolates + latent
pip install "llmintent[models]"     # accelerate stack for large HF models
```

```python
from llmintent.isolates import identify_isolates, form_motifs, trajectory_from_motifs
from llmintent.iv_motifs import LayerCausalSuite
from llmintent import latent

isos = identify_isolates(text="I want X. I cannot Y. I will do Z.")
motifs = form_motifs(isos)
traj = trajectory_from_motifs(motifs, isos)
result = LayerCausalSuite.from_text("I want X. I will do Z.").run(outcome_hint="Z")
thought = latent.inspect_text("I want X. I cannot Y.")
```

## CLI umbrella

```bash
python -m llmintent isolates --text "..."
python -m llmintent motifs --text "..."
python -m llmintent reasoning-trajectory --text "..."
python -m llmintent trajectory --text "..."          # same as reasoning-trajectory
python -m llmintent trajectory --prompt "..."        # activation trajectory (model)
python -m llmintent iv-motifs --text "..." --mock-iv
python -m llmintent latent --text "..."              # ThoughtReport (offline)
python -m llmintent compile --text "a dark shape rushing toward me"
python -m llmintent anatomy --text "I hear a song because a dark shape is looming."
python -m llmintent models list
```

## Epistemic notes

- Motifs / trajectories are **structural hypotheses**, not proven cognition.
- Abstract L0–L4 layers are a scaffold unless bound to residual indices.
- **Indication ≠ causation** — see IntentIsolates `LAYER_CAUSAL_IV.md` and IV reports.
- Latent ThoughtReports are **probes/heuristics**, not human mind-reading or proven model goals.

## Related packages

| Package | Role vs suite |
|---------|----------------|
| [intent-isolates](https://github.com/ehallford11714/intent-isolates) | Extractable lib; also vendored here |
| [latent-intent-inspect](https://github.com/ehallford11714/latent-intent-inspect) | Extractable latent inspect; soft-preferred by `llmintent.latent` |
| AutoCausalLib | Soft IV / `isolates-causal` bridge |
| CausalIVSuite | Preferred `causaliv` 2SLS when installed |
