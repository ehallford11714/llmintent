# LLMIntent

Python library derived from the **SemanticExtractionLLms** research notebook. It extracts semantic structure from transformer weights and runtime hidden states: morpheme wells, semantic poles, layer pivots, chain-of-thought intensity, compaction metrics (SSO), J-space layer thoughts, and cognitive module kernels.

## Source

The reference notebook lives at `reference/SemanticExtractionLLms.ipynb`. Code cells are extracted to `reference/extracted_cells.py`.

## Install

```powershell
git clone https://github.com/ehallford11714/llmintent.git
cd llmintent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
```

## Quick start

```python
from llmintent import LLMIntentAnalyzer

analyzer = LLMIntentAnalyzer("gpt2", load_glove=False)
report = analyzer.analyze_prompt("The quick brown fox jumps over the lazy")
print(report.activation_layers)
print(report.intensity_sweep.head())
analyzer.cleanup()
```

## CLI

```powershell
llmintent analyze --model gpt2 --prompt "Two plus two equals"
llmintent trace --model gpt2 --prompt "The spider has 8 legs" --transport --track 8 6
llmintent layers --model gpt2 --prompt "Let's think step by step"
llmintent cognitive --model gpt2 --twin-a "simple prompt" --twin-b "CoT prompt"
llmintent query --model gpt2 --concept "subtraction" --prompt "Eight minus two equals" --twin-b "Let's think step by step..."
llmintent trajectory --model gpt2 --prompt "Eight minus two equals" --twin-b "Let's think step by step..." --concepts subtraction eight
llmintent compare-cot --model gpt2 --direct "I have ten apples..." --cot "Let's think step by step..."
```

## Modules

| Module | Purpose |
|--------|---------|
| `metrics` | SSO score, Shannon entropy, KL divergence |
| `activation` | Inference pivot, workspace peak, motor onset, intensity peak |
| `layers` | Layer → regime, role, top intent, cognitive module |
| `jspace` | Logit/J-lens decode, transport maps, intent traces |
| `cognitive` | Identity, reasoning, meta-reasoning, ideation kernels |
| `trajectory` | Unified activation trajectory mapping across layers |
| `query` | Semantic concept → layer activation via KL-Barlow-KNN |
| `morphemes` | Lemma/morpheme extraction (Stanza, spaCy, polyglot) |
| `projection` | GloVe ↔ model embedding projection matrix |
| `poles` | Semantic, grammatical, numerical reference poles |
| `weight_semantics` | Weight-slice → vocabulary KNN → semantic units |
| `steering` | Layer-wise pole intensity and CoT comparison |
| `compaction` | SVD-based semantic isolate detection |
| `analyzer` | High-level `LLMIntentAnalyzer` facade |

---

## Advanced Features

LLMIntent combines three research lines into one pipeline: the **SemanticExtractionLLms** notebook (weight semantics, steering, compaction), Anthropic's **Global Workspace / J-space** paper (verbal layer thoughts), and a novel **cognitive kernel** framework (KL + twin Barlow minimization).

### 1. Activation layer identification

Pinpoints where computation "turns on" inside a transformer for a given prompt.

```python
from llmintent import LLMIntentAnalyzer

analyzer = LLMIntentAnalyzer("gpt2")
layers = analyzer.identify_activation("Two plus two equals")
# {
#   "inference_pivot": 4,    # largest entropy drop (maturation)
#   "workspace_peak": 5,       # max J-space occupancy in middle layers
#   "motor_onset": 11,         # decode aligns with final output
#   "intensity_peak": 3,       # max numerical-pole similarity
# }
```

**Use cases:** locate the inference pivot for CoT vs direct prompts, find where numerical reasoning concentrates, compare activation profiles across models.

---

### 2. Transformer layer correspondence map

Every layer gets a functional label — not just depth, but *what it is doing*.

```python
layer_map = analyzer.layer_correspondence(
    "Question: 12 * 2 - 5 = ? Answer:",
    twin_b="Question: 12 * 2 - 5 = ? Answer: Let's think step by step...",
)
print(layer_map[[
    "layer", "regime", "role", "top_intent",
    "dominant_module", "kl_divergence", "is_activation_pivot"
]])
```

| Column | Meaning |
|--------|---------|
| `regime` | sensory → workspace → motor (Anthropic bands) |
| `role` | Human-readable function (e.g. "Abstract reasoning & silent verbal thoughts") |
| `top_intent` | Dominant decoded token at that layer |
| `dominant_module` | identity / reasoning / meta_reasoning / ideation |
| `kl_divergence` | Twin-prompt structural tension at this layer |

---

### 3. J-space layer thoughts (Anthropic Jacobian lens)

Surfaces **"words on the model's mind"** at each layer — including silent intermediates before the final token.

Based on [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/) (Gurnee et al., 2026).

```python
analyzer = LLMIntentAnalyzer("gpt2", fit_jspace_transport=True)

trace = analyzer.intent_trace(
    "Question: A spider has 8 legs. Remove 2. Answer:",
    track_tokens=["8", "6", "spider"],
)

# Top thought at each depth
for layer in [0, 3, 6, 11]:
    print(f"L{layer}: {trace.top_thought_at(layer)!r}")

# Token rank evolution across layers
print(trace.rank_curves)  # {"8": [None, 45, 12, 3, ...], ...}
print(trace.regime_bands) # {"sensory": (0,3), "workspace": (4,8), "motor": (9,12)}
```

**Transport lens:** `fit_jspace_transport=True` fits linear maps `J_l` so `h_final ≈ J_l @ h_l`, correcting for representational rotation that breaks the standard logit lens in early/mid layers.

**Sparse decomposition:** active verbal intents via greedy matching pursuit over the unembedding dictionary (`jspace.decompose`).

---

### 4. Cognitive module kernels (KL + Twin Barlow)

Four cognitive functions identified per layer by comparing **twin prompts** (e.g. direct vs chain-of-thought):

| Module | Detection signal | Cognitive role |
|--------|-----------------|----------------|
| **identity** | Low KL + high Barlow diagonal | Stable self-representation; twin-invariant binding |
| **reasoning** | Mid-high KL + high J-space occupancy | Primary computation in workspace band |
| **meta_reasoning** | KL spikes + Barlow off-diagonal coupling | Monitoring/restructuring ("thinking about thinking") |
| **ideation** | High entropy + low motor alignment | Divergent generation before readout commit |

```python
profile = analyzer.cognitive_modules(
    twin_a="I have five apples and eat two. I now have exactly",
    twin_b=(
        "Question: I have five apples and eat two. How many remain? "
        "Answer: Let's think step by step. Five minus two equals"
    ),
)

for kernel in profile.kernels:
    print(f"{kernel.module:15} L{kernel.layer:2d}  score={kernel.score:.3f}  intent={kernel.top_intent!r}")

print(profile.layer_assignments[["layer", "dominant_module", "reasoning", "meta_reasoning"]])
```

**Algorithm:**
1. Compute per-layer KL(P_twin_b ‖ P_twin_a) on next-token distributions
2. Collect twin hidden-state trajectories; minimize **Barlow Twins loss** (diagonal → 1, off-diagonal → 0) weighted by KL
3. Extract combined kernel basis via KL-weighted SVD + Barlow projector
4. Score each layer for four modules; assign dominant module + peak kernel per module

---

### 5. Steering, compaction, and weight semantics (notebook lineage)

From the original SemanticExtractionLLms research:

```python
# CoT vs direct intensity sweep
sweep = analyzer.compare_prompts({
    "Direct": "If I have ten apples and lose three, I have",
    "CoT": "Question: ... Answer: Let's think step by step...",
})

# KL stress test (simple vs complex prompt)
stress = analyzer.stress_test(
    "I have five apples and I eat two. I now have exactly",
    "If I start with the square root of twenty-five and subtract the smallest prime...",
)

# Full analysis with compaction + block semantics
report = analyzer.analyze_prompt(
    prompt,
    cot_prompt=cot_prompt,
    twin_b=cot_prompt,
    include_compaction=True,
    include_block_semantics=True,
    track_tokens=["8", "6"],
)
print(report.compaction)        # SSO isolate density per layer
print(report.inference_pivot) # compaction-derived pivot
```

**SSO (Semantic-Structural Orthogonality):** `(|SemSim| - |StrSim|) / (|SemSim| + |StrSim|)` — measures semantic purity of FFN weight components after GloVe projection.

---

### 6. Full analysis report

`analyze_prompt()` returns an `AnalysisReport` combining all subsystems:

```python
report = analyzer.analyze_prompt(
    "Question: 12 * 2 - 5 = ? Answer:",
    cot_prompt="... Let's think step by step ...",
    twin_b="... Let's think step by step ...",
    include_jspace=True,
    include_compaction=False,
    track_tokens=["24", "19"],
)

report.activation_layers   # pivot layers
report.intent_trace        # J-space IntentTrace
report.layer_map           # full correspondence + cognitive modules
report.cognitive_profile   # CognitiveModuleProfile
report.intensity_sweep     # numerical pole intensity per layer
report.entropy_trajectory  # maturation curve
report.cot_comparison      # direct vs CoT at pivot
report.pivot_entropy       # entropy validation at pivot
```

---

### 9. Unified trajectory mapping (`trajectory.py`)

Single API that merges all per-layer signals into one trajectory table:

```python
mapping = analyzer.trajectory_map(
    prompt="Question: Eight minus two equals ? Answer:",
    twin_b="... Let's think step by step ...",
    concepts=["subtraction", "eight", "step by step"],
)

print(mapping.pivots)          # inference_pivot, workspace_peak, ...
print(mapping.layers)          # entropy, KL, intensity, top_intent, regime, cognitive module
print(mapping.layers_for_concept("subtraction"))
```

Each row = one layer. Columns include `entropy`, `kl_divergence`, `intensity`, `top_intent`, `regime`, `dominant_module`, `is_activation_pivot`, and optional `concept_*_activation` columns.

---

### 8. Semantic concept query (KL + Barlow + KNN)

Directly query a **semantic concept** (plain text) and get back which layers in the activation trajectory it activates.

```python
result = analyzer.query_concept(
    concept="subtraction",
    prompt="Question: Eight minus two equals ? Answer:",
    twin_b="Question: ... Answer: Let's think step by step. Eight minus two is",
)

print(result.peak_layer)       # e.g. 5
print(result.matched_layers)   # [5, 4, 6, 3, 7]
print(result.knn_ranking)        # KNN + fused scores per layer
print(result.trajectory)         # full trajectory with concept_activation column
```

**Strategy:**
1. Build per-layer **KL + twin Barlow** feature vectors from twin prompts
2. Embed concept text into the same space (token embeddings + contextual hidden state)
3. **KNN (cosine)** retrieves nearest layers in Barlow-projected space
4. Re-rank by `KNN sim × KL weight × Barlow invariance × semantic probe`
5. Annotate full activation trajectory with `concept_similarity` and `concept_activation`

```python
# Batch query
results = analyzer.query_concepts(
    ["identity", "reasoning", "ideation", "numerical"],
    prompt,
    twin_b=cot_prompt,
)
```

---

### 7. Low-level API

For custom pipelines without the facade:

```python
from llmintent.kernels import minimize_twin_barlow, per_layer_kl_profile, collect_twin_hidden_matrix
from llmintent.jspace import decode_intents, fit_transport_maps, sparse_intent_decomposition
from llmintent.cognitive import build_cognitive_module_profile
from llmintent.metrics import calculate_sso_score, kl_divergence, shannon_entropy

# Direct kernel fitting
kl, _ = per_layer_kl_profile(bundle, twin_a, twin_b)
h_a, h_b = collect_twin_hidden_matrix(bundle, twin_a, twin_b)
projector, metrics = minimize_twin_barlow(h_a, h_b, kl, proj_dim=32)

# Single-layer intent decode
intents = decode_intents(bundle, hidden_state, layer=6, transport=projector, top_k=10)
sparse = sparse_intent_decomposition(bundle, hidden_state, k=16)
```

---

## Examples

| Script | What it demonstrates |
|--------|---------------------|
| `examples/basic_steering.py` | Intensity sweep + entropy trajectory |
| `examples/cot_intensity.py` | Direct vs CoT comparison |
| `examples/jspace_layer_thoughts.py` | J-space trace + activation layers |
| `examples/cognitive_kernels.py` | Identity/reasoning/meta/ideation kernels |
| `examples/trajectory_mapping.py` | Unified activation trajectory map |
| `examples/query_concept.py` | Semantic concept → layer activation query |

## References

- SemanticExtractionLLms notebook (Kineteq research)
- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/) — Gurnee et al., Anthropic, 2026
- [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) — Zbontar et al., 2021

## License

MIT
