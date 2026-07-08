# LLMIntent

Python library derived from the **SemanticExtractionLLms** research notebook. It extracts semantic structure from transformer weights and runtime hidden states: morpheme wells, semantic poles, layer pivots, chain-of-thought intensity, and compaction metrics (SSO).

## Source

The reference notebook lives at `reference/SemanticExtractionLLms.ipynb` (copied from Downloads). Code cells are extracted to `reference/extracted_cells.py`.

## Install

```powershell
git clone https://github.com/ehallford11714/LLMIntent.git
cd LLMIntent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
python -m spacy download en_core_web_sm
python -c "import stanza; stanza.download('en')"
```

Or from a local checkout:

```powershell
cd C:\Users\ehall\OneDrive\Desktop\research\LLMIntent
pip install -e ".[all]"
```

## Quick start

```python
from llmintent import LLMIntentAnalyzer

analyzer = LLMIntentAnalyzer("distilbert-base-uncased")
report = analyzer.analyze_prompt("The quick brown fox jumps over the lazy")
print(report.intensity_sweep.head())
print(report.pivot_entropy)

analyzer.cleanup()
```

## CLI

```powershell
llmintent analyze --model distilbert-base-uncased --prompt "Two plus two equals"
llmintent compare-cot --model gpt2 --direct "I have ten apples and lose three, I have" --cot "Question: ... Answer: Let's think step by step..."
```

## Modules

| Module | Purpose |
|--------|---------|
| `metrics` | SSO score, Shannon entropy, KL divergence |
| `morphemes` | Lemma/morpheme extraction (Stanza, spaCy, polyglot) |
| `projection` | GloVe ↔ model embedding projection matrix |
| `poles` | Semantic, grammatical, numerical reference poles |
| `weight_semantics` | Weight-slice → vocabulary KNN → semantic units |
| `steering` | Layer-wise pole intensity and CoT comparison |
| `compaction` | SVD-based semantic isolate detection |
| `analyzer` | High-level `LLMIntentAnalyzer` facade |

## Examples

See `examples/` for runnable scripts mirroring notebook sections.
