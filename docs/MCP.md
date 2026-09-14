# LLMIntent MCP (1.4.0)

Stdio MCP so Cursor, VS Code, Claude Code, or any other agent can drive the suite without loading Qwen 27B.

Default tools stay **offline**. `li_latent` uses `backend=rule`. HF / 27B is opt-in.

## Start

```bash
python -m llmintent.mcp
python -m llmintent mcp
llmintent-mcp
python -m llmintent mcp --install
```

`--install` prints host JSON for Cursor (`.cursor/mcp.json`), VS Code, Claude Code, and a generic snippet.

This repo already ships [`.cursor/mcp.json`](../.cursor/mcp.json). Point `PYTHONPATH` at `src` if you are running from a checkout.

## Tools

| Tool | What it does |
|------|----------------|
| `li_status` | Version, atlas region ids, tool names |
| `li_install` | Host JSON snippets |
| `li_models` | Curated suite registry (offline) |
| `li_compile` | Closed region catalogue; unmatched English is dropped |
| `li_anatomy` | What each region **does**, occupancy **through each prompt span**, connectome IV, A vs B ablation. `draft=true` appends a guided report |
| `li_draft` | Prose report: template, or `slm` / `endpoint` |
| `li_isolates` / `li_motifs` / `li_trajectory` | Offline isolates path |
| `li_iv` | Indication vs IV (mock by default) |
| `li_latent` | ThoughtReport; default `backend=rule` |

Resource `anatomy://atlas` and prompt `anatomy_report` are also advertised.

Typical agent loop: `li_compile` → `li_anatomy` → `li_draft`.

## SLM / agent guide

The structured anatomy map is the source of truth. The guide only drafts prose from those facts.

**Template (offline, default)**

```bash
python -m llmintent anatomy --text "I hear a song because a dark shape is looming." --draft
python -m llmintent guide --text "I hear a song because a dark shape is looming."
```

**Live SLM in this suite**

```bash
python -m llmintent anatomy --text "..." --draft --slm gpt2
```

**OpenAI-compatible endpoint**

```bash
python -m llmintent anatomy --text "..." --draft --endpoint http://127.0.0.1:8000/v1/chat/completions
```

Env aliases: `LLMINTENT_GUIDE_URL`, `LLMINTENT_GUIDE_MODEL`, `LLMINTENT_GUIDE_KEY`.

**Your own agent (Python)**

```python
from llmintent.anatomy import map_anatomy, draft_anatomy_report

report = map_anatomy("I hear a song because a dark shape is looming.")
draft = draft_anatomy_report(
    report,
    agent=lambda system, user: my_agent.complete(system, user),
)
print(draft.markdown)
```

Resolution order: callable `agent` → `endpoint` / `LLMINTENT_GUIDE_URL` → live `slm` key → deterministic template.

Do not claim fly neuropils live inside the transformer. Residuals are correlates, not mind-reading.
