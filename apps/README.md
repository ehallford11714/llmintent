# LLMIntent Live UI

Launch the real-time Streamlit app (Phi-3, Qwen 0.5B, SLMs).

```powershell
# From repo root
pip install -e ".[live]"
llmintent live ui

# Or directly
.\scripts\run_live_ui.ps1
```

Default URL: http://localhost:8501

Optional API server (separate terminal):

```powershell
llmintent live serve --model qwen-0.5b --port 8765
```
