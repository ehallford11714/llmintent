# Launch LLMIntent Live Streamlit UI
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Installing llmintent[live]..." -ForegroundColor Cyan
python -m pip install -e ".[live]" -q

Write-Host "Starting Live UI at http://localhost:8501" -ForegroundColor Green
python -m streamlit run src/llmintent/live/ui.py --server.port 8501
