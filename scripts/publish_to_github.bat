@echo off
REM One-time publish helper for github.com/ehallford11714/LLMIntent
setlocal

where gh >nul 2>&1
if errorlevel 1 (
  echo GitHub CLI not found. Install: winget install GitHub.cli
  exit /b 1
)

gh auth status >nul 2>&1
if errorlevel 1 (
  echo Log in to GitHub first:
  gh auth login --hostname github.com --git-protocol https --web
)

cd /d "%~dp0"

gh repo view ehallford11714/LLMIntent >nul 2>&1
if errorlevel 1 (
  gh repo create LLMIntent --public --source=. --remote=origin --description "Semantic extraction and intent analysis for transformer LLMs"
) else (
  git remote remove origin 2>nul
  git remote add origin https://github.com/ehallford11714/LLMIntent.git
)

git push -u origin main
echo.
echo Done: https://github.com/ehallford11714/LLMIntent
