# Publishing LLMIntent to PyPI

## One-time setup

```powershell
cd research\LLMIntent
python -m pip install --upgrade build twine
copy .env.example .env
```

Edit `.env` and set your PyPI API token:

```env
TWINE_USERNAME=__token__
TWINE_PASSWORD=pypi-...your-token...
```

Create tokens at [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/).  
Never commit `.env` or paste tokens in chat.

## Publish (recommended)

```powershell
.\scripts\publish.ps1
```

TestPyPI first (optional):

```powershell
# Add TWINE_TESTPYPI_PASSWORD to .env, then:
.\scripts\publish.ps1 -TestPyPI
```

Upload an existing build without rebuilding:

```powershell
.\scripts\publish.ps1 -SkipBuild
```

## Manual publish

```powershell
python -m build
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."
python -m twine upload dist/*
```

Or use `%USERPROFILE%\.pypirc` (optional):

```ini
[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
```

## After upload

```powershell
pip install llmintent
pip install "llmintent[all]"
llmintent --help
```

## Version bumps

Before each release, update **both**:

1. `pyproject.toml` → `[project].version`
2. `src/llmintent/__init__.py` → `__version__`

Then run `.\scripts\publish.ps1`.

## Troubleshooting 403 Forbidden

- **Username** must be exactly `__token__` (not your PyPI login name)
- **Password** is the full token including the `pypi-` prefix (no quotes in `.env`)
- **Revoke** any token pasted in chat; create a new one
- **Verify** your PyPI account email at [pypi.org/manage/account/](https://pypi.org/manage/account/)
- Token scope: use **Entire account** for the first upload of a new project

## What gets published

Included:

- `src/llmintent/` Python package
- `README.md`, `LICENSE`
- CLI entry point: `llmintent`

Excluded:

- `tests/`, `data/`, `llmintent_retraces/`
- `.env` (gitignored)
