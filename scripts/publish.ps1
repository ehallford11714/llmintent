# Publish llmintent to PyPI or TestPyPI
param(
    [switch]$TestPyPI,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$EnvFile = Join-Path $Root ".env"
if (-not (Test-Path $EnvFile)) {
    throw "Missing .env - copy .env.example to .env and set TWINE_PASSWORD"
}

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    if ($line -match '^\s*([^#=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim().Trim('"').Trim("'")
        if ($value -match "REPLACE_WITH") {
            throw "Set a real value for $name in .env"
        }
        Set-Item -Path "env:$name" -Value $value
    }
}

if ($TestPyPI) {
    if ($env:TWINE_TESTPYPI_PASSWORD) {
        if ($env:TWINE_TESTPYPI_USERNAME) {
            $env:TWINE_USERNAME = $env:TWINE_TESTPYPI_USERNAME
        } else {
            $env:TWINE_USERNAME = "__token__"
        }
        $env:TWINE_PASSWORD = $env:TWINE_TESTPYPI_PASSWORD
    }
    if (-not $env:TWINE_PASSWORD) {
        throw "Set TWINE_TESTPYPI_PASSWORD or TWINE_PASSWORD in .env for TestPyPI"
    }
    $RepoArgs = @("--repository", "testpypi")
    $Target = "TestPyPI"
} else {
    if (-not $env:TWINE_USERNAME) { $env:TWINE_USERNAME = "__token__" }
    if (-not $env:TWINE_PASSWORD) {
        throw "Set TWINE_PASSWORD in .env"
    }
    if ($env:TWINE_USERNAME -ne "__token__") {
        Write-Warning "TWINE_USERNAME should be __token__ for API tokens (got $($env:TWINE_USERNAME))"
    }
    if ($env:TWINE_PASSWORD -notmatch '^pypi-') {
        throw "TWINE_PASSWORD must start with pypi- (check .env for typos or quotes)"
    }
    $RepoArgs = @()
    $Target = "PyPI"
}

$env:TWINE_NON_INTERACTIVE = "1"

python -m pip install --upgrade build twine | Out-Null

if (-not $SkipBuild) {
    Write-Host "Building sdist + wheel..."
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    python -m build
    if ($LASTEXITCODE -ne 0) { throw "python -m build failed" }
}

if (-not (Test-Path dist)) {
    throw "No dist/ folder - run without -SkipBuild first"
}

Write-Host "Uploading to $Target..."
& python -m twine upload @RepoArgs --verbose dist/*
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Upload failed (403 usually means bad/revoked token or unverified PyPI email)."
    Write-Host "1. Create new token: https://pypi.org/manage/account/token/"
    Write-Host "2. Set .env: TWINE_USERNAME=__token__  TWINE_PASSWORD=pypi-..."
    Write-Host "3. Verify email on PyPI account"
    throw "twine upload failed"
}

Write-Host "Done."
if ($TestPyPI) {
    Write-Host "Install: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmintent"
} else {
    Write-Host "Install: pip install llmintent"
}
