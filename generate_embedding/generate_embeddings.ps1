# Runs menu embedding via the Gemini API. Loads GEMINI_API_KEY from the
# project's .env automatically (never printed); an already-set environment
# variable still wins over the file. No local model, GPU, or download needed.
[CmdletBinding()]
param(
    [string]$InputFile = 'structure.json',
    [string]$Output = 'generate_embedding/data/menu-embeddings.json',
    [string]$PythonPath = '.venv/Scripts/python.exe',
    [ValidateRange(1, 2147483647)][int]$BatchSize = 10,
    [ValidateRange(1, 32768)][int]$MaxLength = 2048
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found at '$PythonPath'. Create .venv and install requirements.txt plus generate_embedding/requirements-gemini.txt first."
    }
    if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
        throw "Menu file not found at '$InputFile'. Generate it with generate_embedding/generate_structureFile.ps1 or pass -InputFile."
    }

    $envFile = Join-Path $projectRoot '.env'
    if (Test-Path -LiteralPath $envFile -PathType Leaf) {
        foreach ($line in Get-Content -LiteralPath $envFile) {
            if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
            $key, $value = $line -split '=', 2
            $key = $key.Trim()
            $value = $value.Trim()
            if ($key -and $value -and -not (Test-Path "env:$key")) {
                Set-Item -Path "env:$key" -Value $value
            }
        }
    }
    if (-not $env:GEMINI_API_KEY) {
        throw 'GEMINI_API_KEY is not set. Fill it in in .env, or set it directly: $env:GEMINI_API_KEY = "..."'
    }

    Write-Output 'Generating product embeddings using the Gemini API...'
    $embeddingArguments = @(
        '-m', 'generate_embedding.embed_menu',
        '--input', $InputFile,
        '--output', $Output,
        '--batch-size', $BatchSize,
        '--max-length', $MaxLength
    )
    & $PythonPath @embeddingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Embedding failed with exit code $LASTEXITCODE. See the error above."
    }
} finally {
    Pop-Location
}
