# Runs menu embedding with an already cached Hugging Face model.
# No package installation, model download, or OCR/menu generation is performed.
[CmdletBinding()]
param(
    [string]$InputFile = 'structure.json',
    [string]$Output = 'data/menu-embeddings.json',
    [string]$PythonPath = '.venv-embeddings/Scripts/python.exe',
    [ValidateRange(1, 2147483647)][int]$BatchSize = 2,
    [ValidateRange(1, 32768)][int]$MaxLength = 512,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found at '$PythonPath'. Complete the setup in docs/embeddings.md on the model PC first."
    }
    if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
        throw "Menu file not found at '$InputFile'. Generate it with scripts/generate_structureFile.ps1 or pass -InputFile."
    }

    Write-Output 'Generating product embeddings using the cached Qwen model...'
    $embeddingArguments = @(
        '-m', 'app.services.embed_menu',
        '--input', $InputFile,
        '--output', $Output,
        '--batch-size', $BatchSize,
        '--max-length', $MaxLength,
        '--device', $Device.ToLowerInvariant(),
        '--local-files-only'
    )
    & $PythonPath @embeddingArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Embedding failed with exit code $LASTEXITCODE. The model must already be cached; this script never downloads it. See the error above."
    }
} finally {
    Pop-Location
}
