# Syncs data/menu-embeddings.json vectors to Pinecone. No package
# installation or embedding generation is performed. Requires
# $env:PINECONE_API_KEY to already be set; never pass it as a parameter.
[CmdletBinding()]
param(
    [string]$EmbeddingsFile = 'data/menu-embeddings.json',
    [string]$MenuFile = 'structure.json',
    [string]$PythonPath = '.venv/Scripts/python.exe',
    [string]$IndexName,
    [string]$Namespace,
    [string]$Cloud,
    [string]$Region,
    [ValidateRange(1, 1000)][int]$BatchSize = 100,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found at '$PythonPath'. Create .venv and install requirements.txt first."
    }
    if (-not $DryRun -and -not $env:PINECONE_API_KEY) {
        throw 'PINECONE_API_KEY is not set. Set it in this shell first: $env:PINECONE_API_KEY = "..."'
    }

    $syncArguments = @(
        '-m', 'app.services.pinecone_sync',
        '--embeddings', $EmbeddingsFile,
        '--menu', $MenuFile,
        '--batch-size', $BatchSize
    )
    if ($IndexName) { $syncArguments += @('--index', $IndexName) }
    if ($Namespace) { $syncArguments += @('--namespace', $Namespace) }
    if ($Cloud) { $syncArguments += @('--cloud', $Cloud) }
    if ($Region) { $syncArguments += @('--region', $Region) }
    if ($DryRun) {
        $syncArguments += '--dry-run'
        Write-Output 'Dry run: computing the sync plan only. No Pinecone network calls will be made.'
    } else {
        Write-Output 'Syncing product vectors to Pinecone...'
    }
    & $PythonPath @syncArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pinecone sync failed with exit code $LASTEXITCODE. See the error above."
    }
} finally {
    Pop-Location
}
