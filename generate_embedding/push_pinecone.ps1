# Syncs data/menu-embeddings.json vectors to Pinecone. No package
# installation or embedding generation is performed. Requires
# PINECONE_API_KEY to already be set (directly, or via a local .env file
# in the project root -- see .env and generate_embedding/pinecone.md); never pass it as
# a parameter.
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

    # Load .env (if present) without ever printing its contents; an
    # already-set environment variable always wins over the file.
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

    if (-not $DryRun -and -not $env:PINECONE_API_KEY) {
        throw 'PINECONE_API_KEY is not set. Fill it in in .env, or set it directly: $env:PINECONE_API_KEY = "..."'
    }

    $syncArguments = @(
        '-m', 'generate_embedding.pinecone_sync',
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
