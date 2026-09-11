[CmdletBinding()]
param(
    [switch]$SkipOcr,
    [switch]$PrepareOnly,
    [string]$ImageInput = 'image',
    [string]$OcrDirectory = 'data/qwen-ocr',
    [string]$Output = 'structure.json',
    [string]$PythonPath = '.venv/Scripts/python.exe',
    [string]$OllamaUrl = 'http://127.0.0.1:11435',
    [string]$VisionModel = 'qwen3-vl:4b-instruct',
    [string]$TextModel = 'qwen3:4b-instruct-2507-q4_K_M',
    [string]$Currency,
    [string]$RunDir,
    [ValidateRange(1, 2147483647)][int]$NumCtx = 32768,
    [ValidateRange(1, 2147483647)][int]$NumPredict = 12288,
    [ValidateRange(1, 2147483647)][int]$Timeout = 900,
    [ValidateRange(1, 2147483647)][int]$OcrNumCtx = 8192,
    [ValidateRange(1, 2147483647)][int]$OcrNumPredict = 4096,
    [ValidateRange(1, 2147483647)][int]$OcrTimeout = 600
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found at '$PythonPath'. Create .venv and install requirements.txt; see docs/ocr.md."
    }
    if ($NumPredict -ge $NumCtx) {
        throw 'NumPredict must be smaller than NumCtx to leave room for the OCR prompt.'
    }
    if (-not ($SkipOcr -or $PrepareOnly)) {
        if ($OcrNumPredict -ge $OcrNumCtx) {
            throw 'OcrNumPredict must be smaller than OcrNumCtx.'
        }
        Write-Output 'Transcribing menu images with Qwen vision...'
        $ocrArguments = @(
            '-m', 'app.services.ocr',
            '--input', $ImageInput,
            '--output', $OcrDirectory,
            '--model', $VisionModel,
            '--ollama-url', $OllamaUrl,
            '--num-ctx', $OcrNumCtx,
            '--num-predict', $OcrNumPredict,
            '--timeout', $OcrTimeout
        )
        & $PythonPath @ocrArguments
        if ($LASTEXITCODE -ne 0) {
            throw "OCR failed with exit code $LASTEXITCODE. Menu generation was not started."
        }
    }

    $structureArguments = @(
        '-m', 'app.services.structure_menu',
        '--input', $OcrDirectory,
        '--output', $Output,
        '--model', $TextModel,
        '--ollama-url', $OllamaUrl,
        '--num-ctx', $NumCtx,
        '--num-predict', $NumPredict,
        '--timeout', $Timeout
    )
    if ($Currency) { $structureArguments += @('--currency', $Currency) }
    if ($RunDir) { $structureArguments += @('--run-dir', $RunDir) }
    if ($PrepareOnly) {
        $structureArguments += '--prepare-only'
        Write-Output 'Preparing generation artifacts from existing OCR; no model will be loaded.'
    } else {
        Write-Output 'Generating the consolidated menu with Qwen...'
    }
    & $PythonPath @structureArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Menu generation failed with exit code $LASTEXITCODE. Inspect the reported run directory."
    }
} finally {
    Pop-Location
}
