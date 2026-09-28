# start.ps1 - Invoice OCR Streamlit Workbench Launcher
$ErrorActionPreference = "Continue"

# Resolve project root from script directory
$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# Set UTF-8 encoding for console output
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "      Invoice OCR - Streamlit Workbench Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Project Directory: $ProjectRoot"
Write-Host ""

# Detect Python in virtual environment
$PythonExe = $null

# 1. Check if .venv has Streamlit
$dotVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $dotVenvPython) {
    & $dotVenvPython -c "import streamlit" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $PythonExe = $dotVenvPython
    }
}

# 2. Check if venv has Streamlit (Python 3.11 with full GPU OCR stack)
if (-not $PythonExe) {
    $venvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        & $venvPython -c "import streamlit" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonExe = $venvPython
        }
    }
}

# 3. Fallback
if (-not $PythonExe) {
    if (Test-Path -LiteralPath $dotVenvPython) {
        $PythonExe = $dotVenvPython
    } elseif (Test-Path -LiteralPath $venvPython) {
        $PythonExe = $venvPython
    }
}

if (-not $PythonExe) {
    Write-Host "[ERROR] Virtual environment Python executable not found!" -ForegroundColor Red
    Write-Host "Looked for:" -ForegroundColor Red
    Write-Host "  - $dotVenvPython" -ForegroundColor Red
    Write-Host "  - $venvPython" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

$AppEntry = Join-Path $ProjectRoot "app\ui\workbench.py"
if (-not (Test-Path -LiteralPath $AppEntry)) {
    Write-Host "[ERROR] Streamlit entry point not found: $AppEntry" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host "Python Runtime : $PythonExe" -ForegroundColor Green
Write-Host "UI Entry Point : $AppEntry" -ForegroundColor Green
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
Write-Host "(The browser will open automatically. Keep this window open.)" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

& $PythonExe -m streamlit run $AppEntry

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Red
    Write-Host "[ERROR] Streamlit exited with code $LASTEXITCODE." -ForegroundColor Red
    Write-Host "==========================================================" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit..."
