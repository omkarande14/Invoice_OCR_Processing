@echo off
setlocal
cd /d "%~dp0"

:: Set console code page to UTF-8 for unicode support (emojis, symbols)
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ==========================================================
echo       Invoice OCR - Streamlit Workbench Launcher
echo ==========================================================
echo Project Directory: %CD%
echo.

:: Detect Python in virtual environment (.venv or venv)
set "PYTHON_EXE="

:: 1. Check if .venv has Streamlit
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    )
)

:: 2. Check if venv has Streamlit (Python 3.11 with full GPU OCR stack)
if not defined PYTHON_EXE (
    if exist "%~dp0venv\Scripts\python.exe" (
        "%~dp0venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
        )
    )
)

:: 3. Fallback: use whichever python.exe exists
if not defined PYTHON_EXE (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    ) else if exist "%~dp0venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    )
)

:: Error handling: No Python found
if not defined PYTHON_EXE (
    echo [ERROR] Virtual environment Python executable not found!
    echo Looked for:
    echo   - "%~dp0.venv\Scripts\python.exe"
    echo   - "%~dp0venv\Scripts\python.exe"
    echo.
    echo Please make sure the virtual environment exists.
    echo.
    pause
    exit /b 1
)

:: Error handling: Entry point missing
set "APP_ENTRY=%~dp0app\ui\workbench.py"
if not exist "%APP_ENTRY%" (
    echo [ERROR] Streamlit entry point not found:
    echo   "%APP_ENTRY%"
    echo.
    pause
    exit /b 1
)

echo Python Runtime : %PYTHON_EXE%
echo UI Entry Point : %APP_ENTRY%
echo.
echo Starting Streamlit application...
echo (The browser will open automatically. Keep this window open.)
echo ==========================================================
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_ENTRY%"

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo [ERROR] Streamlit exited with error code %errorlevel%.
    echo Review the error log above to diagnose.
    echo ==========================================================
)

echo.
pause