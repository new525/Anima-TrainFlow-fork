@echo off
setlocal
cd /d %~dp0

set "PY_EXE=%~dp0python_embeded\python.exe"

if not exist "%PY_EXE%" (
    echo [ERROR] Portable Python not found at:
    echo "%PY_EXE%"
    pause
    exit
)

echo Starting the ToriiGate 0.5 captioner...
echo (first use downloads llama.cpp and the model: see tools\toriigate-captioner\README.md)
echo.

"%PY_EXE%" tools\toriigate-captioner\captioner_ui.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Script crashed. Check the error message above.
    pause
)
