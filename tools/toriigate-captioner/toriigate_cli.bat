@echo off
rem Command-line captioner:  toriigate_cli.bat "C:\Images\MyDataset" [options]    (--help lists them)
setlocal
set "PY_EXE=%~dp0..\..\python_embeded\python.exe"
if not exist "%PY_EXE%" (
    echo [ERROR] Portable Python not found at "%PY_EXE%"
    exit /b 1
)
"%PY_EXE%" "%~dp0toriigate_captioner.py" %*
exit /b %errorlevel%
