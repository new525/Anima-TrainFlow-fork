@echo off
setlocal
cd /d %~dp0

set "PY_EXE=%~dp0python_embeded\python.exe"

if not exist "%PY_EXE%" (
    echo [ERROR] python_embeded folder not found!
    pause
    exit
)

echo Optional, for MOSS-Image: installs Triton, which the ConvRot INT8 option needs to run at full speed.
echo Only RTX 30 series and newer cards can use it; on an older card this says so and installs nothing.
echo Run Install_Requirements.bat first. Anima training does not need this.
echo.

"%PY_EXE%" tools\install_triton.py %*

echo.
pause
