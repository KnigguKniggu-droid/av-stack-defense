@echo off
title AV Stack Defense - Cross-Layer Harness
cd /d "%~dp0"
REM Make Icarus Verilog reachable so the FPGA layer runs.
set "PATH=%PATH%;C:\iverilog\bin"
echo Running the cross-layer AV defense harness...
echo.
python harness.py
echo.
echo ============================================================
echo Done. Press any key to close this window.
pause >nul
