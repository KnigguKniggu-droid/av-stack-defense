@echo off
title AV Stack Defense - Live Visual
cd /d "%~dp0"
set "PATH=%PATH%;C:\iverilog\bin"
echo Running detectors and rendering the live visual dashboard...
echo (this takes ~30 seconds; the animated GPS sim is generated fresh)
echo.
python viz\build.py
echo.
echo Dashboard opened in your browser. Press any key to close this window.
pause >nul
