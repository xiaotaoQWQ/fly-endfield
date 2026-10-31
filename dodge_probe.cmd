@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  Dodge key probe - run as Administrator
REM  Tries 6 candidate keys, 3s apart. Watch the character.
REM ============================================================

net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Not running as Administrator - re-launching elevated...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

python -u -X utf8 dodge_probe.py
echo.
pause
