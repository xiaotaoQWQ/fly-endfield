@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  Combat test - run as Administrator
REM
REM  Why admin: keyboard injection uses the Interception driver.
REM  Without admin the fly calculates fine but NO key reaches
REM  the game at all.
REM
REM  What this does:
REM    - real run (no --dry): actually presses keys
REM    - ONNX recognition enabled (threaded, ~6 Hz)
REM    - combo / skill / ultimate enabled
REM    - NO --overlay, so nothing covers the screen
REM    - 300 seconds
REM
REM  Log: out\combat_test.log
REM ============================================================

net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Not running as Administrator - re-launching elevated...
    echo.
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo.
echo   Running as Administrator - OK
echo.

if not exist out mkdir out
del /q out\combat_test.log 2>nul

python -u -X utf8 endfield_fly.py ^
    --hz 2.5 ^
    --motor pop ^
    --eye main --eye-hz 30 ^
    --forage --evac --auto-outcome --combo --skill-auto --ult ^
    --attack-hold 1.0 ^
    --onnx-hz 6 ^
    --seconds 300 ^
    --stuck-sec 3.0 --escape-grace 10 ^
    --out out\combat_test.csv > out\combat_test.log 2>&1

echo.
echo   Done. Log: out\combat_test.log
echo.
pause
