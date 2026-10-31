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
REM    - ONNX recognition on its own thread. The bottom-right corner
REM      of the overlay shows:  ONNX 15box 125ms  combo!  ult
REM      hp-warn  combo3/4
REM    - big overlay panel so you can watch what it decides
REM      (overlay sits at y 0..300; the combo badge is at y 554
REM       and the character bars at y 1308 - nothing important
REM       gets covered)
REM    - live status page:  http://127.0.0.1:9120
REM    - 300 seconds
REM
REM  Keys: HOME = player takeover (fly stops pressing, panel stays),
REM        9 = combat mode toggle
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
    --overlay ^
    --tune --tune-port 9120 ^
    --forage --evac --auto-outcome --combo --skill-auto --ult ^
    --combo-hunger 0.0 --dodge-cool 1.2 ^
    --attack-hold 1.0 ^
    --onnx-hz 6 --onnx-conf 0.22 ^
    --seconds 300 ^
    --stuck-sec 3.0 --escape-grace 10 ^
    --out out\combat_test.csv > out\combat_test.log 2>&1

echo.
echo   Done. Log: out\combat_test.log
echo   Live page was: http://127.0.0.1:9120
echo.
pause
