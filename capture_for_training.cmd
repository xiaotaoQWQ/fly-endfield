@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  Capture training frames for the "Si combo prompt" detector
REM
REM  Why: MaaEnd's model does not recognize Si's combo prompt
REM  (combo_raw stays 0). We cannot retrain their model - we only
REM  have the ONNX export, not the weights or the dataset.
REM
REM  So: train a small SECOND model that only recognizes Si's
REM  prompt, and run it alongside the main one.
REM
REM  This script only COLLECTS frames. It also fights normally.
REM
REM  BEFORE running:
REM    1. put Si in your team
REM    2. get into a fight where Si's combo prompt shows up
REM    3. keep fighting for the whole 300 seconds
REM
REM  Frames land in:  out\train_snaps\
REM  Expected count:  about 375 files (every 2nd tick)
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
if exist out\train_snaps rmdir /s /q out\train_snaps
mkdir out\train_snaps

echo   Collecting frames into out\train_snaps ...
echo.

python -u -X utf8 endfield_fly.py ^
    --hz 2.5 ^
    --motor pop ^
    --eye main --eye-hz 30 ^
    --tune --tune-port 9131 ^
    --forage --evac --auto-outcome --combo --skill-auto --ult ^
    --combo-hunger 0.0 --dodge-cool 1.2 ^
    --attack-hold 1.0 ^
    --onnx-hz 6 --onnx-conf 0.22 ^
    --snapdir out\train_snaps --snap-every 2 ^
    --seconds 300 ^
    --stuck-sec 3.0 --escape-grace 10 ^
    --out out\capture.csv > out\capture.log 2>&1

echo.
echo   Done.
echo   Frames: out\train_snaps
echo   Log:    out\capture.log
echo.
dir /b out\train_snaps | find /c /v "" > out\_snapcount.txt
set /p SNAPN=<out\_snapcount.txt
echo   Captured %SNAPN% frames.
echo.
pause
