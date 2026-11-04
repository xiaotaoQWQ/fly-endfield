@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  Crisis Contract (危机合约) launcher  -- run as Administrator
REM
REM  Same engine as test_combat.cmd, plus:
REM    - 危机合约调试台  http://127.0.0.1:9121
REM      所有冷却可调到 0 / 一键归零 / 实时状态（连携成-败、闪避三路、挨打、血量）
REM    - cooldowns preset to 0 for max output
REM
REM  Keys: HOME = player takeover,  9 = combat mode toggle
REM
REM  Log: out\cc_run.log
REM ============================================================

net session >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Not running as Administrator - re-launching elevated...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo.
echo   Running as Administrator - OK
echo.

if not exist out mkdir out
del /q out\cc_run.log 2>nul

echo   危机合约调试台: http://127.0.0.1:9121
echo   叠加面板: HOME = 玩家接管 / 9 = 战斗模式
echo.

python -u -X utf8 endfield_fly.py ^
    --hz 2.5 ^
    --motor pop ^
    --eye main --eye-hz 30 ^
    --overlay ^
    --tune --tune-port 9120 ^
    --cc-debug --cc-port 9121 ^
    --forage --evac --auto-outcome --combo --skill-auto --ult ^
    --combo-hunger 0.0 --combo-cool 0 --combo-cool-full 0 --dodge-cool 0.6 ^
    --attack-hold 1.0 ^
    --onnx-hz 6 --onnx-conf 0.22 ^
    --seconds 600 ^
    --stuck-sec 3.0 --escape-grace 10 ^
    --out out\cc_run.csv > out\cc_run.log 2>&1

echo.
echo   Done. Log: out\cc_run.log
echo   Debug page was: http://127.0.0.1:9121
echo.
pause
