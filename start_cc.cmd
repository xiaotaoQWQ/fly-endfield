@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM ============================================================
REM  Crisis Contract (危机合约) launcher  -- run as Administrator
REM
REM  Same engine as test_combat.cmd, plus:
REM    - 危机合约调试窗口（本地 tkinter，不是网页）
REM      所有冷却可拖到 0 / 一键全部归零 / 闪避开-关 / 实时状态
REM    - cooldowns preset to 0 for max output
REM
REM  Keys: HOME = player takeover,  9 = combat mode toggle
REM  Manual start: 只弹调试窗口，点「启动」才开始跑果蝇
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

echo   会弹出「果蝇全脑驾驶终末地」窗口：参数区最上面就是「危机合约 · 专用」
echo   叠加面板: HOME = 玩家接管 / 9 = 战斗模式
echo.

python -u -X utf8 endfield_fly.py ^
    --hz 2.5 ^
    --motor pop ^
    --eye main --eye-hz 30 ^
    --overlay --overlay-scale 1.0 --overlay-center --overlay-strip-row --overlay-nostatus ^
    --cc-debug ^
    --forage --auto-outcome --combo --skill-auto --ult ^
    --always-combat --combat ^
    --dodge-primary onnx --dodge-onnx-conf 0.15 --combo-hunger 0.0 --combo-cool 0 --combo-cool-full 0 --dodge-cool 0.6 ^
    --attack-hold 1.0 ^
    --onnx-hz 6 --onnx-conf 0.22 ^
    --seconds 600 ^
    --stuck-sec 3.0 --escape-grace 10 ^
    --out out\cc_run.csv > out\cc_run.log 2>&1

echo.
echo   Done. Log: out\cc_run.log
echo   Debug window was: the app window (危机合约 block on top)
echo.
pause
