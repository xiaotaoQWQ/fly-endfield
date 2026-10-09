@echo off
REM ============================================================
REM  Elevated worker -- launched by ..\start_here.cmd
REM
REM  Do NOT double-click this directly unless you already know
REM  Endfield is running; the checker lives in the parent script.
REM
REM  !! KEEP THIS FILE PURE ASCII !!  (see start_here.cmd for why)
REM
REM  Runs in --panel mode: opens the application window and WAITS.
REM  Nothing drives the game until you press Start, and Start stays
REM  disabled unless Endfield is detected.
REM
REM  Runs as admin so the Interception driver can inject input.
REM  Mouse BUTTONS cannot go through Interception -- those use
REM  SendInput instead (see win_click.py).
REM ============================================================

setlocal
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM ---- control port; pass a number as arg 1 to override ----
set PORT=%~1
if "%PORT%"=="" set PORT=8791

REM ---- keep a persistent copy of everything the program prints ----
REM  The console window is easy to lose and impossible to read after the
REM  fact. Tee it to a file so a failure can be diagnosed later.
REM  (cmd has no tee; redirect and use "type" afterwards.)
if not exist "%~dp0out" mkdir "%~dp0out"
set LOG=%~dp0out\console.log
set EF_DEBUG_UI=1
echo. > "%LOG%"

echo.
echo   ============================================================
echo    Fly brain driving Endfield  --  control window
echo   ============================================================
echo.
echo    An application window is opening.
echo    Press Start in it to begin. It stays disabled unless
echo    Endfield is detected.
echo.
echo    Browser copy of the same panel:
echo      http://127.0.0.1:%PORT%/
echo.
echo    END key in game also stops. Closing the window quits.
echo.
echo    Full console log: %LOG%
echo   ============================================================
echo.

python -u -X utf8 endfield_fly.py ^
  --panel --tune-port %PORT% ^
  --hz 2.5 --motor pop --eye main --eye-hz 30 ^
  --overlay --stuck-sec 3.0 --escape-grace 10 ^
  --forage --evac --auto-outcome --combo --skill-auto --ult ^
  --attack-hold 1.0 ^
  --out out\run.csv >> "%LOG%" 2>&1

echo.
echo   ---- exited (code %ERRORLEVEL%) ----
echo   Console log : %LOG%
echo   Log / CSV   : %~dp0out\run.csv
echo   Brain patch : %~dp0out\fly_brain_patch.npz
echo.
echo   ---- last 20 lines ----
powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 20"
echo.
pause
endlocal
