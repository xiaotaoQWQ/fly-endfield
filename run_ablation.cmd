@echo off
REM %~dp0 = 本脚本所在目录。★ 别写死绝对路径 —— 文件夹改名过一次
REM （E:\终末地 → E:\zmd），写死的路径当场全废。
cd /d "%~dp0"
set PY=python -X utf8 endfield_fly.py
set COMMON=--seconds 60 --hz 3 --turn-gain 3.5 --eye main --no-escape --eye-crop "300,530,1960,550" --eye-ds 4

echo ==== A: skip-center only (old minimap pairs) ====
%PY% %COMMON% --eye-skip "0.38,0.03,0.62,1.0" --out out\abl_A.csv > out\abl_A.txt 2>&1

echo ==== B: new pairs only (no skip-center) ====
%PY% %COMMON% --eye-skip none --pairs out\vote_pairs_main.npz --out out\abl_B.csv > out\abl_B.txt 2>&1

echo ==== C: skip-center + new pairs ====
%PY% %COMMON% --eye-skip "0.38,0.03,0.62,1.0" --pairs out\vote_pairs_main.npz --out out\abl_C.csv > out\abl_C.txt 2>&1

echo ALL_DONE
