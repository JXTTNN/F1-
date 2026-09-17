@echo off
chcp 65001 >nul
title F1OPT 遥测链路诊断

REM ── 自动定位仓库根目录（本脚本位于 assets\ 内）──
set "ROOT=%~dp0.."
cd /d "%ROOT%"

REM ── 诊断脚本纯标准库，venv 即可；无 venv 回退系统 Python ──
set "PY="
if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not defined PY (
    py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    python -c "import sys" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [失败] 未找到可用的 Python。
    pause
    exit /b 1
)

echo 使用解释器：%PY%
echo.
%PY% scripts\diag_telemetry_udp.py %*
echo.
pause
