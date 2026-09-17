@echo off
chcp 65001 >nul
title F1OPT 桌面遥测接收器

REM ── 自动定位仓库根目录（本脚本位于 assets\ 内，路径无关）──
set "ROOT=%~dp0.."
cd /d "%ROOT%"

REM ── 优先用虚拟环境 Python（需含 tkinter）；否则回退系统 Python ──
REM 桌面接收器只用标准库（tkinter + socket + sqlite3），系统 Python 即可运行。
set "PY="
if exist "%ROOT%\.venv\Scripts\python.exe" (
    "%ROOT%\.venv\Scripts\python.exe" -c "import tkinter" >nul 2>&1 && set "PY=%ROOT%\.venv\Scripts\python.exe"
)
if not defined PY (
    py -3 -c "import tkinter" >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    python -c "import tkinter" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [失败] 未找到带 tkinter 的 Python。
    echo 桌面遥测接收器需要 tkinter（Python 安装时勾选 tcl/tk）。
    echo.
    pause
    exit /b 1
)

echo 使用解释器：%PY%
echo 数据目录：%ROOT%\data\recordings
echo 游戏内请开启 UDP 遥测，目标 127.0.0.1:20777
echo.
%PY% -m setup_tuner.collector
