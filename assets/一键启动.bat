@echo off
chcp 65001 >nul
title F1OPT 赛车调教优化助手

REM ── 自动定位仓库根目录（本脚本位于 assets\ 内，路径无关）──
set "ROOT=%~dp0.."
cd /d "%ROOT%"

REM ── 定位虚拟环境 Python ──
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [失败] 未找到虚拟环境 Python：%PY%
    echo 请先完成安装（在仓库根目录）：
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo ============================================
echo   F1OPT 赛车调教优化助手
echo   地址：http://127.0.0.1:8000（浏览器将自动打开）
echo   关闭本窗口即停止服务
echo ============================================
echo.

REM ── 端口 8000 占用清理 ──
netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [警告] 端口 8000 已被占用，正在尝试关闭旧进程...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
        taskkill /PID %%a /F >nul 2>nul
    )
    timeout /t 2 /nobreak >nul 2>nul
    netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [失败] 端口 8000 仍被占用，无法启动。
        echo 请手动关闭占用端口的程序后重试。
        echo.
        pause
        exit /b 1
    )
    echo [已清理] 端口已释放。
    echo.
)

REM ── 启动（cli 会自动打开浏览器）──
"%PY%" -m setup_tuner.cli
pause
