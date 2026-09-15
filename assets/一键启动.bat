@echo off
chcp 65001 >nul
title F1OPT 赛车调教优化助手

REM 自动定位 F1OPT.exe 所在目录
set "EXE_DIR=%~dp0"
if not exist "%EXE_DIR%\F1OPT.exe" (
    set "EXE_DIR=%~dp0..\dist"
)
if not exist "%EXE_DIR%\F1OPT.exe" (
    echo [失败] 未找到 F1OPT.exe
    echo 请确保 F1OPT.exe 在同一目录或 dist 目录中。
    echo.
    pause
    exit /b 1
)

echo ============================================
echo   F1OPT 赛车调教优化助手
echo ============================================
echo.

REM 检查端口 8000 是否已被占用
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

echo 正在启动 F1OPT，请稍候...
echo.

REM 切换到 exe 所在目录（Nuitka onefile 需要正确的工作目录）
cd /d "%EXE_DIR%"

REM 直接调用 exe（不使用 start，因为 start 与 Nuitka onefile GUI exe 不兼容）
REM BAT 窗口会保持打开，关闭窗口即停止 F1OPT
F1OPT.exe
