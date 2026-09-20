@echo off
chcp 65001 >nul
title F1OPT 赛车调教优化助手（便携版）

REM ============================================================
REM  便携版启动器 —— 与 F1OPT.exe 放在同一目录（安装文件夹）
REM  1. 启动 exe（它会自动打开浏览器面板）
REM  2. 轮询 /api/v1/health 确认服务就绪
REM  3. 数据全部落在本目录内的 data\（录制 / 数据库 / 派生数据）
REM  注：不写服务地址到其它位置，也不使用 caret 续行符
REM ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "EXE=%ROOT%F1OPT.exe"
if not exist "%EXE%" (
    echo [失败] 未找到 F1OPT.exe（应与本启动器放在同一目录）
    echo        当前目录：%ROOT%
    echo.
    pause
    exit /b 1
)

set "PORT=8000"
set "HEALTHURL=http://127.0.0.1:%PORT%/api/v1/health"

echo ============================================
echo   F1OPT 赛车调教优化助手（便携版）
echo   安装目录：%ROOT%
echo   数据目录：%ROOT%data\
echo   面板地址：http://127.0.0.1:%PORT%
echo ============================================
echo.
echo 正在启动 F1OPT.exe ...

start "" "%EXE%"

echo 等待服务就绪（轮询 health 接口）...
set "READY="
for /l %%i in (1,1,60) do (
    if not defined READY (
        curl -s -o nul --max-time 2 "%HEALTHURL%" >nul 2>nul
        if not errorlevel 1 (
            set "READY=1"
            echo [就绪] 服务已响应 health 检查，尝试次数 %%i
        ) else (
            timeout /t 1 /nobreak >nul 2>nul
        )
    )
)

if not defined READY (
    echo [警告] 60 秒内未等到 health 就绪。
    echo        请查看是否有防火墙拦截，或手动打开 %HEALTHURL%
) else (
    echo.
    echo 浏览器应已自动打开。若未打开，请手动访问：
    echo   http://127.0.0.1:%PORT%
)

echo.
echo 关闭 F1OPT 请在本窗口按任意键后结束进程，或直接关闭 F1OPT.exe 窗口。

REM 打开面板（若浏览器未被 exe 自动打开）
if defined READY (
    start "" "%HEALTHURL%"
)

pause
