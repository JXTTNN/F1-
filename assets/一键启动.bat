@echo off
chcp 65001 >nul
title F1OPT 赛车调教优化助手
echo 正在启动 F1OPT 赛车调教优化助手...
echo.
echo 浏览器将自动打开，请勿关闭此窗口。
echo 如需停止，请按 Ctrl+C 或关闭此窗口。
echo.
F1OPT.exe
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 启动失败！错误代码: %ERRORLEVEL%
    echo 请检查 F1OPT.exe 是否存在。
    pause
)