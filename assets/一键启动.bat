@echo off
chcp 65001 >nul
title F1OPT 赛车调教优化助手
echo ============================================
echo   F1OPT 赛车调教优化助手
echo ============================================
echo.
echo 正在启动，请稍候...
echo.

REM 启动 F1OPT.exe（GUI 模式，start 不等待返回）
start "" F1OPT.exe

REM 轮询 API 就绪（最多等 20 秒）
set /a count=0
:waitloop
timeout /t 1 /nobreak >nul 2>nul
set /a count+=1
powershell -NoProfile -Command "try{$r=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/v1/health' -UseBasicParsing -TimeoutSec 2;if($r.StatusCode -eq 200){exit 0}else{exit 1}}catch{exit 1}" >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :ready
if %count% LSS :20 goto :waitloop

echo [失败] 启动超时（20秒内 API 未就绪）
echo 请检查 F1OPT.exe 是否被杀毒软件拦截。
echo.
pause
exit /b 1

:ready
echo [成功] F1OPT 已启动！
echo.
echo   浏览器应已自动打开，如未打开请手动访问：
echo   http://127.0.0.1:8000
echo.
echo   停止服务：关闭此窗口或任务管理器结束 F1OPT.exe
echo.
echo   按任意键关闭此窗口（F1OPT 将继续在后台运行）...
pause >nul
exit /b 0
