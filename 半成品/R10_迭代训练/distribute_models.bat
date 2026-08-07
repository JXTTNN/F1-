@echo off
REM ============================================================
REM R10 exe 即插即用修复脚本（一次性分发）
REM 作用：把主项目 models/ 下最强的可用模型分发到 dist/F1LLM/models/
REM   —— 解决「双击 exe 后找不到 GGUF 而降级 MOCK」的核心兼容性
REM ============================================================
setlocal
chcp 65001 >nul

set "DST=dist\F1LLM\models"
set "SRC=models"

echo.
echo [R10] 分发模型到 exe 同目录 ...
if not exist "%DST%" mkdir "%DST%"

set "M1=qwen2.5-0.5b-instruct-q4_k_m.gguf"
if exist "%SRC%\%M1%" (
    copy /Y "%SRC%\%M1%" "%DST%\%M1%" >nul
    echo   [OK] 已分发 %M1%
) else (
    echo   [!!] 源模型不存在: %SRC%\%M1%
)

set "M2=qwen2.5-1.5b-instruct-q4_k_m.gguf"
if exist "%SRC%\%M2%" (
    copy /Y "%SRC%\%M2%" "%DST%\%M2%" >nul
    echo   [OK] 已分发 %M2%
)

echo.
echo [R10] dist\F1LLM\models 当前内容：
dir /B "%DST%" 2>nul
echo.
echo 完成：双击 dist\F1LLM\F1LLM.exe 即可本地即插即用（无需 API）。
