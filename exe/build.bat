@echo off
chcp 65001 >nul 2>&1
REM ============================================
REM  F1OPT EXE 构建脚本 (Windows)
REM  用法: 双击或命令行运行 build.bat
REM  产物: dist\f1opt\f1opt.exe
REM ============================================

echo ============================================
echo   F1OPT EXE 构建脚本
echo ============================================
echo.

REM --- 检查 Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python, 请先安装 Python 3.11+
    pause
    exit /b 1
)

REM --- 安装依赖 ---
REM 只装运行时依赖 (含 torch)，不装 dev 依赖 (pytest/ruff/mypy 与 EXE 无关)
echo [1/3] 安装依赖...
pip install -e . -q
pip install pyinstaller -q

REM --- 清理旧产物 ---
echo [2/3] 清理旧构建...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM --- 打包 ---
echo [3/3] 开始打包 (可能需要几分钟)...
pyinstaller exe\f1opt.spec --noconfirm

if exist "dist\f1opt\f1opt.exe" (
    echo.
    echo ============================================
    echo   构建成功!
    echo   产物: dist\f1opt\f1opt.exe
    echo ============================================
    echo.
    echo 测试运行:
    echo   dist\f1opt\f1opt.exe --help
) else (
    echo.
    echo [错误] 构建失败, 请检查上方日志
)
pause
