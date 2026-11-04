@echo off
setlocal
title F1OPT Portable

REM ============================================================
REM  F1OPT portable launcher - keep this file next to F1OPT.exe
REM
REM  Design note: the launcher runs F1OPT.exe in the FOREGROUND.
REM  The app is built with a console, so this window IS the log
REM  window; closing it stops the app - no orphan process, no
REM  "double click and nothing happens". An earlier version
REM  started the exe detached and then died itself, which made the
REM  browser open a moment before the service disappeared.
REM
REM  This file is deliberately ASCII-only: cmd.exe reads a batch
REM  file in fixed-size chunks and misparses a multi-byte character
REM  straddling a chunk boundary (reproduced on a zh-CN system with
REM  both UTF-8 and GBK). The app's own console output is Chinese.
REM ============================================================

set "ROOT=%~dp0"
set "ROOTDIR=%ROOT:~0,-1%"
cd /d "%ROOTDIR%"

set "PORT=8000"
set "PANEL=http://127.0.0.1:%PORT%"
set "HEALTHURL=%PANEL%/api/v1/health"

echo ============================================
echo   F1OPT Portable
echo   Folder : %ROOTDIR%
echo   Panel  : %PANEL%
echo ============================================
echo.

if exist "%ROOTDIR%\F1OPT.exe" goto have_exe
echo [FAIL] F1OPT.exe not found next to this launcher.
echo        Current folder: %ROOTDIR%
echo.
pause
exit /b 1

:have_exe
where curl >nul 2>nul
if errorlevel 1 goto run_app

curl -s -o nul --max-time 2 "%HEALTHURL%" >nul 2>nul
if errorlevel 1 goto run_app
echo [INFO] An instance is already running - opening the panel.
start "" "%PANEL%"
ping -n 4 127.0.0.1 >nul
exit /b 0

:run_app
echo [INFO] Starting F1OPT.exe.
echo        Keep this window open: it is the app log.
echo        Closing it stops the app. The browser opens by itself.
echo.
"%ROOTDIR%\F1OPT.exe"
set "RC=%errorlevel%"
echo.
echo [INFO] F1OPT exited with code %RC%.
if "%RC%"=="0" goto done
echo        Scroll up for the error. Full log: %ROOTDIR%\data\logs\f1opt.log
echo.
pause

:done
exit /b %RC%
