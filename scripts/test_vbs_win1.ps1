﻿# 测试 VBS(窗口模式1) → BAT(直接运行exe)
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== VBS(窗口1) → BAT(直接exe) ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# BAT脚本：直接运行exe（不用start /B）
$batContent = @"
@echo off
cd /d "%~dp0"
F1OPT.exe
"@
$batPath = "$distDir\F1OPT后台启动.bat"
$batContent | Out-File -FilePath $batPath -Encoding ASCII

# VBS脚本：用窗口模式1（正常窗口）启动BAT
$testVbs = @"
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\F1OPT后台启动.bat"
WshShell.CurrentDirectory = scriptDir
WshShell.Run batPath, 1, False
WScript.Sleep 15000
WshShell.Run "http://127.0.0.1:8000"
"@
$testVbsPath = "$distDir\test_win1_vbs.vbs"
$testVbs | Out-File -FilePath $testVbsPath -Encoding ASCII

Write-Host "[1] VBS(窗口1) → BAT(直接exe)..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$testVbsPath`"" -NoNewWindow

# 等待API
$apiReady = $false
for ($i = 1; $i -le 25; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            $apiReady = $true
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

# 检查进程
Write-Host ""
Write-Host "[2] 检查 F1OPT 进程..." -ForegroundColor Yellow
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理测试VBS
Remove-Item $testVbsPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan