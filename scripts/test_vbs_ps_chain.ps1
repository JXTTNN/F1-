﻿# 测试 VBS → PowerShell(有控制台) → cmd /c BAT → start /B exe
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== VBS → PowerShell → cmd → BAT → exe ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 创建BAT脚本
$batContent = @"
@echo off
cd /d "%~dp0"
start /B F1OPT.exe
"@
$batPath = "$distDir\F1OPT后台启动.bat"
$batContent | Out-File -FilePath $batPath -Encoding ASCII

# 创建VBS脚本：启动PowerShell（最小化窗口），PowerShell运行BAT后保持运行
$testVbs = @"
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\F1OPT后台启动.bat"
psCmd = "powershell.exe -NoProfile -WindowStyle Minimized -Command ""cmd /c '" & batPath & "'; Start-Sleep -Seconds 999999"""
WshShell.Run psCmd, 7, False
WScript.Sleep 10000
WshShell.Run "http://127.0.0.1:8000"
"@
$testVbsPath = "$distDir\test_ps_vbs2.vbs"
$testVbs | Out-File -FilePath $testVbsPath -Encoding ASCII

Write-Host "[1] VBS → PowerShell(Minimized) → cmd → BAT → exe..." -ForegroundColor Yellow
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
Write-Host "[2] 检查进程..." -ForegroundColor Yellow
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

$psProc = Get-Process -Name "powershell" -ErrorAction SilentlyContinue
if ($psProc) {
    Write-Host "    PowerShell进程数: $($psProc.Count)" -ForegroundColor Yellow
}

# 清理测试VBS
Remove-Item $testVbsPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan