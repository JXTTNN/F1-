﻿# 测试用PowerShell作为中间层
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== PowerShell中间层方案 ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 方案1：Start-Process powershell → cmd /c start /B
Write-Host "[1] PowerShell中间层（隐藏窗口）..." -ForegroundColor Yellow
$startTime = Get-Date
$psCmd = "Set-Location '$distDir'; cmd /c `"start /B F1OPT.exe`""
Start-Process -FilePath "powershell.exe" -ArgumentList '-NoProfile', '-Command', $psCmd -WindowStyle Hidden

for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 3 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
if ($f1opt) {
    $f1opt | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
}

# 方案2：VBS → PowerShell → cmd /c start /B
Write-Host ""
Write-Host "[2] VBS → PowerShell → cmd..." -ForegroundColor Yellow

$testVbs = @"
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
psCmd = "powershell.exe -NoProfile -Command ""Set-Location '" & scriptDir & "'; cmd /c start /B F1OPT.exe"""
WshShell.Run psCmd, 0, False
WScript.Sleep 10000
WshShell.Run "http://127.0.0.1:8000"
"@
$testVbsPath = "$distDir\test_ps_vbs.vbs"
$testVbs | Out-File -FilePath $testVbsPath -Encoding ASCII

$startTime2 = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$testVbsPath`"" -NoNewWindow

for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime2).TotalSeconds
            Write-Host "    ✅ VBS→PS→cmd 启动成功！API ${elapsed}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt2) {
    Write-Host "    ✅ F1OPT PID: $($f1opt2.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理测试VBS
Remove-Item $testVbsPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan