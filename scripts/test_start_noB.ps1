﻿# 测试用 start（不带/B）启动exe
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== start（不带/B）方案 ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 测试1：cmd /c "start F1OPT.exe"（在PowerShell中直接运行）
Write-Host "[1] cmd /c start F1OPT.exe（不带/B）..." -ForegroundColor Yellow
Set-Location $distDir
$startTime = Get-Date
cmd /c "start F1OPT.exe"

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

$f1opt1 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt1) {
    Write-Host "    ✅ F1OPT PID: $($f1opt1.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
if ($f1opt1) {
    $f1opt1 | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
}

# 测试2：Start-Process cmd.exe /c "start F1OPT.exe"
Write-Host ""
Write-Host "[2] Start-Process cmd /c start F1OPT.exe..." -ForegroundColor Yellow
$startTime2 = Get-Date
Start-Process -FilePath "cmd.exe" -ArgumentList '/c', 'start F1OPT.exe' -WorkingDirectory $distDir -WindowStyle Normal

for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime2).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 3 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt2) {
    Write-Host "    ✅ F1OPT PID: $($f1opt2.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
if ($f1opt2) {
    $f1opt2 | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
}

# 测试3：VBS → cmd /c "start F1OPT.exe"
Write-Host ""
Write-Host "[3] VBS → cmd /c start F1OPT.exe..." -ForegroundColor Yellow

$batContent3 = @"
@echo off
cd /d "%~dp0"
start F1OPT.exe
"@
$batPath3 = "$distDir\test_start.bat"
$batContent3 | Out-File -FilePath $batPath3 -Encoding ASCII

$testVbs = @"
Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\test_start.bat"
WshShell.CurrentDirectory = scriptDir
WshShell.Run "cmd /c " & Chr(34) & batPath & Chr(34), 0, False
WScript.Sleep 15000
WshShell.Run "http://127.0.0.1:8000"
"@
$testVbsPath = "$distDir\test_start_vbs.vbs"
$testVbs | Out-File -FilePath $testVbsPath -Encoding ASCII

$startTime3 = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$testVbsPath`"" -NoNewWindow

for ($i = 1; $i -le 25; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime3).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt3 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt3) {
    Write-Host "    ✅ F1OPT PID: $($f1opt3.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
Remove-Item $batPath3, $testVbsPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan