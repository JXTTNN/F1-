﻿# 验证理论：cmd /k（不退出）+ Start-Process
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== cmd /k + Start-Process 验证 ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 测试1：Start-Process cmd /k "start /B F1OPT.exe"
Write-Host "[1] Start-Process cmd /k start /B..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "cmd.exe" -ArgumentList '/k', 'start /B F1OPT.exe' -WorkingDirectory $distDir -WindowStyle Normal

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

# 测试2：Start-Process cmd /k BAT（BAT中 start /B F1OPT.exe）
Write-Host ""
Write-Host "[2] Start-Process cmd /k BAT..." -ForegroundColor Yellow
$batContent = @"
@echo off
cd /d "%~dp0"
start /B F1OPT.exe
"@
$batPath = "$distDir\test_k.bat"
$batContent | Out-File -FilePath $batPath -Encoding ASCII

$startTime2 = Get-Date
Start-Process -FilePath "cmd.exe" -ArgumentList '/k', "`"$batPath`"" -WorkingDirectory $distDir -WindowStyle Normal

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
Remove-Item $batPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan