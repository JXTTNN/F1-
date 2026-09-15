﻿# 排除VBS因素：直接在PowerShell中用cmd /c运行BAT
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== 直接cmd /c运行BAT ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 测试1：cmd /c "start /B F1OPT.exe"（已知成功的方式）
Write-Host "[1] 已知成功方式: cmd /c start /B..." -ForegroundColor Yellow
Set-Location $distDir
$startTime = Get-Date
cmd /c "start /B F1OPT.exe"

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {}
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

# 测试2：cmd /c BAT（BAT中用 start /B F1OPT.exe）
Write-Host ""
Write-Host "[2] cmd /c BAT(start /B)..." -ForegroundColor Yellow
$batContent1 = @"
@echo off
cd /d "%~dp0"
start /B F1OPT.exe
"@
$batPath1 = "$distDir\test_bat1.bat"
$batContent1 | Out-File -FilePath $batPath1 -Encoding ASCII

$startTime2 = Get-Date
cmd /c "`"$batPath1`""

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {}
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

# 测试3：cmd /c BAT（BAT中直接 F1OPT.exe，不用start）
Write-Host ""
Write-Host "[3] cmd /c BAT(直接exe)..." -ForegroundColor Yellow
$batContent2 = @"
@echo off
cd /d "%~dp0"
F1OPT.exe
"@
$batPath2 = "$distDir\test_bat2.bat"
$batContent2 | Out-File -FilePath $batPath2 -Encoding ASCII

$startTime3 = Get-Date
# 用Start-Process异步启动BAT（因为直接运行会阻塞）
Start-Process -FilePath "cmd.exe" -ArgumentList '/c', "`"$batPath2`"" -WorkingDirectory $distDir -WindowStyle Normal

for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
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

# 清理测试BAT
Remove-Item $batPath1, $batPath2 -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan