﻿# 测试 UseShellExecute=$false（继承标准流）
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== UseShellExecute=false 测试 ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 方案1：用 [System.Diagnostics.Process] + UseShellExecute=$false
Write-Host "[1] ProcessStartInfo(UseShellExecute=false)..." -ForegroundColor Yellow
$startTime = Get-Date

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = '/c "start /B F1OPT.exe"'
$psi.WorkingDirectory = $distDir
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $false

$p = [System.Diagnostics.Process]::Start($psi)
Write-Host "    cmd PID: $($p.Id)" -ForegroundColor Green

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

# 方案2：用 [System.Diagnostics.Process] 直接启动 F1OPT.exe（不用cmd）
Write-Host ""
Write-Host "[2] ProcessStartInfo 直接启动exe..." -ForegroundColor Yellow
$startTime2 = Get-Date

$psi2 = New-Object System.Diagnostics.ProcessStartInfo
$psi2.FileName = "$distDir\F1OPT.exe"
$psi2.WorkingDirectory = $distDir
$psi2.UseShellExecute = $false
$psi2.CreateNoWindow = $false

$p2 = [System.Diagnostics.Process]::Start($psi2)
Write-Host "    exe PID: $($p2.Id)" -ForegroundColor Green

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

# 方案3：用 cmd /c BAT（UseShellExecute=false）
Write-Host ""
Write-Host "[3] ProcessStartInfo cmd /c BAT..." -ForegroundColor Yellow
$batContent = @"
@echo off
cd /d "%~dp0"
start /B F1OPT.exe
"@
$batPath = "$distDir\test_psi.bat"
$batContent | Out-File -FilePath $batPath -Encoding ASCII

$startTime3 = Get-Date

$psi3 = New-Object System.Diagnostics.ProcessStartInfo
$psi3.FileName = "cmd.exe"
$psi3.Arguments = "/c `"$batPath`""
$psi3.WorkingDirectory = $distDir
$psi3.UseShellExecute = $false
$psi3.CreateNoWindow = $false

$p3 = [System.Diagnostics.Process]::Start($psi3)
Write-Host "    cmd PID: $($p3.Id)" -ForegroundColor Green

for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime3).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 3 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt3 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt3) {
    Write-Host "    ✅ F1OPT PID: $($f1opt3.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
Remove-Item $batPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan