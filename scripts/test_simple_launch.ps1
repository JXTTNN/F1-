﻿# 最简测试：直接用 cmd /c 启动 F1OPT
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== 最简启动测试 ===" -ForegroundColor Cyan

# 方式1：直接用 cmd /c（在 bash 工具的 PowerShell 中）
Write-Host "[1] cmd /c 方式..." -ForegroundColor Yellow
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList '/c', 'cd /d D:\F1OPT-Test\dist && start /B "" F1OPT.exe' -PassThru -WindowStyle Normal
Write-Host "    cmd PID: $($proc.Id)" -ForegroundColor Green

# 等待 API
for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "    ${i}秒..." -ForegroundColor DarkGray
    }
}

$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT 未运行" -ForegroundColor Red
}

Write-Host ""
# 方式2：直接运行 exe（设置工作目录）
Write-Host "[2] 直接运行 exe（设工作目录）..." -ForegroundColor Yellow
$proc2 = Start-Process -FilePath "$distDir\F1OPT.exe" -WorkingDirectory $distDir -PassThru -WindowStyle Normal
Write-Host "    exe PID: $($proc2.Id)" -ForegroundColor Green

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "    ${i}秒..." -ForegroundColor DarkGray
    }
}

$f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt2) {
    Write-Host "    ✅ F1OPT PID: $($f1opt2.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT 未运行" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan