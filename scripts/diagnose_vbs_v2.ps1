﻿# 诊断 VBS 脚本启动失败的原因
$distDir = "D:\F1OPT-Test\dist"
$exePath = "$distDir\F1OPT.exe"

Write-Host "=== VBS 启动诊断 ===" -ForegroundColor Cyan

# 1. 直接用 cmd /c 执行（模拟VBS的命令）
Write-Host "[1] 用 cmd /c 直接执行..." -ForegroundColor Yellow
$cmd = "cd /d `"$distDir`" && start /B `"`" `"$exePath`""
Write-Host "    命令: cmd /c $cmd" -ForegroundColor Green
$startTime = Get-Date
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -WindowStyle Normal -PassThru | Select-Object Id, ProcessName | Format-Table
Write-Host "    cmd 已启动，等待 API..." -ForegroundColor Yellow

# 2. 等待 API
$apiReady = $false
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($response.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    ✅ API 就绪！耗时 ${elapsed} 秒" -ForegroundColor Green
            $apiReady = $true
            break
        }
    } catch {
        if ($i % 3 -eq 0) {
            Write-Host "    ${i}秒... 等待中" -ForegroundColor DarkGray
        }
    }
}

# 3. 检查进程
Write-Host ""
Write-Host "[2] 检查 F1OPT 进程..." -ForegroundColor Yellow
$f1optProc = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optProc) {
    Write-Host "    ✅ F1OPT 进程存在, PID: $($f1optProc.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT 进程不存在" -ForegroundColor Red
}

# 4. 如果直接cmd方式成功，说明问题在VBS的WshShell.Run
if ($apiReady) {
    Write-Host ""
    Write-Host "[3] 直接 cmd 方式成功，问题在 VBS 的 WshShell.Run" -ForegroundColor Yellow
    Write-Host "    需要检查 VBS 中 WshShell.Run 的行为" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 诊断完成 ===" -ForegroundColor Cyan