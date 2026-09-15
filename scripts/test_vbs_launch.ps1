﻿# 测试 VBS 脚本能否正常启动 F1OPT.exe
# 步骤：1.清理残留进程 2.运行VBS 3.等待并检查进程和API

$distDir = "D:\F1OPT-Test\dist"
$vbsPath = "$distDir\F1OPT桌面启动.vbs"

Write-Host "=== VBS启动测试 ===" -ForegroundColor Cyan
Write-Host ""

# 1. 清理残留进程
Write-Host "[1] 清理残留进程..." -ForegroundColor Yellow
$existing = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Stop-Process -Force
    Write-Host "    已终止 $($existing.Count) 个残留进程" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
} else {
    Write-Host "    无残留进程" -ForegroundColor Green
}

# 2. 运行 VBS 脚本（用 wscript）
Write-Host "[2] 运行 VBS 脚本..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath`"" -NoNewWindow
Write-Host "    wscript 已启动，等待 API 就绪..." -ForegroundColor Yellow

# 3. 等待并检查 API（最多30秒）
Write-Host "[3] 等待 API 就绪..." -ForegroundColor Yellow
$apiReady = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($response.code -eq 200) {
            $apiReady = $true
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    API 就绪！耗时 ${elapsed} 秒" -ForegroundColor Green
            Write-Host "    响应: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 5 -eq 0) {
            Write-Host "    ${i}秒... API 尚未就绪" -ForegroundColor DarkGray
        }
    }
}

# 4. 检查 F1OPT 进程
Write-Host ""
Write-Host "[4] 检查 F1OPT 进程..." -ForegroundColor Yellow
$f1optProc = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optProc) {
    Write-Host "    F1OPT 进程存在！" -ForegroundColor Green
    Write-Host "    PID: $($f1optProc.Id)" -ForegroundColor Green
    Write-Host "    启动时间: $($f1optProc.StartTime)" -ForegroundColor Green
} else {
    Write-Host "    F1OPT 进程不存在！" -ForegroundColor Red
}

# 5. 检查 cmd 进程（VBS启动的中间cmd）
Write-Host ""
Write-Host "[5] 检查 cmd 进程..." -ForegroundColor Yellow
$cmdProcs = Get-Process -Name "cmd" -ErrorAction SilentlyContinue
if ($cmdProcs) {
    Write-Host "    cmd 进程数: $($cmdProcs.Count)" -ForegroundColor Yellow
    foreach ($p in $cmdProcs) {
        Write-Host "    PID: $($p.Id), 启动时间: $($p.StartTime)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "    无 cmd 进程" -ForegroundColor Green
}

# 6. 检查 wscript 进程
Write-Host ""
Write-Host "[6] 检查 wscript 进程..." -ForegroundColor Yellow
$wscriptProcs = Get-Process -Name "wscript" -ErrorAction SilentlyContinue
if ($wscriptProcs) {
    Write-Host "    wscript 进程数: $($wscriptProcs.Count)" -ForegroundColor Yellow
    foreach ($p in $wscriptProcs) {
        Write-Host "    PID: $($p.Id), 启动时间: $($p.StartTime)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "    无 wscript 进程（已正常退出）" -ForegroundColor Green
}

# 7. 总结
Write-Host ""
Write-Host "=== 测试总结 ===" -ForegroundColor Cyan
if ($apiReady -and $f1optProc) {
    Write-Host "✅ VBS脚本成功启动F1OPT.exe，API已就绪" -ForegroundColor Green
    Write-Host "   窗口模式 1（正常窗口）可以正常工作" -ForegroundColor Green
} elseif ($f1optProc -and -not $apiReady) {
    Write-Host "⚠️ F1OPT进程存在但API未就绪" -ForegroundColor Yellow
} elseif (-not $f1optProc -and $apiReady) {
    Write-Host "⚠️ API就绪但F1OPT进程名可能不同" -ForegroundColor Yellow
} else {
    Write-Host "❌ VBS脚本未能启动F1OPT.exe" -ForegroundColor Red
    Write-Host "   需要进一步诊断" -ForegroundColor Red
}