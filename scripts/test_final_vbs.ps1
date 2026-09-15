﻿# 测试更新后的正式VBS脚本
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== 测试正式VBS脚本（Exec版）===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 运行正式VBS脚本
Write-Host "[1] 运行正式VBS脚本..." -ForegroundColor Yellow
$startTime = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$distDir\F1OPT桌面启动.vbs`"" -NoNewWindow

# 等待API
$apiReady = $false
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
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

# 测试API端点
if ($apiReady) {
    Write-Host ""
    Write-Host "[3] 测试API端点..." -ForegroundColor Yellow
    
    $r1 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "    health: $($r1.StatusCode)" -ForegroundColor Green
    
    $r2 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
    $tracks = ($r2.Content | ConvertFrom-Json).data
    Write-Host "    tracks: $($r2.StatusCode), $($tracks.Count)条" -ForegroundColor Green
    
    $r3 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
    $fields = ($r3.Content | ConvertFrom-Json).data
    Write-Host "    setup/fields: $($r3.StatusCode), $($fields.Count)项" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan