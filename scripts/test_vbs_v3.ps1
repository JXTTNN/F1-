﻿# 用之前验证过的方式启动F1OPT
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== 用验证过的方式启动 ===" -ForegroundColor Cyan

# 先清理
Write-Host "[0] 清理..." -ForegroundColor Yellow
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 2
    Write-Host "    已清理旧进程" -ForegroundColor Yellow
}

# 切换到dist目录（关键！）
Set-Location $distDir
Write-Host "[1] 当前目录: $(Get-Location)" -ForegroundColor Green

# 用验证过的方式启动
Write-Host "[2] cmd /c start /B F1OPT.exe..." -ForegroundColor Yellow
$startTime = Get-Date
cmd /c "start /B F1OPT.exe"
Write-Host "    命令已执行" -ForegroundColor Green

# 等待API
Write-Host "[3] 等待API..." -ForegroundColor Yellow
$apiReady = $false
for ($i = 1; $i -le 15; $i++) {
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
        if ($i % 3 -eq 0) {
            Write-Host "    ${i}秒..." -ForegroundColor DarkGray
        }
    }
}

# 检查进程
Write-Host ""
Write-Host "[4] 检查进程..." -ForegroundColor Yellow
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id), 启动时间: $($f1opt.StartTime)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT 未运行" -ForegroundColor Red
}

if ($apiReady) {
    Write-Host ""
    Write-Host "✅ 启动成功！现在测试 VBS 脚本..." -ForegroundColor Green
    
    # 清理后测试VBS
    Write-Host ""
    Write-Host "[5] 清理进程，测试 VBS..." -ForegroundColor Yellow
    $f1opt | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    
    # 确认已清理
    $check = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
    if ($check) {
        Write-Host "    进程仍在，等待..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        $check = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
    }
    if (-not $check) {
        Write-Host "    已清理" -ForegroundColor Green
    } else {
        Write-Host "    无法清理，继续测试VBS" -ForegroundColor Yellow
    }
    
    # 运行VBS
    Write-Host "[6] 运行 VBS 脚本..." -ForegroundColor Yellow
    $vbsStart = Get-Date
    Start-Process -FilePath "wscript.exe" -ArgumentList "`"$distDir\F1OPT桌面启动.vbs`"" -NoNewWindow
    
    # 等待API
    for ($i = 1; $i -le 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
            if ($r.code -eq 200) {
                $elapsed = ((Get-Date) - $vbsStart).TotalSeconds
                Write-Host "    ✅ VBS启动成功！API ${elapsed}秒就绪" -ForegroundColor Green
                break
            }
        } catch {
            if ($i % 5 -eq 0) {
                Write-Host "    ${i}秒..." -ForegroundColor DarkGray
            }
        }
    }
    
    $f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
    if ($f1opt2) {
        Write-Host "    ✅ F1OPT PID: $($f1opt2.Id)" -ForegroundColor Green
    } else {
        Write-Host "    ❌ VBS启动后F1OPT未运行" -ForegroundColor Red
        Write-Host "    需要诊断VBS的WshShell.Run行为" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan