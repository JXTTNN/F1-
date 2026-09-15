﻿# 用cscript正确运行VBS脚本，捕获错误
# cscript的正确选项是 //nologo（不是//nologon）

Write-Host "===== 用cscript运行VBS脚本 ====="
Write-Host ""

# 先清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 用cscript运行VBS脚本（//nologo 抑制banner，//T:30 30秒超时）
Write-Host "[1] 运行 cscript //nologo //T:30 VBS脚本..."
Write-Host "    工作目录: D:\F1OPT-Test\dist"
Write-Host ""

Set-Location "D:\F1OPT-Test\dist"
$result = & cscript //nologo //T:30 "F1OPT桌面启动.vbs" 2>&1
Write-Host "cscript退出码: $LASTEXITCODE"
Write-Host "cscript输出:"
Write-Host $result

# 检查F1OPT进程是否启动了
Start-Sleep -Seconds 3
Write-Host ""
Write-Host "[2] 检查F1OPT进程..."
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "  -> F1OPT进程已启动！PID=$($f1opt.Id)"
    
    # 测试API
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
        Write-Host "  -> API响应: $($r.StatusCode) OK"
    } catch {
        Write-Host "  -> API未响应"
    }
} else {
    Write-Host "  -> F1OPT进程未启动"
}

# 检查端口
Write-Host ""
Write-Host "[3] 检查端口8000..."
$portTest = netstat -ano | Select-String ":8000 " | Select-String "LISTENING"
if ($portTest) {
    Write-Host "  -> 端口8000正在监听"
} else {
    Write-Host "  -> 端口8000未监听"
}

# 清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "===== 完成 ====="