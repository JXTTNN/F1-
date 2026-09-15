﻿# 验证修复后的VBS脚本
# 用cscript运行，检查是否有编译错误

Write-Host "===== 验证修复后的VBS脚本 ====="
Write-Host ""

# 先清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 用cscript运行VBS脚本（//nologo抑制banner，//T:30 30秒超时）
Write-Host "[1] 用cscript运行VBS脚本..."
Set-Location "D:\F1OPT-Test\dist"
$result = & cscript //nologo //T:30 "F1OPT桌面启动.vbs" 2>&1
Write-Host "  -> cscript退出码: $LASTEXITCODE"
if ($result) {
    Write-Host "  -> cscript输出:"
    $result | ForEach-Object { Write-Host "     $_" }
}

# 检查F1OPT进程
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
        
        $r2 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
        $tracks = ($r2.Content | ConvertFrom-Json).data
        Write-Host "  -> 赛道列表: $($tracks.Count)条"
        
        $r3 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
        $fields = ($r3.Content | ConvertFrom-Json).data
        Write-Host "  -> 调教参数: $($fields.Count)项"
    } catch {
        Write-Host "  -> API未响应: $($_.Exception.Message)"
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
Write-Host ""
Write-Host "[4] 清理..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 已清理"

Write-Host ""
Write-Host "===== 验证完成 ====="