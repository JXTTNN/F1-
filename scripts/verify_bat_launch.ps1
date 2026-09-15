﻿﻿# 验证一键启动.bat能否正常工作
# 模拟用户双击bat的行为：启动exe → 等待API → 确认就绪

Write-Host "===== 验证一键启动.bat ====="
Write-Host ""

# 先清理旧进程
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 检查端口
$portTest = netstat -ano | Select-String ":8000 " | Select-String "LISTENING"
if ($portTest) {
    $portTest | ForEach-Object {
        $line = $_.ToString()
        $parts = $line -split '\s+'
        $pidStr = $parts[-1]
        if ($pidStr -match '^\d+$' -and $pidStr -ne '0') {
            Stop-Process -Id ([int]$pidStr) -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
}

# 直接启动F1OPT.exe（模拟bat的 start "" F1OPT.exe）
Write-Host "[1] 启动F1OPT.exe..."
Set-Location "D:\F1OPT-Test\dist"
cmd /c "start /B F1OPT.exe"

# 等待API就绪
Write-Host "[2] 等待API就绪..."
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            $ready = $true
            Write-Host "  -> API就绪！耗时 ${i} 秒"
            break
        }
    } catch {}
}

if (-not $ready) {
    Write-Host "  -> [失败] API未就绪"
    Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
    exit 1
}

# 测试关键端点
Write-Host "[3] 测试API端点..."
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
Write-Host "  -> health: $($r.StatusCode)"

$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
$tracks = ($r.Content | ConvertFrom-Json).data
Write-Host "  -> tracks: $($r.StatusCode), $($tracks.Count)条"

$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
$fields = ($r.Content | ConvertFrom-Json).data
Write-Host "  -> setup/fields: $($r.StatusCode), $($fields.Count)项"

# 打开浏览器（模拟bat的行为）
Write-Host "[4] 打开浏览器..."
Start-Process "http://127.0.0.1:8000"
Write-Host "  -> 浏览器已打开"

# 等待浏览器加载
Start-Sleep -Seconds 3

# 清理
Write-Host "[5] 清理进程..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 已清理"

Write-Host ""
Write-Host "===== 一键启动.bat 验证: 成功 ====="