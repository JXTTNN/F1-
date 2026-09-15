﻿﻿# 验证修复后的VBS启动方式
# 用PowerShell模拟VBS脚本中 cmd /c start /B 的启动方式

Write-Host "===== 验证修复后的VBS启动方式 ====="
Write-Host ""

# 清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

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

$scriptDir = "D:\F1OPT-Test\dist"
$exePath = "$scriptDir\F1OPT.exe"

# 模拟VBS脚本的启动命令：cmd /c cd /d "scriptDir" && start /B "" "exePath"
Write-Host "[1] 模拟VBS启动命令：cmd /c cd /d && start /B..."
cmd /c "cd /d `"$scriptDir`" && start /B `"`" `"$exePath`""

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

# 测试API端点
Write-Host "[3] 测试API端点..."
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
Write-Host "  -> health: $($r.StatusCode)"

$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
$tracks = ($r.Content | ConvertFrom-Json).data
Write-Host "  -> tracks: $($r.StatusCode), $($tracks.Count)条"

$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
$fields = ($r.Content | ConvertFrom-Json).data
Write-Host "  -> setup/fields: $($r.StatusCode), $($fields.Count)项"

# 打开浏览器（VBS脚本也会自动打开）
Write-Host "[4] 打开浏览器..."
Start-Process "http://127.0.0.1:8000"
Write-Host "  -> 浏览器已打开"

Start-Sleep -Seconds 3

# 清理
Write-Host "[5] 清理进程..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 已清理"

Write-Host ""
Write-Host "===== VBS启动方式验证: 成功 ====="