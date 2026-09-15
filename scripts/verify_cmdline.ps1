﻿# 验证VBS脚本中的cmdLine拼接是否正确
# 并用更长超时重新测试

Write-Host "===== 验证VBS cmdLine拼接 ====="
Write-Host ""

$scriptDir = "D:\F1OPT-Test\dist"
$exePath = "$scriptDir\F1OPT.exe"

# 模拟VBS中的cmdLine拼接
# VBS: cmdLine = "cmd /c cd /d " & Chr(34) & scriptDir & Chr(34) & " && start /B " & Chr(34) & Chr(34) & " " & Chr(34) & exePath & Chr(34)
$cmdLine = "cmd /c cd /d `"$scriptDir`" && start /B `"`" `"$exePath`""
Write-Host "[1] cmdLine = $cmdLine"
Write-Host ""

# 先清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 直接用这个cmdLine启动
Write-Host "[2] 执行cmdLine..."
cmd /c "cd /d `"$scriptDir`" && start /B `"`" `"$exePath`""

# 等待API就绪
Write-Host "[3] 等待API就绪..."
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

if ($ready) {
    Write-Host ""
    Write-Host "[4] 测试API端点..."
    $r = Invoke-WebRequest -Uri "http://127.0@0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "  -> health: $($r.StatusCode)"
    
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
    $tracks = ($r.Content | ConvertFrom-Json).data
    Write-Host "  -> tracks: $($tracks.Count)条"
    
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
    $fields = ($r.Content | ConvertFrom-Json).data
    Write-Host "  -> setup/5fields: $($fields.Count)项"
    
    Write-Host ""
    Write-Host "[5] 打开浏览器..."
    Start-Process "http://127.0.0.1:8000"
    Write-Host "  -> 浏览器已打开"
    
    Start-Sleep -Seconds 3
} else {
    Write-Host "  -> API未就绪"
}

# 清理
Write-Host ""
Write-Host "[6] 清理..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 已清理"

Write-Host ""
Write-Host "===== 验证完成 ====="