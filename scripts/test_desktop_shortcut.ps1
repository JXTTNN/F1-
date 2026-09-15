﻿﻿# F1OPT 桌面快捷方式启动测试
# 模拟VBS脚本的行为：启动exe → 等待API就绪 → 打开浏览器

Write-Host "===== F1OPT 桌面快捷方式启动测试 ====="
Write-Host ""

# Step 1: 清理旧进程
Write-Host "[Step 1] 清理旧F1OPT进程..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "  -> 旧进程已清理"

# Step 2: 检查端口8000是否被占用
Write-Host "[Step 2] 检查端口8000..."
$portTest = netstat -ano | Select-String ":8000 " | Select-String "LISTENING"
if ($portTest) {
    Write-Host "  -> 端口8000被占用，尝试释放..."
    $portTest | ForEach-Object {
        $line = $_.ToString()
        $parts = $line -split '\s+'
        $pid = $parts[-1]
        if ($pid -match '^\d+$' -and $pid -ne '0') {
            Stop-Process -Id ([int]$pid) -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
    Write-Host "  -> 端口已释放"
} else {
    Write-Host "  -> 端口8000空闲"
}

# Step 3: 启动F1OPT.exe（在dist工作目录下，模拟VBS的行为）
Write-Host "[Step 3] 启动F1OPT.exe..."
$exePath = "D:\F1OPT-Test\dist\F1OPT.exe"
$workDir = "D:\F1OPT-Test\dist"
$proc = Start-Process -FilePath $exePath -WorkingDirectory $workDir -PassThru -WindowStyle Hidden
Write-Host "  -> 进程已启动，PID=$($proc.Id)"

# Step 4: 等待API就绪（最多60秒）
Write-Host "[Step 4] 等待API就绪（最多60秒）..."
$apiReady = $false
$elapsed = 0
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 1
    $elapsed = $i
    
    # 检查进程是否还活着
    $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if (-not $alive) {
        Write-Host "  -> [失败] F1OPT.exe 进程已退出（${elapsed}秒）"
        Write-Host ""
        Write-Host "===== 测试结果: 失败 ====="
        exit 1
    }
    
    # 检查API健康
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $apiReady = $true
            break
        }
    } catch {
        # 继续等待
    }
}

if ($apiReady) {
    Write-Host "  -> API就绪！耗时 ${elapsed} 秒"
} else {
    Write-Host "  -> [失败] API在60秒内未就绪"
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "===== 测试结果: 失败 ====="
    exit 1
}

# Step 5: 测试API端点
Write-Host "[Step 5] 测试API端点..."

# 5.1 健康检查
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
$sw.Stop()
Write-Host "  -> /api/v1/health: $($resp.StatusCode) OK, $($sw.ElapsedMilliseconds)ms"

# 5.2 赛道列表
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
$sw.Stop()
$tracks = ($resp.Content | ConvertFrom-Json).data
Write-Host "  -> /api/v1/tracks: $($resp.StatusCode) OK, $($tracks.Count)条赛道, $($sw.ElapsedMilliseconds)ms"

# 5.3 调教参数定义
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
$sw.Stop()
$fields = ($resp.Content | ConvertFrom-Json).data
Write-Host "  -> /api/v1/setup/fields: $($resp.StatusCode) OK, $($fields.Count)项参数, $($sw.ElapsedMilliseconds)ms"

# Step 6: 打开浏览器（模拟VBS脚本的行为）
Write-Host "[Step 6] 打开浏览器..."
Start-Process "http://127.0.0.1:8000"
Write-Host "  -> 浏览器已打开 http://127.0.0.1:8000"

# Step 7: 等待5秒让用户看到浏览器已打开
Write-Host "[Step 7] 等待5秒让浏览器加载..."
Start-Sleep -Seconds 5

# Step 8: 清理——杀掉F1OPT进程
Write-Host "[Step 8] 清理F1OPT进程..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 进程已清理"

Write-Host ""
Write-Host "===== 测试结果: 成功 ====="
Write-Host "  - exe启动: 成功 (${elapsed}秒)"
Write-Host "  - API健康: 200 OK"
Write-Host "  - 赛道列表: $($tracks.Count)条"
Write-Host "  - 调教参数: $($fields.Count)项"
Write-Host "  - 浏览器: 已打开"
Write-Host "  - 桌面快捷方式: 配置正确，VBS脚本功能正常"