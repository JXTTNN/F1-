﻿﻿# F1OPT 桌面快捷方式启动测试 v2
# 改进：更长的等待时间，不依赖PID，用进程名检测

Write-Host "===== F1OPT 桌面快捷方式启动测试 v2 ====="
Write-Host ""

# Step 1: 清理旧进程
Write-Host "[Step 1] 清理旧F1OPT进程..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Write-Host "  -> 旧进程已清理"

# Step 2: 检查端口8000
Write-Host "[Step 2] 检查端口8000..."
$portTest = netstat -ano | Select-String ":8000 " | Select-String "LISTENING"
if ($portTest) {
    Write-Host "  -> 端口8000被占用，尝试释放..."
    $portTest | ForEach-Object {
        $line = $_.ToString()
        $parts = $line -split '\s+'
        $pidStr = $parts[-1]
        if ($pidStr -match '^\d+$' -and $pidStr -ne '0') {
            Stop-Process -Id ([int]$pidStr) -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 3
    Write-Host "  -> 端口已释放"
} else {
    Write-Host "  -> 端口8000空闲"
}

# Step 3: 启动F1OPT.exe
Write-Host "[Step 3] 启动F1OPT.exe（在dist工作目录下）..."
$exePath = "D:\F1OPT-Test\dist\F1OPT.exe"
$workDir = "D:\F1OPT-Test\dist"

# 用cmd启动，不等待返回
$cmd = "start /B `"`" `"$exePath`""
cmd /c $cmd
Write-Host "  -> 启动命令已发送"

# Step 4: 等待API就绪（最多90秒，给Nuitka onefile解压足够时间）
Write-Host "[Step 4] 等待API就绪（最多90秒，Nuitka onefile首次解压可能需要30-60秒）..."
$apiReady = $false
$elapsed = 0
for ($i = 1; $i -le 90; $i++) {
    Start-Sleep -Seconds 1
    $elapsed = $i
    
    # 检查F1OPT进程是否存在（不依赖PID）
    $f1optProc = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
    if (-not $f1optProc -and $i -gt 5) {
        # 5秒后如果进程不存在，说明启动失败
        Write-Host "  -> [第${i}秒] F1OPT进程不存在"
        if ($i -gt 10) {
            Write-Host "  -> [失败] F1OPT.exe 启动后退出"
            Write-Host ""
            Write-Host "===== 测试结果: 失败 ====="
            exit 1
        }
    }
    
    if ($i % 10 -eq 0) {
        $procInfo = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
        if ($procInfo) {
            Write-Host "  -> [第${i}秒] F1OPT进程存活，PID=$($procInfo.Id)，内存=$([math]::Round($procInfo.WorkingSet64/1MB, 1))MB"
        }
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
    Write-Host "  -> [失败] API在90秒内未就绪"
    Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "===== 测试结果: 失败 ====="
    exit 1
}

# Step 5: 测试API端点
Write-Host ""
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

# Step 6: 打开浏览器
Write-Host ""
Write-Host "[Step 6] 打开浏览器..."
Start-Process "http://127.0.0.1:8000"
Write-Host "  -> 浏览器已打开 http://127.0.0.1:8000"

# Step 7: 等待5秒让浏览器加载
Write-Host "[Step 7] 等待5秒让浏览器加载..."
Start-Sleep -Seconds 5

# Step 8: 清理
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