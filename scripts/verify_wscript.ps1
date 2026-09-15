﻿# 用wscript运行VBS脚本（用户双击快捷方式的实际方式）
# 通过cmd启动wscript，绕过安全策略

Write-Host "===== 用wscript运行VBS脚本 ====="
Write-Host ""

# 先清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 通过cmd启动wscript运行VBS脚本
Write-Host "[1] 通过cmd启动wscript..."
cmd /c "start wscript.exe D:\F1OPT-Test\dist\F1OPT桌面启动.vbs"

# wscript会异步运行，我们需要等待API就绪
Write-Host "[2] 等待API就绪（最多90秒）..."
$ready = $false
for ($i = 1; $i -le 90; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            $ready = $true
            Write-Host "  -> API就绪！耗时 ${i} 秒"
            break
        }
    } catch {}
    
    if ($i % 10 -eq 0) {
        $f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
        $wsh = Get-Process -Name "wscript" -ErrorAction SilentlyContinue
        Write-Host "  -> [第${i}秒] F1OPT=$($f1opt.Id) wscript=$($wsh.Id)"
    }
}

if ($ready) {
    Write-Host ""
    Write-Host "[3] 测试API端点..."
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "  -> health: $($r.StatusCode)"
    
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
    $tracks = ($r.Content | ConvertFrom-Json).data
    Write-Host "  -> tracks: $($tracks.Count)条"
    
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
    $fields = ($r.Content | ConvertFrom-Json).data
    Write-Host "  -> setup/fields: $($fields.Count)项"
    
    Write-Host ""
    Write-Host "[4] VBS脚本应已自动打开浏览器"
} else {
    Write-Host "  -> [失败] API在90秒内未就绪"
    
    # 检查进程状态
    $f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
    $wsh = Get-Process -Name "wscript" -ErrorAction SilentlyContinue
    Write-Host "  -> F1OPT进程: $(if($f1opt){'存在 PID='+$f1opt.Id}else{'不存在'})"
    Write-Host "  -> wscript进程: $(if($wsh){'存在 PID='+$wsh.Id}else{'不存在'})"
}

# 清理
Start-Sleep -Seconds 3
Write-Host ""
Write-Host "[5] 清理..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 已清理"

Write-Host ""
Write-Host "===== 验证完成 ====="