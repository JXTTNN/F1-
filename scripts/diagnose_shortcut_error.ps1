﻿# 诊断桌面快捷方式启动错误
# 1. 检查端口和进程状态
# 2. 尝试用cscript运行VBS脚本（命令行版本，能看到错误）
# 3. 如果VBS失败，直接用cmd启动exe

Write-Host "===== 诊断桌面快捷方式启动错误 ====="
Write-Host ""

# 1. 检查F1OPT进程
Write-Host "[1] 检查F1OPT进程..."
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "  -> F1OPT进程存在！PID=$($f1opt.Id)，内存=$([math]::Round($f1opt.WorkingSet64/1MB,1))MB"
    Write-Host "  -> 可能是之前的进程没关闭，导致端口冲突"
} else {
    Write-Host "  -> 无F1OPT进程"
}

# 2. 检查端口8000
Write-Host ""
Write-Host "[2] 检查端口8000..."
$portTest = netstat -ano | Select-String ":8000 " | Select-String "LISTENING"
if ($portTest) {
    Write-Host "  -> 端口8000被占用！"
    $portTest | ForEach-Object { Write-Host "  -> $_" }
} else {
    Write-Host "  -> 端口8000空闲"
}

# 3. 检查wscript进程
Write-Host ""
Write-Host "[3] 检查wscript进程..."
$wscript = Get-Process -Name "wscript" -ErrorAction SilentlyContinue
if ($wscript) {
    Write-Host "  -> wscript进程存在！PID=$($wscript.Id)"
    Write-Host "  -> 可能是之前的VBS脚本还在运行"
} else {
    Write-Host "  -> 无wscript进程"
}

# 4. 尝试用cscript运行VBS脚本（命令行版本，能看到错误输出）
Write-Host ""
Write-Host "[4] 尝试用cscript运行VBS脚本..."
Write-Host "  -> cscript是命令行版Windows脚本宿主，能看到错误信息"

# 先清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 用cscript运行VBS脚本（5秒超时，只看是否能启动）
Write-Host "  -> 运行: cscript //nologon F1OPT桌面启动.vbs"
$job = Start-Job -ScriptBlock {
    Set-Location "D:\F1OPT-Test\dist"
    cscript //nologon "D:\F1OPT-Test\dist\F1OPT桌面启动.vbs" 2>&1
}

# 等待5秒看cscript输出
Start-Sleep -Seconds 5

# 检查F1OPT进程是否启动了
$f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt2) {
    Write-Host "  -> F1OPT进程已启动！PID=$($f1opt2.Id)"
    
    # 测试API
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
        Write-Host "  -> API响应: $($r.StatusCode) OK"
    } catch {
        Write-Host "  -> API未响应: $($_.Exception.Message)"
    }
} else {
    Write-Host "  -> F1OPT进程未启动"
    
    # 获取cscript的输出
    $output = Receive-Job $job -ErrorAction SilentlyContinue
    if ($output) {
        Write-Host "  -> cscript输出: $output"
    }
}

# 清理job
Stop-Job $job -ErrorAction SilentlyContinue
Remove-Job $job -ErrorAction SilentlyContinue

# 5. 如果cscript方式也失败，尝试直接cmd启动
Write-Host ""
Write-Host "[5] 尝试直接cmd启动..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Set-Location "D:\F1OPT-Test\dist"
cmd /c "start /B F1OPT.exe"
Start-Sleep -Seconds 5

$f1opt3 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt3) {
    Write-Host "  -> cmd启动成功！PID=$($f1opt3.Id)"
    
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
        Write-Host "  -> API响应: $($r.StatusCode) OK"
    } catch {
        Write-Host "  -> API未响应: $($_.Exception.Message)"
    }
    
    Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "  -> cmd启动也失败"
}

# 6. 检查VBS脚本是否有语法问题
Write-Host ""
Write-Host "[6] 检查VBS脚本语法..."
$vbsContent = Get-Content "D:\F1OPT-Test\dist\F1OPT桌面启动.vbs" -Raw
Write-Host "  -> VBS脚本行数: $(($vbsContent -split "`n").Count)"
Write-Host "  -> 第38行（启动命令）:"
$lines = $vbsContent -split "`n"
Write-Host "     $($lines[37].Trim())"

Write-Host ""
Write-Host "===== 诊断完成 ====="