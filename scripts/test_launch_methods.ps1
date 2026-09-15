﻿﻿# 测试不同的exe启动方式
# 找出哪种方式能成功启动F1OPT.exe

Write-Host "===== F1OPT.exe 启动方式测试 ====="
Write-Host ""

# 清理
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$exePath = "D:\F1OPT-Test\dist\F1OPT.exe"
$workDir = "D:\F1OPT-Test\dist"

# 方式1: Start-Process + WorkingDirectory + WindowStyle Hidden
Write-Host "[方式1] Start-Process -WindowStyle Hidden..."
$proc = Start-Process -FilePath $exePath -WorkingDirectory $workDir -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 5
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) {
    Write-Host "  -> 成功！PID=$($proc.Id)"
    Stop-Process -Id $proc.Id -Force
    Start-Sleep -Seconds 2
} else {
    Write-Host "  -> 失败（进程已退出）"
}

# 方式2: Start-Process + WorkingDirectory（默认窗口）
Write-Host "[方式2] Start-Process（默认窗口）..."
$proc = Start-Process -FilePath $exePath -WorkingDirectory $workDir -PassThru
Start-Sleep -Seconds 5
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) {
    Write-Host "  -> 成功！PID=$($proc.Id)"
    Stop-Process -Id $proc.Id -Force
    Start-Sleep -Seconds 2
} else {
    Write-Host "  -> 失败（进程已退出）"
}

# 方式3: cmd /c start /B（在dist目录下）
Write-Host "[方式3] cmd /c start /B..."
Set-Location $workDir
cmd /c "start /B F1OPT.exe"
Start-Sleep -Seconds 5
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "  -> 成功！PID=$($f1opt.Id)"
    Stop-Process -Name "F1OPT" -Force
    Start-Sleep -Seconds 2
} else {
    Write-Host "  -> 失败（进程不存在）"
}

# 方式4: cmd /c（显式cd + exe路径）
Write-Host "[方式4] cmd /c cd /d && exe..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
cmd /c "cd /d D:\F1OPT-Test\dist && start /B F1OPT.exe"
Start-Sleep -Seconds 5
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "  -> 成功！PID=$($f1opt.Id)"
    Stop-Process -Name "F1OPT" -Force
    Start-Sleep -Seconds 2
} else {
    Write-Host "  -> 失败（进程不存在）"
}

# 方式5: 直接运行exe（不设置工作目录）
Write-Host "[方式5] 直接运行exe（无工作目录）..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Set-Location "D:\IDEProjects\demo"
$proc = Start-Process -FilePath $exePath -PassThru
Start-Sleep -Seconds 5
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) {
    Write-Host "  -> 成功！PID=$($proc.Id)"
    Stop-Process -Id $proc.Id -Force
    Start-Sleep -Seconds 2
} else {
    Write-Host "  -> 失败（进程已退出）"
}

Write-Host ""
Write-Host "===== 测试完成 ====="