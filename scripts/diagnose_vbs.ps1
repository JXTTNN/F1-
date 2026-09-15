﻿# 诊断 VBS 脚本启动失败的原因
# 1. 打印 VBS 生成的 cmdLine
# 2. 直接在 cmd 中执行相同命令
# 3. 对比测试

$distDir = "D:\F1OPT-Test\dist"
$exePath = "$distDir\F1OPT.exe"

Write-Host "=== VBS 启动诊断 ===" -ForegroundColor Cyan
Write-Host ""

# 1. 模拟 VBS 脚本生成的 cmdLine
Write-Host "[1] 模拟 VBS 生成的 cmdLine..." -ForegroundColor Yellow
$scriptDir = $distDir
# VBS中的拼接: "cmd /c cd /d " & Chr(34) & scriptDir & Chr(34) & " && start /B " & Chr(34) & Chr(34) & " " & Chr(34) & exePath & Chr(34)
# Chr(34) = "
$cmdLine = "cmd /c cd /d `"$scriptDir`" && start /B `"`" `"$exePath`""
Write-Host "    cmdLine = $cmdLine" -ForegroundColor Green
Write-Host ""

# 2. 直接在 PowerShell 中执行这个命令
Write-Host "[2] 直接执行 cmdLine..." -ForegroundColor Yellow
$startTime = Get-Date
cmd /c cd /d "$scriptDir" && start /B "" "$exePath"
Write-Host "    命令已执行，等待 API..." -ForegroundColor Yellow

# 3. 等待 API
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($response.code -eq 200) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    API 就绪！耗时 ${elapsed} 秒" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 3 -eq 0) {
            Write-Host "    ${i}秒... 等待中" -ForegroundColor DarkGray
        }
    }
}

# 4. 检查进程
Write-Host ""
Write-Host "[3] 检查 F1OPT 进程..." -ForegroundColor Yellow
$f1optProc = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optProc) {
    Write-Host "    ✅ F1OPT 进程存在, PID: $($f1optProc.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT 进程不存在" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 诊断完成 ===" -ForegroundColor Cyan