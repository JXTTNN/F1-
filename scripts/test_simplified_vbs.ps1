﻿# 测试简化后的VBS命令
$distDir = "D:\F1OPT-Test\dist"

Write-Host "=== 简化VBS命令测试 ===" -ForegroundColor Cyan

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 测试1：用 cmd /c "start /B F1OPT.exe"（验证过的成功方式）
Write-Host ""
Write-Host "[1] 验证过的成功方式..." -ForegroundColor Yellow
Set-Location $distDir
$startTime = Get-Date
cmd /c "start /B F1OPT.exe"

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {}
}

$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
Write-Host ""
Write-Host "[2] 清理进程..." -ForegroundColor Yellow
$f1opt | ForEach-Object { $_.Kill() }
Start-Sleep -Seconds 3

# 测试2：用 Start-Process cmd.exe -ArgumentList '/c start /B F1OPT.exe' -WorkingDirectory $distDir
Write-Host "[3] Start-Process方式..." -ForegroundColor Yellow
$startTime2 = Get-Date
$proc = Start-Process -FilePath "cmd.exe" -ArgumentList '/c', 'start /B F1OPT.exe' -WorkingDirectory $distDir -PassThru -WindowStyle Normal
Write-Host "    cmd PID: $($proc.Id)" -ForegroundColor Green

for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 3 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt2 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt2) {
    Write-Host "    ✅ F1OPT PID: $($f1opt2.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 清理
Write-Host ""
Write-Host "[4] 清理进程..." -ForegroundColor Yellow
$f1opt2 | ForEach-Object { $_.Kill() }
Start-Sleep -Seconds 3

# 测试3：用 wscript 运行简化VBS
Write-Host "[5] 测试简化VBS脚本..." -ForegroundColor Yellow

# 创建简化VBS
$simpleVbs = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "$distDir"
WshShell.Run "cmd /c start /B F1OPT.exe", 1, False
WScript.Sleep 5000
WshShell.Run "http://127.0.0.1:8000"
"@
$simpleVbsPath = "$distDir\test_simple.vbs"
$simpleVbs | Out-File -FilePath $simpleVbsPath -Encoding ASCII

$startTime3 = Get-Date
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$simpleVbsPath`"" -NoNewWindow

for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.code -eq 200) {
            Write-Host "    ✅ VBS启动成功！API ${i}秒就绪" -ForegroundColor Green
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

$f1opt3 = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt3) {
    Write-Host "    ✅ F1OPT PID: $($f1opt3.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ VBS启动后F1OPT未运行" -ForegroundColor Red
}

# 清理测试VBS
Remove-Item $simpleVbsPath -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== 测试完成 ===" -ForegroundColor Cyan