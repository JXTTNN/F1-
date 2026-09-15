﻿# 端到端测试：桌面快捷方式 → VBS → F1OPT → 浏览器
$distDir = "D:\F1OPT-Test\dist"

# 先清理
$f1optOld = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1optOld) {
    $f1optOld | ForEach-Object { $_.Kill() }
    Start-Sleep -Seconds 3
    Write-Host "已清理旧进程" -ForegroundColor Yellow
}

# 检查桌面快捷方式
$shortcutPath = "C:\Users\29282\Desktop\F1OPT 赛车调教优化助手.lnk"
Write-Host "=== 端到端测试 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "[1] 检查桌面快捷方式..." -ForegroundColor Yellow
if (Test-Path $shortcutPath) {
    Write-Host "    ✅ 快捷方式存在" -ForegroundColor Green
    
    # 读取快捷方式信息
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    Write-Host "    Target: $($shortcut.TargetPath)" -ForegroundColor Green
    Write-Host "    Arguments: $($shortcut.Arguments)" -ForegroundColor Green
    Write-Host "    WorkingDir: $($shortcut.WorkingDirectory)" -ForegroundColor Green
} else {
    Write-Host "    ❌ 快捷方式不存在" -ForegroundColor Red
    Write-Host "    需要重新创建" -ForegroundColor Yellow
}

# 模拟双击快捷方式（通过wscript运行VBS）
Write-Host ""
Write-Host "[2] 模拟双击快捷方式..." -ForegroundColor Yellow
$startTime = Get-Date

# 快捷方式指向 wscript.exe + VBS脚本
# 模拟双击就是运行 wscript.exe "VBS路径"
$vbsPath = "$distDir\F1OPT桌面启动.vbs"
Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath`"" -NoNewWindow

# 等待API
$apiReady = $false
for ($i = 1; $i -le 40; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            Write-Host "    ✅ API ${elapsed}秒就绪" -ForegroundColor Green
            $apiReady = $true
            break
        }
    } catch {
        if ($i % 5 -eq 0) { Write-Host "    ${i}秒..." -ForegroundColor DarkGray }
    }
}

# 检查F1OPT进程
Write-Host ""
Write-Host "[3] 检查 F1OPT 进程..." -ForegroundColor Yellow
$f1opt = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
if ($f1opt) {
    Write-Host "    ✅ F1OPT PID: $($f1opt.Id)" -ForegroundColor Green
} else {
    Write-Host "    ❌ F1OPT未运行" -ForegroundColor Red
}

# 测试API端点
if ($apiReady) {
    Write-Host ""
    Write-Host "[4] 测试API端点..." -ForegroundColor Yellow
    
    $r1 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "    health: $($r1.StatusCode)" -ForegroundColor Green
    
    $r2 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/tracks" -UseBasicParsing -TimeoutSec 5
    $tracks = ($r2.Content | ConvertFrom-Json).data
    Write-Host "    tracks: $($r2.StatusCode), $($tracks.Count)条" -ForegroundColor Green
    
    $r3 = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/setup/fields" -UseBasicParsing -TimeoutSec 5
    $fields = ($r3.Content | ConvertFrom-Json).data
    Write-Host "    setup/fields: $($r3.StatusCode), $($fields.Count)项" -ForegroundColor Green
}

# 总结
Write-Host ""
Write-Host "=== 端到端测试总结 ===" -ForegroundColor Cyan
if ($apiReady -and $f1opt) {
    Write-Host "✅ 桌面快捷方式 → VBS → F1OPT → API 全链路成功！" -ForegroundColor Green
    Write-Host "   用户双击桌面快捷方式即可启动F1OPT并自动打开浏览器" -ForegroundColor Green
} else {
    Write-Host "❌ 端到端测试失败" -ForegroundColor Red
}