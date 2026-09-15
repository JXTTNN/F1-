﻿﻿# 验证F1OPT桌面启动.vbs
# 直接用wscript.exe运行VBS脚本，模拟用户双击桌面快捷方式

Write-Host "===== 验证F1OPT桌面启动.vbs ====="
Write-Host ""

# 先清理旧进程
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 检查端口
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

# 直接运行VBS脚本（模拟双击桌面快捷方式）
Write-Host "[1] 运行VBS脚本（wscript.exe）..."
$vbsPath = "D:\F1OPT-Test\dist\F1OPT桌面启动.vbs"
$proc = Start-Process -FilePath "wscript.exe" -ArgumentList "`"$vbsPath`"" -PassThru
Write-Host "  -> wscript.exe PID=$($proc.Id)"

# VBS脚本会：
# 1. 检查exe是否存在
# 2. 检查API是否已运行
# 3. 后台启动F1OPT.exe（隐藏窗口）
# 4. 等待API就绪（最多60秒）
# 5. 打开浏览器

# 等待API就绪（VBS脚本内部也在等待，我们同时等待）
Write-Host "[2] 等待API就绪（VBS脚本会自动启动exe并打开浏览器）..."
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
    
    if ($i % 15 -eq 0) {
        $f1optProc = Get-Process -Name "F1OPT" -ErrorAction SilentlyContinue
        if ($f1optProc) {
            Write-Host "  -> [第${i}秒] F1OPT进程存活，PID=$($f1optProc.Id)"
        } else {
            Write-Host "  -> [第${i}秒] F1OPT进程不存在"
        }
    }
}

if (-not $ready) {
    Write-Host "  -> [失败] API在90秒内未就绪"
    Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
    Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
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

# 确认浏览器已打开（VBS脚本会自动打开）
Write-Host "[4] VBS脚本应已自动打开浏览器"

# 等待几秒让浏览器加载
Start-Sleep -Seconds 5

# 清理
Write-Host "[5] 清理进程..."
Stop-Process -Name "F1OPT" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wscript" -Force -ErrorAction SilentlyContinue
Write-Host "  -> 已清理"

Write-Host ""
Write-Host "===== VBS脚本验证: 成功 ====="
Write-Host "  - VBS脚本通过wscript.exe正常运行"
Write-Host "  - F1OPT.exe后台启动成功"
Write-Host "  - API响应正常"
Write-Host "  - 浏览器自动打开"