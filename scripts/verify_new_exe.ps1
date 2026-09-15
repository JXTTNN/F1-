﻿﻿# 验证新打包的F1OPT.exe启动和API响应
# 作者: 验证测试工程师
# 日期: 2026-09-14

$ErrorActionPreference = "Continue"
$exePath = "D:\F1OPT-Test\dist\F1OPT.exe"
$baseUrl = "http://127.0.0.1:8000"
$result = [ordered]@{
    exePath = $exePath
    exeExists = $false
    exeSize = $null
    exeLastWrite = $null
    startSuccess = $false
    apiReadySeconds = $null
    apiReady = $false
    endpoints = @()
    processKilled = $false
    errorMessage = $null
}

# 1. 检查exe文件
if (Test-Path $exePath) {
    $result.exeExists = $true
    $fi = Get-Item $exePath
    $result.exeSize = $fi.Length
    $result.exeLastWrite = $fi.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    Write-Output ("[OK] exe存在: {0} ({1} bytes, {2})" -f $exePath, $fi.Length, $result.exeLastWrite)
} else {
    $result.errorMessage = "exe文件不存在: $exePath"
    Write-Output "[FAIL] $($result.errorMessage)"
    $result | ConvertTo-Json -Depth 5
    exit 1
}

# 2. 杀掉可能存在的旧进程
Write-Output "[INFO] 清理可能存在的旧F1OPT进程..."
Stop-Process -Name F1OPT -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 3. 启动exe
Write-Output "[INFO] 启动 F1OPT.exe ..."
$startTime = Get-Date
try {
    $proc = Start-Process -FilePath $exePath -PassThru -WindowStyle Minimized
    $result.startSuccess = $true
    Write-Output ("[OK] 进程已启动, PID={0}" -f $proc.Id)
} catch {
    $result.errorMessage = "启动exe失败: $_"
    Write-Output "[FAIL] $($result.errorMessage)"
    $result | ConvertTo-Json -Depth 5
    exit 1
}

# 4. 轮询健康检查端点（最多60秒）
Write-Output "[INFO] 轮询 /api/v1/health (最多60秒)..."
$healthReady = $false
$readySeconds = 0
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 1
    $readySeconds = $i
    try {
        $resp = Invoke-WebRequest -Uri "$baseUrl/api/v1/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $healthReady = $true
            break
        }
    } catch {
        # 继续等待
    }
}

$result.apiReadySeconds = $readySeconds
$result.apiReady = $healthReady
if ($healthReady) {
    Write-Output ("[OK] API就绪, 耗时 {0} 秒" -f $readySeconds)
} else {
    Write-Output "[FAIL] API在60秒内未就绪"
    $result.errorMessage = "API在60秒内未就绪"
    # 杀掉进程
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    $result.processKilled = $true
    $result | ConvertTo-Json -Depth 5
    exit 1
}

# 5. 测试各API端点
$endpoints = @(
    @{ name = "健康检查"; method = "GET"; path = "/api/v1/health"; expectType = "object"; expectKeys = @("status") },
    @{ name = "赛道列表"; method = "GET"; path = "/api/v1/tracks"; expectType = "array"; expectMinCount = 24 },
    @{ name = "调教参数定义"; method = "GET"; path = "/api/v1/setup/fields"; expectType = "array"; expectMinCount = 21 },
    @{ name = "症状列表"; method = "GET"; path = "/api/v1/symptoms"; expectType = "array"; expectMinCount = 15 }
)

foreach ($ep in $endpoints) {
    $epResult = [ordered]@{
        name = $ep.name
        path = $ep.path
        method = $ep.method
        statusCode = $null
        responseTimeMs = $null
        success = $false
        dataInfo = $null
        errorMessage = $null
    }
    Write-Output ("[INFO] 测试 {0} {1} ..." -f $ep.method, $ep.path)
    try {
        $t0 = Get-Date
        $resp = Invoke-WebRequest -Uri ($baseUrl + $ep.path) -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
        $t1 = Get-Date
        $epResult.statusCode = $resp.StatusCode
        $epResult.responseTimeMs = [int]($t1 - $t0).TotalMilliseconds

        # 解析JSON
        $json = $resp.Content | ConvertFrom-Json

        if ($ep.expectType -eq "array") {
            $count = @($json).Count
            $epResult.dataInfo = "array count=$count"
            if ($count -ge $ep.expectMinCount) {
                $epResult.success = $true
                Write-Output ("[OK] {0}: HTTP {1}, {2}ms, {3} (期望>={4})" -f $ep.name, $epResult.statusCode, $epResult.responseTimeMs, $epResult.dataInfo, $ep.expectMinCount)
            } else {
                $epResult.errorMessage = "数量不足: 实际=$count, 期望>=$($ep.expectMinCount)"
                Write-Output ("[FAIL] {0}: 数量不足 实际={1}, 期望>={2}" -f $ep.name, $count, $ep.expectMinCount)
            }
        } elseif ($ep.expectType -eq "object") {
            $keys = @()
            if ($json -is [System.Management.Automation.PSCustomObject]) {
                $keys = $json.PSObject.Properties.Name
            }
            $epResult.dataInfo = "object keys=$($keys -join ',')"
            $missingKeys = $ep.expectKeys | Where-Object { $_ -notin $keys }
            if ($missingKeys.Count -eq 0) {
                $epResult.success = $true
                Write-Output ("[OK] {0}: HTTP {1}, {2}ms, {3}" -f $ep.name, $epResult.statusCode, $epResult.responseTimeMs, $epResult.dataInfo)
            } else {
                $epResult.errorMessage = "缺少键: $($missingKeys -join ',')"
                Write-Output ("[FAIL] {0}: 缺少键 {1}" -f $ep.name, ($missingKeys -join ','))
            }
        }
    } catch {
        $epResult.errorMessage = "请求失败: $_"
        Write-Output ("[FAIL] {0}: {1}" -f $ep.name, $_)
    }
    $result.endpoints += [PSCustomObject]$epResult
}

# 6. 杀掉F1OPT进程
Write-Output "[INFO] 验证完成, 杀掉F1OPT进程..."
Stop-Process -Name F1OPT -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
$stillRunning = Get-Process -Name F1OPT -ErrorAction SilentlyContinue
if (-not $stillRunning) {
    $result.processKilled = $true
    Write-Output "[OK] F1OPT进程已终止"
} else {
    Write-Output "[WARN] F1OPT进程仍在运行"
}

# 7. 输出结果
$allSuccess = ($result.endpoints | Where-Object { -not $_.success }).Count -eq 0
$result.overallSuccess = $allSuccess
Write-Output ""
Write-Output "========== 验证结果摘要 =========="
Write-Output ("exe存在: {0}" -f $result.exeExists)
Write-Output ("exe大小: {0} bytes ({1:N2} MB)" -f $result.exeSize, ($result.exeSize / 1MB))
Write-Output ("exe修改时间: {0}" -f $result.exeLastWrite)
Write-Output ("启动成功: {0}" -f $result.startSuccess)
Write-Output ("API就绪耗时: {0} 秒" -f $result.apiReadySeconds)
Write-Output ("所有端点通过: {0}" -f $allSuccess)
foreach ($ep in $result.endpoints) {
    $status = if ($ep.success) { "PASS" } else { "FAIL" }
    Write-Output ("  [{0}] {1} ({2}ms)" -f $status, $ep.name, $ep.responseTimeMs)
}
Write-Output ("进程已清理: {0}" -f $result.processKilled)
Write-Output "==================================="

# 保存结果到JSON文件
$resultJson = $result | ConvertTo-Json -Depth 5
$resultFile = "D:\F1OPT-Test\scripts\verify_new_exe_result.json"
$resultJson | Out-File -FilePath $resultFile -Encoding utf8
Write-Output ("结果已保存到: {0}" -f $resultFile)

if ($allSuccess) {
    Write-Output "[RESULT] 验证通过"
    exit 0
} else {
    Write-Output "[RESULT] 验证失败"
    exit 2
}