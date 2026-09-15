﻿﻿# 测试F1OPT API各端点（假设exe已启动并监听8000）
$baseUrl = "http://127.0.0.1:8000"
$results = @()

$endpoints = @(
    @{ name = "健康检查"; path = "/api/v1/health"; expectType = "object"; expectKeys = @("status") },
    @{ name = "赛道列表"; path = "/api/v1/tracks"; expectType = "array"; expectMinCount = 24 },
    @{ name = "调教参数定义"; path = "/api/v1/setup/fields"; expectType = "array"; expectMinCount = 21 },
    @{ name = "症状列表"; path = "/api/v1/symptoms"; expectType = "array"; expectMinCount = 15 }
)

Write-Output "========== API端点测试 =========="
foreach ($ep in $endpoints) {
    $r = [ordered]@{
        name = $ep.name
        path = $ep.path
        statusCode = $null
        responseTimeMs = $null
        success = $false
        dataInfo = $null
        errorMessage = $null
    }
    try {
        $t0 = Get-Date
        $resp = Invoke-WebRequest -Uri ($baseUrl + $ep.path) -TimeoutSec 15 -UseBasicParsing -ErrorAction Stop
        $t1 = Get-Date
        $r.statusCode = $resp.StatusCode
        $r.responseTimeMs = [int]($t1 - $t0).TotalMilliseconds
        $json = $resp.Content | ConvertFrom-Json

        if ($ep.expectType -eq "array") {
            $count = @($json).Count
            $r.dataInfo = "array count=$count"
            if ($count -ge $ep.expectMinCount) {
                $r.success = $true
                Write-Output ("[PASS] {0}: HTTP {1}, {2}ms, {3} (期望>={4})" -f $ep.name, $r.statusCode, $r.responseTimeMs, $r.dataInfo, $ep.expectMinCount)
            } else {
                $r.errorMessage = "数量不足: 实际=$count, 期望>=$($ep.expectMinCount)"
                Write-Output ("[FAIL] {0}: 数量不足 实际={1}, 期望>={2}" -f $ep.name, $count, $ep.expectMinCount)
            }
        } elseif ($ep.expectType -eq "object") {
            $keys = @()
            if ($json -is [System.Management.Automation.PSCustomObject]) {
                $keys = $json.PSObject.Properties.Name
            }
            $r.dataInfo = "object keys=$($keys -join ',')"
            $missingKeys = $ep.expectKeys | Where-Object { $_ -notin $keys }
            if ($missingKeys.Count -eq 0) {
                $r.success = $true
                Write-Output ("[PASS] {0}: HTTP {1}, {2}ms, {3}" -f $ep.name, $r.statusCode, $r.responseTimeMs, $r.dataInfo)
            } else {
                $r.errorMessage = "缺少键: $($missingKeys -join ',')"
                Write-Output ("[FAIL] {0}: 缺少键 {1}" -f $ep.name, ($missingKeys -join ','))
            }
        }
    } catch {
        $r.errorMessage = "请求失败: $_"
        Write-Output ("[FAIL] {0}: {1}" -f $ep.name, $_)
    }
    $results += [PSCustomObject]$r
}

$allPass = ($results | Where-Object { -not $_.success }).Count -eq 0
Write-Output ""
Write-Output ("所有端点通过: {0}" -f $allPass)
Write-Output "=================================="

# 保存结果
$results | ConvertTo-Json -Depth 4 | Out-File "D:\F1OPT-Test\scripts\api_test_result.json" -Encoding utf8
Write-Output "结果已保存到 api_test_result.json"

if ($allPass) { exit 0 } else { exit 2 }