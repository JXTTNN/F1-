﻿﻿# 验证桌面快捷方式配置
$lnkPath = "C:\Users\29282\Desktop\F1OPT 赛车调教优化助手.lnk"
$vbsPath = "D:\F1OPT-Test\dist\F1OPT桌面启动.vbs"
$result = [ordered]@{
    lnkExists = $false
    vbsExists = $false
    targetPath = $null
    targetArgs = $null
    targetCorrect = $false
    vbsContentOk = $false
    exePathInVbs = $null
    urlInVbs = $null
    success = $false
}

Write-Output "========== 桌面快捷方式验证 =========="

# 1. 检查快捷方式文件
if (Test-Path $lnkPath) {
    $result.lnkExists = $true
    Write-Output "[PASS] 快捷方式文件存在: $lnkPath"
} else {
    Write-Output "[FAIL] 快捷方式文件不存在: $lnkPath"
}

# 2. 检查VBS脚本
if (Test-Path $vbsPath) {
    $result.vbsExists = $true
    Write-Output "[PASS] VBS脚本存在: $vbsPath"
} else {
    Write-Output "[FAIL] VBS脚本不存在: $vbsPath"
}

# 3. 读取快捷方式目标
if ($result.lnkExists) {
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($lnkPath)
    $result.targetPath = $lnk.TargetPath
    $result.targetArgs = $lnk.Arguments
    Write-Output ("目标程序: {0}" -f $lnk.TargetPath)
    Write-Output ("目标参数: {0}" -f $lnk.Arguments)
    Write-Output ("工作目录: {0}" -f $lnk.WorkingDirectory)
    
    # 验证目标是wscript.exe
    $targetName = [System.IO.Path]::GetFileName($lnk.TargetPath)
    if ($targetName -ieq "wscript.exe") {
        Write-Output "[PASS] 目标程序是 wscript.exe"
    } else {
        Write-Output "[FAIL] 目标程序不是 wscript.exe, 实际: $targetName"
    }
    
    # 验证参数包含VBS路径
    if ($lnk.Arguments -like "*F1OPT桌面启动.vbs*" -or $lnk.Arguments -like "*$vbsPath*") {
        $result.targetCorrect = $true
        Write-Output "[PASS] 参数包含VBS脚本路径"
    } else {
        Write-Output ("[WARN] 参数未直接包含预期VBS路径, 参数: {0}" -f $lnk.Arguments)
        # 检查参数中的路径是否存在
        $argPath = $lnk.Arguments.Trim('"').Trim("'")
        if (Test-Path $argPath) {
            $result.targetCorrect = $true
            Write-Output "[PASS] 参数中的路径存在: $argPath"
        } else {
            Write-Output "[FAIL] 参数中的路径不存在: $argPath"
        }
    }
}

# 4. 验证VBS脚本内容
if ($result.vbsExists) {
    $vbsContent = Get-Content $vbsPath -Raw
    # 检查VBS中是否引用了F1OPT.exe
    if ($vbsContent -match "F1OPT\.exe") {
        $result.vbsContentOk = $true
        Write-Output "[PASS] VBS脚本引用了 F1OPT.exe"
    } else {
        Write-Output "[FAIL] VBS脚本未引用 F1OPT.exe"
    }
    # 检查VBS中的URL
    if ($vbsContent -match "http://127\.0\.0\.1:8000") {
        $result.urlInVbs = "http://127.0.0.1:8000"
        Write-Output "[PASS] VBS脚本使用端口8000"
    } else {
        Write-Output "[WARN] VBS脚本中未找到8000端口"
    }
    # 检查VBS中是否通过scriptDir构建路径
    if ($vbsContent -match "scriptDir") {
        Write-Output "[PASS] VBS脚本使用相对路径(scriptDir)定位exe"
    }
}

$result.success = ($result.lnkExists -and $result.vbsExists -and $result.targetCorrect -and $result.vbsContentOk)
Write-Output ""
Write-Output ("快捷方式验证通过: {0}" -f $result.success)
Write-Output "=================================="

$result | ConvertTo-Json -Depth 4 | Out-File "D:\F1OPT-Test\scripts\shortcut_verify_result.json" -Encoding utf8
Write-Output "结果已保存到 shortcut_verify_result.json"