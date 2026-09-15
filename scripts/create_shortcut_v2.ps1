﻿# 重新创建桌面快捷方式
# 确保快捷方式真正创建在桌面上

Write-Host "===== 重新创建桌面快捷方式 ====="
Write-Host ""

$desktopPath = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktopPath "F1OPT 赛车调教优化助手.lnk"
$vbsPath = "D:\F1OPT-Test\dist\F1OPT桌面启动.vbs"
$exePath = "D:\F1OPT-Test\dist\F1OPT.exe"

Write-Host "[1] 桌面路径: $desktopPath"
Write-Host "[2] 快捷方式路径: $lnkPath"
Write-Host "[3] VBS脚本路径: $vbsPath"
Write-Host "[4] EXE路径: $exePath"
Write-Host ""

# 检查VBS和EXE是否存在
Write-Host "[5] 检查依赖文件..."
Write-Host "  VBS存在: $(Test-Path $vbsPath)"
Write-Host "  EXE存在: $(Test-Path $exePath)"
Write-Host ""

# 删除旧的快捷方式（如果存在）
if (Test-Path $lnkPath) {
    Remove-Item $lnkPath -Force
    Write-Host "[6] 已删除旧快捷方式"
} else {
    Write-Host "[6] 无旧快捷方式"
}

# 创建快捷方式
Write-Host "[7] 创建快捷方式..."
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath = "C:\Windows\system32\wscript.exe"
$lnk.Arguments = "`"$vbsPath`""
$lnk.WorkingDirectory = "D:\F1OPT-Test\dist"
$lnk.IconLocation = "$exePath,0"
$lnk.Description = "F1OPT 赛车调教优化助手 - 一键启动"
$lnk.Save()

# 验证快捷方式是否真的创建了
Write-Host ""
Write-Host "[8] 验证快捷方式..."
if (Test-Path $lnkPath) {
    $fileInfo = Get-Item $lnkPath
    Write-Host "  -> 快捷方式已创建！"
    Write-Host "  -> 路径: $($fileInfo.FullName)"
    Write-Host "  -> 大小: $($fileInfo.Length) 字节"
    Write-Host "  -> 时间: $($fileInfo.LastWriteTime)"
    
    # 验证快捷方式内容
    $lnk2 = $ws.CreateShortcut($lnkPath)
    Write-Host "  -> 目标: $($lnk2.TargetPath)"
    Write-Host "  -> 参数: $($lnk2.Arguments)"
    Write-Host "  -> 工作目录: $($lnk2.WorkingDirectory)"
    Write-Host "  -> 图标: $($lnk2.IconLocation)"
} else {
    Write-Host "  -> [失败] 快捷方式未创建！"
}

# 列出桌面上所有.lnk文件确认
Write-Host ""
Write-Host "[9] 桌面上所有快捷方式:"
Get-ChildItem $desktopPath -Filter "*.lnk" | ForEach-Object {
    Write-Host "  $($_.Name) ($($_.Length) bytes)"
}

Write-Host ""
Write-Host "===== 完成 ====="