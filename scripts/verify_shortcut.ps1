﻿﻿$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut("C:\Users\29282\Desktop\F1OPT 赛车调教优化助手.lnk")
Write-Host "快捷方式目标: $($lnk.TargetPath)"
Write-Host "快捷方式参数: $($lnk.Arguments)"
Write-Host "快捷方式工作目录: $($lnk.WorkingDirectory)"
Write-Host "快捷方式图标: $($lnk.IconLocation)"
Write-Host ""
Write-Host "VBS脚本存在: $(Test-Path 'D:\F1OPT-Test\dist\F1OPT桌面启动.vbs')"
Write-Host "F1OPT.exe存在: $(Test-Path 'D:\F1OPT-Test\dist\F1OPT.exe')"