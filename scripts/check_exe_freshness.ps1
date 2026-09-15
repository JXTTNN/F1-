﻿﻿$exeTime = (Get-Item 'D:\F1OPT-Test\dist\F1OPT.exe').LastWriteTime
Write-Host "EXE打包时间: $exeTime"
Write-Host ""
Write-Host "exe打包后修改的源文件:"
Get-ChildItem 'D:\F1OPT-Test\setup_tuner' -Recurse -File -Filter '*.py' | Where-Object { $_.LastWriteTime -gt $exeTime } | ForEach-Object {
    Write-Host "  $($_.FullName.Replace('D:\F1OPT-Test\', ''))  -  $($_.LastWriteTime)"
}
Write-Host ""
Write-Host "exe打包后修改的UI文件:"
Get-ChildItem 'D:\F1OPT-Test\setup_tuner\ui' -Recurse -File | Where-Object { $_.LastWriteTime -gt $exeTime } | ForEach-Object {
    Write-Host "  $($_.FullName.Replace('D:\F1OPT-Test\', ''))  -  $($_.LastWriteTime)"
}