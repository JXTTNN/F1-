﻿# 重新打包 F1OPT-portable.zip（包含修复后的VBS脚本）
$distDir = "D:\F1OPT-Test\dist"
$zipPath = "$distDir\F1OPT-portable.zip"

Write-Host "=== 重新打包便携包 ===" -ForegroundColor Cyan

# 删除旧zip
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
    Write-Host "已删除旧zip" -ForegroundColor Yellow
}

# 打包
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)

# 添加 F1OPT.exe
$entry1 = $zip.CreateEntry("F1OPT.exe")
$stream1 = $entry1.Open()
$fileBytes1 = [System.IO.File]::ReadAllBytes("$distDir\F1OPT.exe")
$stream1.Write($fileBytes1, 0, $fileBytes1.Length)
$stream1.Close()
Write-Host "  + F1OPT.exe ($([math]::Round($fileBytes1.Length / 1MB, 2)) MB)" -ForegroundColor Green

# 添加 一键启动.bat
$entry2 = $zip.CreateEntry("一键启动.bat")
$stream2 = $entry2.Open()
$fileBytes2 = [System.IO.File]::ReadAllBytes("$distDir\一键启动.bat")
$stream2.Write($fileBytes2, 0, $fileBytes2.Length)
$stream2.Close()
Write-Host "  + 一键启动.bat ($([math]::Round($fileBytes2.Length / 1KB, 2)) KB)" -ForegroundColor Green

# 添加 F1OPT桌面启动.vbs
$entry3 = $zip.CreateEntry("F1OPT桌面启动.vbs")
$stream3 = $entry3.Open()
$fileBytes3 = [System.IO.File]::ReadAllBytes("$distDir\F1OPT桌面启动.vbs")
$stream3.Write($fileBytes3, 0, $fileBytes3.Length)
$stream3.Close()
Write-Host "  + F1OPT桌面启动.vbs ($([math]::Round($fileBytes3.Length / 1KB, 2)) KB)" -ForegroundColor Green

$zip.Dispose()

# 验证zip
$zipSize = (Get-Item $zipPath).Length / 1MB
Write-Host ""
Write-Host "便携包打包完成：$zipPath" -ForegroundColor Green
Write-Host "大小：$([math]::Round($zipSize, 2)) MB" -ForegroundColor Green

# 验证zip内容
$zipVerify = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
Write-Host ""
Write-Host "zip内容验证：" -ForegroundColor Yellow
foreach ($entry in $zipVerify.Entries) {
    Write-Host "  $($entry.Name) - $([math]::Round($entry.Length / 1KB, 2)) KB" -ForegroundColor Green
}
$zipVerify.Dispose()

Write-Host ""
Write-Host "=== 打包完成 ===" -ForegroundColor Cyan