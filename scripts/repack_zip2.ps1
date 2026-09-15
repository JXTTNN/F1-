﻿# 用 Compress-Archive 重新打包
$distDir = "D:\F1OPT-Test\dist"
$zipPath = "$distDir\F1OPT-portable.zip"

Write-Host "=== 重新打包便携包 ===" -ForegroundColor Cyan

# 删除旧zip
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
    Write-Host "已删除旧zip" -ForegroundColor Yellow
}

# 用 Compress-Archive 打包
$files = @(
    "$distDir\F1OPT.exe",
    "$distDir\一键启动.bat",
    "$distDir\F1OPT桌面启动.vbs"
)

Compress-Archive -Path $files -DestinationPath $zipPath -CompressionLevel Optimal

# 验证
if (Test-Path $zipPath) {
    $zipSize = (Get-Item $zipPath).Length / 1MB
    Write-Host "便携包打包完成：$zipPath" -ForegroundColor Green
    Write-Host "大小：$([math]::Round($zipSize, 2)) MB" -ForegroundColor Green
    
    # 验证zip内容
    $zipInfo = Get-Item $zipPath
    Write-Host ""
    Write-Host "zip内容验证：" -ForegroundColor Yellow
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    foreach ($entry in $archive.Entries) {
        Write-Host "  $($entry.Name) - $([math]::Round($entry.Length / 1KB, 2)) KB" -ForegroundColor Green
    }
    $archive.Dispose()
} else {
    Write-Host "❌ 打包失败" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 打包完成 ===" -ForegroundColor Cyan