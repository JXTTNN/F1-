# F1OPT 便携包构建脚本 (Windows Embeddable Python)
# 零 PyInstaller：直接跑官方 Python，杜绝自解压/冻结导致的闪退。
#
# 本地手动构建:  powershell -ExecutionPolicy Bypass -File exe/portable-build.ps1
# CI 构建:       .github/workflows/release-portable.yml (推 p* tag 触发)

param(
    [string]$Version = "1.4.1",
    # 与项目 requires-python (>=3.11) 及 CI setup-python 保持一致
    [string]$PyVersion = "3.11.9"
)

$ErrorActionPreference = "Stop"
$PyShort = $PyVersion.Substring(0, $PyVersion.LastIndexOf('.'))  # 3.11

Write-Host "=== 构建 f1opt-portable v$Version (Embeddable Python $PyVersion) ==="

# 1. 下载官方 Embeddable Python (免安装)
$embedZip  = "python-$PyVersion-embed-amd64.zip"
$embedUrl  = "https://www.python.org/ftp/python/$PyVersion/$embedZip"
if (-not (Test-Path $embedZip)) {
    Write-Host "[1/7] 下载 Embeddable Python ..."
    Invoke-WebRequest -Uri $embedUrl -OutFile $embedZip -UseBasicParsing
}

# 2. 解压到便携目录
$PORTABLE = "f1opt-portable"
if (Test-Path $PORTABLE) { Remove-Item $PORTABLE -Recurse -Force }
Expand-Archive -Path $embedZip -DestinationPath $PORTABLE -Force
Push-Location $PORTABLE

# 3. 启用 site-packages：把 ._pth 里被注释的 #import site 取消注释
$pthFile = "python$($PyShort -replace '\.','')._pth"   # python311._pth
(Get-Content $pthFile) | ForEach-Object {
    if ($_ -match '^#import site') { 'import site' } else { $_ }
} | Set-Content $pthFile

# 4. 引导 pip（Embeddable 版不含 pip，需 get-pip.py）
Write-Host "[4/7] 引导 pip ..."
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py" -UseBasicParsing
& ".\python.exe" get-pip.py --quiet
if ($LASTEXITCODE -ne 0) { throw "get-pip.py 执行失败" }

# 5. 安装 f1opt 及全部依赖 (torch 等) 到本便携包的 site-packages
Write-Host "[5/7] 安装 f1opt + 依赖 (torch 体积较大，请稍候) ..."
$RepoRoot = (Resolve-Path (Join-Path (Get-Location) "..\..")).Path
& ".\python.exe" -m pip install "$RepoRoot\." --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install 失败" }

# 6. 冒烟测试：确认能 --help 不闪退
Write-Host "[6/7] 冒烟测试 python -m f1opt.cli --help ..."
& ".\python.exe" -m f1opt.cli --help
if ($LASTEXITCODE -ne 0) { throw "便携包 --help 测试失败" }

# 7. 生成双击启动器 + 说明，清理临时文件
$bat = @"
@echo off
cd /d "%~dp0"
python.exe -m f1opt.cli %*
"@
$bat | Out-File -FilePath "f1opt.bat" -Encoding ascii

Remove-Item "get-pip.py" -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Pop-Location

# 打包 + 校验
$ZIP = "f1opt-portable-windows.zip"
if (Test-Path $ZIP) { Remove-Item $ZIP -Force }
Compress-Archive -Path $PORTABLE -DestinationPath $ZIP -Force
$sha = (Get-FileHash $ZIP -Algorithm SHA256).Hash
"$sha  $ZIP" | Out-File -Encoding ascii "$ZIP.sha256"

$sizeMB = [math]::Round((Get-Item $ZIP).Length / 1MB, 1)
Write-Host "=== 构建完成 ==="
Write-Host "产物: $ZIP ($sizeMB MiB)"
Write-Host "SHA256: $sha"