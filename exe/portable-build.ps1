# F1OPT 便携包构建脚本 (Windows Embeddable Python)
# 零 PyInstaller：直接跑官方 Python，杜绝自解压/冻结导致的闪退。
#
# 本地手动构建:  powershell -ExecutionPolicy Bypass -File exe/portable-build.ps1
# CI 构建:       .github/workflows/release-portable.yml (推 p* tag 触发)
#
# 产物: <OutDir>/f1opt-portable-windows.zip (+ .sha256)

param(
    # 从 git tag (p1.4.2) 或手动传入, 仅写入 README 用
    [string]$Version = "1.4.2",
    # 与项目 requires-python (>=3.11) 及 CI setup-python 保持一致
    [string]$PyVersion = "3.11.9",
    # 产物输出目录 (CI 传仓库根, 本地默认当前目录)
    [string]$OutDir = "."
)

$ErrorActionPreference = "Stop"
$PyShort = $PyVersion.Substring(0, $PyVersion.LastIndexOf('.'))          # 3.11
$PyNoDot = $PyShort -replace '\.', ''                                    # 311

Write-Host "=== 构建 f1opt-portable v$Version (Embeddable Python $PyVersion) ==="

# 仓库根 = 本脚本所在目录 (exe/) 的上一级。这样无论从哪里调用都能定位 pyproject.toml。
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "仓库根: $RepoRoot"

# 1. 下载官方 Embeddable Python (免安装)
$embedZip = Join-Path $env:RUNNER_TEMP "python-$PyVersion-embed-amd64.zip"
if ($env:RUNNER_TEMP -eq $null) { $embedZip = "python-$PyVersion-embed-amd64.zip" }
$embedUrl = "https://www.python.org/ftp/python/$PyVersion/$embedZip"
if (-not (Test-Path $embedZip)) {
    Write-Host "[1/7] 下载 Embeddable Python ..."
    Invoke-WebRequest -Uri $embedUrl -OutFile $embedZip -UseBasicParsing
}

# 2. 解压到便携目录
$PORTABLE = Join-Path $env:RUNNER_TEMP "f1opt-portable"
if ($env:RUNNER_TEMP -eq $null) { $PORTABLE = "f1opt-portable" }
if (Test-Path $PORTABLE) { Remove-Item $PORTABLE -Recurse -Force }
Expand-Archive -Path $embedZip -DestinationPath $PORTABLE -Force
Push-Location $PORTABLE

# 3. 启用 site-packages：把 ._pth 里被注释的 #import site 取消注释
$pthFile = "python$PyNoDot._pth"   # python311._pth
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
# 用 "可安装的项目目录" 形式，pip 会在该目录解析 pyproject.toml
& ".\python.exe" -m pip install "$RepoRoot" --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install 失败" }

# 6. 冒烟测试：确认能 --help 不闪退
Write-Host "[6/7] 冒烟测试 python -m f1opt.cli --help ..."
& ".\python.exe" -m f1opt.cli --help
if ($LASTEXITCODE -ne 0) { throw "便携包 --help 测试失败" }

# 7. 生成双击启动器 + 说明，清理临时文件
@"
@echo off
cd /d "%~dp0"
python.exe -m f1opt.cli serve
pause
"@ | Out-File -FilePath "启动-F1OPT.bat" -Encoding ascii

@"
@echo off
cd /d "%~dp0"
python.exe -m f1opt.cli %*
"@ | Out-File -FilePath "f1opt.bat" -Encoding ascii

@"
F1OPT Portable v$Version
========================

运行方法（二选一）：
  - 双击 `启动-F1OPT.bat`  ->  启动实时面板 (serve)，浏览器自动打开
  - 双击 `f1opt.bat` + 命令行参数  ->  高级用法，如
        f1opt.bat search --track suzuka --iterations 100

说明：
  - 基于官方 Embeddable Python $PyVersion，零 PyInstaller 冻结，
    彻底避开 onefile/onedir 的「解压失败 => 闪退」问题。
  - 首次 `serve` 加载 torch 模型约 3-10 秒。
  - 建议解压到 SSD 盘，加载更快。

可选 LLM 配置（写进同目录 .env 或系统环境变量）：
  F1OPT_LLM_BACKEND=builtin   (默认，内置小模型，离线可用)
  F1OPT_LLM_BACKEND=local     (本地 Ollama，需先 ollama pull llama3.1)
  F1OPT_LLM_BACKEND=openai    (OpenAI 兼容云端)
  F1OPT_LLM_API_KEY=sk-...
  F1OPT_LLM_MODEL=gpt-4o-mini
"@ | Out-File -FilePath "README.txt" -Encoding UTF8

Remove-Item "get-pip.py" -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Pop-Location

# 打包 + 校验 (输出到 OutDir)
$OutDirFull = (Resolve-Path $OutDir).Path
$ZIP = Join-Path $OutDirFull "f1opt-portable-windows.zip"
if (Test-Path $ZIP) { Remove-Item $ZIP -Force }
Compress-Archive -Path $PORTABLE -DestinationPath $ZIP -Force
$sha = (Get-FileHash $ZIP -Algorithm SHA256).Hash
"$sha  f1opt-portable-windows.zip" | Out-File -Encoding ascii "$ZIP.sha256"

$sizeMB = [math]::Round((Get-Item $ZIP).Length / 1MB, 1)
Write-Host "=== 构建完成 ==="
Write-Host "产物: $ZIP ($sizeMB MiB)"
Write-Host "SHA256: $sha"

# 供 CI 下游 step 使用
if ($env:GITHUB_OUTPUT) {
    "sha=$sha" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
    "size=$sizeMB" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
}