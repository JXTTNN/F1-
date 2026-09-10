# F1OPT 便携包构建脚本 (基于 python-build-standalone)
# 零 PyInstaller：直接跑官方 Python，杜绝自解压/冻结导致的闪退。
#
# 本地手动构建:  powershell -ExecutionPolicy Bypass -File exe/portable-build.ps1
# CI 构建:       .github/workflows/release-portable.yml (推 p* tag 触发)
#
# 产物: <OutDir>/f1opt-portable-windows.zip (+ .sha256)
#
# 注意: python.org 在 GitHub Actions 节点上不稳定 (偶发 404/被 CDN 拦截),
#       因此改用 GitHub 自家的 astral-sh/python-build-standalone 发行包
#       (CI 直连 GitHub 无阻, 且内置 pip, 无需 get-pip 引导)。

param(
    # 从 git tag (p1.4.2) 或手动传入, 仅写入 README 用
    [string]$Version = "1.4.2",
    # 与项目 requires-python (>=3.11) 一致
    [string]$PyVersion = "3.11",
    # 产物输出目录 (CI 传仓库根, 本地默认当前目录)
    [string]$OutDir = "."
)

$ErrorActionPreference = "Stop"

Write-Host "=== 构建 f1opt-portable v$Version (Python $PyVersion, python-build-standalone) ==="

# 仓库根 = 本脚本所在目录 (exe/) 的上一级
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "仓库根: $RepoRoot"

# 1. 从 GitHub API 查找最新 python-build-standalone 资产 (x86_64-windows-msvc install_only)
Write-Host "[1/7] 查询 python-build-standalone 最新 release ..."
$releasesUrl = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
$rel = Invoke-RestMethod -Uri $releasesUrl -Headers @{ "User-Agent" = "f1opt-portable-build" }
$asset = $rel.assets | Where-Object {
    $_.name -match "cpython-$PyVersion" -and
    $_.name -match "x86_64-pc-windows-msvc-install_only" -and
    $_.name -match "\.tar\.gz$" -and
    $_.name -notmatch "debug"
} | Select-Object -First 1
if (-not $asset) { throw "未找到匹配的 python-build-standalone 资产 (cpython-$PyVersion x86_64-pc-windows-msvc-install_only.tar.gz)" }
Write-Host "选定资产: $($asset.name)"

# 2. 下载
$WORK = $env:RUNNER_TEMP
if (-not $WORK) { $WORK = (Get-Location).Path }
$tarGz = Join-Path $WORK $asset.name
Write-Host "[2/7] 下载 $($asset.browser_download_url) ..."
& curl.exe -fSL --retry 3 --max-time 300 -o $tarGz $asset.browser_download_url
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tarGz) -or (Get-Item $tarGz).Length -lt 10MB) {
    throw "python-build-standalone 下载失败"
}

# 3. 解压 (tar.gz -> 得到 python/ 目录)
$PORTABLE_BASE = Join-Path $WORK "f1opt-portable"
if (Test-Path $PORTABLE_BASE) { Remove-Item $PORTABLE_BASE -Recurse -Force }
New-Item -ItemType Directory -Path $PORTABLE_BASE | Out-Null
Write-Host "[3/7] 解压 ..."
tar -xzf $tarGz -C $PORTABLE_BASE
# python-build-standalone 解压后是 $PORTABLE_BASE\python\<内容>
$PYDIR = Join-Path $PORTABLE_BASE "python"
if (-not (Test-Path (Join-Path $PYDIR "python.exe"))) {
    # 某些版本结构可能是 python\install\...；做一次搜索兜底
    $found = Get-ChildItem -Path $PORTABLE_BASE -Recurse -Filter "python.exe" | Select-Object -First 1
    if (-not $found) { throw "解压后未找到 python.exe" }
    $PYDIR = $found.DirectoryName
    Write-Host "python.exe 实际位置: $PYDIR"
}
Write-Host "Python 目录: $PYDIR"

# 4. 验证 pip 可用 (python-build-standalone install_only 自带 pip; 若无则引导)
Write-Host "[4/7] 检查 pip ..."
& "$PYDIR\python.exe" -m pip --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip 缺失，引导 get-pip ..."
    & curl.exe -fSL --retry 3 -o (Join-Path $PYDIR "get-pip.py") "https://bootstrap.pypa.io/get-pip.py"
    & "$PYDIR\python.exe" (Join-Path $PYDIR "get-pip.py") --quiet
    if ($LASTEXITCODE -ne 0) { throw "get-pip 失败" }
}

# 5. 安装 f1opt + 全部依赖 (torch 等) 到该 Python 的 site-packages
Write-Host "[5/7] 安装 f1opt + 依赖 (torch 体积较大，请耐心) ..."
& "$PYDIR\python.exe" -m pip install "$RepoRoot" --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install 失败" }

# 6. 冒烟测试
Write-Host "[6/7] 冒烟测试 python -m f1opt.cli --help ..."
& "$PYDIR\python.exe" -m f1opt.cli --help
if ($LASTEXITCODE -ne 0) { throw "便携包 --help 测试失败" }

# 7. 生成启动器 + 说明，清理缓存
$batDir = $PORTABLE_BASE
@"
@echo off
cd /d "%~dp0\python"
python.exe -m f1opt.cli serve
pause
"@ | Out-File -FilePath (Join-Path $batDir "启动-F1OPT.bat") -Encoding ascii

@"
@echo off
cd /d "%~dp0\python"
python.exe -m f1opt.cli %*
"@ | Out-File -FilePath (Join-Path $batDir "f1opt.bat") -Encoding ascii

@"
F1OPT Portable v$Version
========================

运行方法（二选一）：
  - 双击 `启动-F1OPT.bat`   -> 启动实时面板 (serve)，浏览器自动打开
  - 双击 `f1opt.bat` + 命令行参数 -> 高级用法，如
        f1opt.bat search --track suzuka --iterations 100

说明：
  - 基于 python-build-standalone Python $PyVersion，零 PyInstaller 冻结，
    彻底避开 onefile/onedir 的「解压失败 => 闪退」问题。
  - 首次 `serve` 加载 torch 模型约 3-10 秒。
  - 建议解压到 SSD 盘。

可选 LLM 配置（同目录 .env 或系统环境变量）：
  F1OPT_LLM_BACKEND=builtin   (默认，内置小模型，离线可用)
  F1OPT_LLM_BACKEND=local     (本地 Ollama，需先 ollama pull llama3.1)
  F1OPT_LLM_BACKEND=openai    (OpenAI 兼容云端)
  F1OPT_LLM_API_KEY=sk-...
  F1OPT_LLM_MODEL=gpt-4o-mini
"@ | Out-File -FilePath (Join-Path $batDir "README.txt") -Encoding UTF8

Get-ChildItem -Path $PORTABLE_BASE -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 打包 + 校验
$OutDirFull = (Resolve-Path $OutDir).Path
$ZIP = Join-Path $OutDirFull "f1opt-portable-windows.zip"
if (Test-Path $ZIP) { Remove-Item $ZIP -Force }
Write-Host "[7/7] 打包 $ZIP ..."
Compress-Archive -Path (Join-Path $PORTABLE_BASE "*") -DestinationPath $ZIP -Force
$sha = (Get-FileHash $ZIP -Algorithm SHA256).Hash
"$sha  f1opt-portable-windows.zip" | Out-File -Encoding ascii "$ZIP.sha256"

$sizeMB = [math]::Round((Get-Item $ZIP).Length / 1MB, 1)
Write-Host "=== 构建完成 ==="
Write-Host "产物: $ZIP ($sizeMB MiB)"
Write-Host "SHA256: $sha"

if ($env:GITHUB_OUTPUT) {
    "sha=$sha" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
    "size=$sizeMB" | Out-File -FilePath $env:GITHUB_OUTPUT -Encoding utf8 -Append
}