"""Nuitka 打包脚本 - 编译 F1OPT.exe 便携版。

用法: python scripts/build_nuitka.py
产出: dist/F1OPT.exe (单文件便携包) + dist/F1OPT-portable.zip

对齐 design.md 2.10.1 Nuitka 编译方案（C-08，弃用 PyInstaller）：
    - 入口：setup_tuner/cli.py（一键启动：启动 uvicorn + 打开浏览器 + 端口占用检测）
    - --onefile 产单文件便携包；--include-data-dir 内嵌 ui/（HTML/JS/CSS + 24 SVG）
    - .bat 一键启动脚本作为便携包双入口（双击即启，FR-DEP-01/02）
    - Nuitka 产物与 .bat 同时收进 zip

编译命令（封装本脚本）::

    python -m nuitka --onefile --windows-console-mode=disable
      --enable-plugin=no-qt --follow-imports
      --include-data-dir=setup_tuner/ui=ui
      --output-filename=F1OPT.exe
      --output-dir=dist
      setup_tuner/cli.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

# Windows CI 默认编码 cp1252 不支持中文，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# =========================================================================== #
# 常量
# =========================================================================== #
# 仓库根目录（scripts/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 入口模块
ENTRY_MODULE = "setup_tuner/cli.py"

# 输出目录与文件名
OUTPUT_DIR = REPO_ROOT / "dist"
OUTPUT_FILENAME = "F1OPT.exe"
PORTABLE_ZIP = "F1OPT-portable.zip"

# 一键启动脚本
BAT_SCRIPT = REPO_ROOT / "assets" / "一键启动.bat"

# Nuitka 编译超时（秒）—— onefile 模式编译较慢，给足 20 分钟
COMPILE_TIMEOUT = 20 * 60


# =========================================================================== #
# 前置检查
# =========================================================================== #
def check_nuitka_installed() -> None:
    """检查 nuitka 是否已安装。

    Raises:
        SystemExit: nuitka 未安装时打印提示并退出（退出码 1）。
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nuitka", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise FileNotFoundError
        version = result.stdout.strip().splitlines()[0] if result.stdout else "unknown"
        print(f"[check] Nuitka 已安装：{version}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(
            "[error] Nuitka 未安装或不可用。\n"
            "        请执行: pip install nuitka\n"
            "        或在 pyproject.toml 中安装 build 依赖: pip install -e \".[build]\"",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def check_entry_exists() -> None:
    """检查入口模块 setup_tuner/cli.py 是否存在。

    Raises:
        SystemExit: 入口不存在时退出（退出码 1）。
    """
    entry = REPO_ROOT / ENTRY_MODULE
    if not entry.exists():
        print(f"[error] 入口模块不存在：{entry}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[check] 入口模块就绪：{entry.relative_to(REPO_ROOT)}")


def check_ui_dir_exists() -> None:
    """检查 setup_tuner/ui/ 目录是否存在（含 HTML/JS/CSS + 24 SVG）。

    Raises:
        SystemExit: UI 目录不存在时退出（退出码 1）。
    """
    ui_dir = REPO_ROOT / "setup_tuner" / "ui"
    if not ui_dir.exists():
        print(f"[error] UI 资源目录不存在：{ui_dir}", file=sys.stderr)
        raise SystemExit(1)
    svg_dir = ui_dir / "tracks"
    svg_count = len(list(svg_dir.glob("*.svg"))) if svg_dir.exists() else 0
    print(f"[check] UI 目录就绪：{ui_dir.relative_to(REPO_ROOT)}（{svg_count} 个 SVG）")


# =========================================================================== #
# Nuitka 编译
# =========================================================================== #
def build_nuitka_command() -> list[str]:
    """构造 Nuitka 编译命令行参数列表。

    Returns:
        完整的命令行参数列表（含 python -m nuitka 前缀）。
    """
    return [
        sys.executable,
        "-m",
        "nuitka",
        # 单文件便携包
        "--onefile",
        # 自动下载依赖（Dependency Walker 等），CI 非交互模式必需
        "--assume-yes-for-downloads",
        # Windows GUI 模式（无控制台窗口）
        "--windows-console-mode=disable",
        # 禁用 Qt 插件（本项目无 Qt 依赖，加速编译）
        "--enable-plugin=no-qt",
        # 递归跟踪所有 import
        "--follow-imports",
        # 内嵌 UI 静态资源（HTML/JS/CSS + 24 SVG）
        "--include-data-dir=setup_tuner/ui=ui",
        # 输出文件名与目录
        f"--output-filename={OUTPUT_FILENAME}",
        f"--output-dir={OUTPUT_DIR}",
        # 入口模块
        ENTRY_MODULE,
    ]


def run_nuitka_compile() -> None:
    """执行 Nuitka 编译。

    Raises:
        SystemExit: 编译失败时退出（退出码 1）。
    """
    cmd = build_nuitka_command()
    print("\n[build] 开始 Nuitka 编译...")
    print("[build] 命令：", " ".join(cmd))
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            timeout=COMPILE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        print(
            f"[error] Nuitka 编译超时（{COMPILE_TIMEOUT // 60} 分钟）。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if result.returncode != 0:
        print(f"[error] Nuitka 编译失败，退出码：{result.returncode}", file=sys.stderr)
        raise SystemExit(1)

    print("\n[build] Nuitka 编译完成。")


# =========================================================================== #
# 产物验证与打包
# =========================================================================== #
def verify_exe() -> Path:
    """验证 dist/F1OPT.exe 存在且非空。

    Returns:
        exe 文件路径。

    Raises:
        SystemExit: 产物不存在或为空时退出（退出码 1）。
    """
    exe_path = OUTPUT_DIR / OUTPUT_FILENAME
    if not exe_path.exists():
        print(f"[error] 编译产物不存在：{exe_path}", file=sys.stderr)
        raise SystemExit(1)

    size_mb = exe_path.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        print(f"[error] 编译产物异常过小：{size_mb:.2f} MB", file=sys.stderr)
        raise SystemExit(1)

    print(f"[verify] 编译产物就绪：{exe_path.relative_to(REPO_ROOT)}（{size_mb:.2f} MB）")
    return exe_path


def package_portable_zip(exe_path: Path) -> Path:
    """打包 dist/F1OPT.exe + assets/一键启动.bat → dist/F1OPT-portable.zip。

    Args:
        exe_path: 已验证的 exe 文件路径。

    Returns:
        zip 文件路径。
    """
    zip_path = OUTPUT_DIR / PORTABLE_ZIP

    print(f"\n[pack] 打包便携包：{zip_path.relative_to(REPO_ROOT)}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # ① F1OPT.exe
        zf.write(exe_path, arcname=OUTPUT_FILENAME)
        print(f"[pack]   + {OUTPUT_FILENAME}")

        # ② 一键启动.bat（若存在）
        if BAT_SCRIPT.exists():
            zf.write(BAT_SCRIPT, arcname=BAT_SCRIPT.name)
            print(f"[pack]   + {BAT_SCRIPT.name}")
        else:
            print(f"[pack]   ! 跳过 {BAT_SCRIPT.name}（assets/一键启动.bat 不存在）")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[pack] 便携包打包完成：{zip_path.relative_to(REPO_ROOT)}（{zip_size_mb:.2f} MB）")
    return zip_path


# =========================================================================== #
# 主入口
# =========================================================================== #
def main() -> int:
    """Nuitka 打包主入口。

    流程：
        1. 前置检查（nuitka / 入口 / UI 目录）；
        2. 清理旧产物；
        3. Nuitka 编译；
        4. 验证 exe 产物；
        5. 打包便携 zip（exe + bat）。

    Returns:
        退出码：0=成功，1=失败。
    """
    print("=" * 70)
    print("F1OPT Nuitka 打包脚本")
    print("=" * 70)

    # ① 前置检查
    check_nuitka_installed()
    check_entry_exists()
    check_ui_dir_exists()

    # ② 清理旧产物
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old_exe = OUTPUT_DIR / OUTPUT_FILENAME
    old_zip = OUTPUT_DIR / PORTABLE_ZIP
    for old in (old_exe, old_zip):
        if old.exists():
            old.unlink()
            print(f"[clean] 清理旧产物：{old.relative_to(REPO_ROOT)}")

    # ③ Nuitka 编译
    run_nuitka_compile()

    # ④ 验证 exe
    exe_path = verify_exe()

    # ⑤ 打包便携 zip
    package_portable_zip(exe_path)

    print("\n" + "=" * 70)
    print("打包完成！")
    print(f"  EXE:    {exe_path.relative_to(REPO_ROOT)}")
    print(f"  便携包: {OUTPUT_DIR / PORTABLE_ZIP}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())