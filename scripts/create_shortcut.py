"""创建F1OPT桌面快捷方式"""
import os
import sys
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
def create_shortcut(target_path, shortcut_path, description="", working_dir=""):
    """（未实现）曾打算用 ctypes 直接调 Shell COM 接口写 .lnk。

    MS-SHLLINK 的 CLSID/IID 与 IPersistFile 调用链过于繁琐，
    已改为下方 ``create_shortcut_powershell``（走 WScript.Shell COM）。
    保留本函数仅为记录该决策，勿调用 —— 它恒返回 False。
    """
    return False


def create_shortcut_powershell(target_path, shortcut_path, description="", working_dir=""):
    """通过生成PowerShell脚本文件来创建快捷方式。"""
    ps_script = f'''$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target_path}"
$Shortcut.WorkingDirectory = "{working_dir}"
$Shortcut.Description = "{description}"
$Shortcut.Save()
Write-Output "OK"
'''
    script_path = os.path.join(os.environ.get("TEMP", "."), "_create_shortcut.ps1")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(ps_script)

    import subprocess
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path],
        capture_output=True, text=True, timeout=10
    )
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    print(f"returncode: {result.returncode}")

    # 清理临时文件
    try:
        os.remove(script_path)
    except OSError:
        pass

    return result.returncode == 0


if __name__ == "__main__":
    target = ROOT / "dist" / "f1opt" / "f1opt.exe"
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    shortcut = os.path.join(desktop, "F1OPT.lnk")
    work_dir = ROOT / "dist" / "f1opt"
    desc = "F1调教优化助手"

    print(f"目标: {target}")
    print(f"快捷方式: {shortcut}")
    print(f"工作目录: {work_dir}")

    # 确认目标文件存在
    if not os.path.exists(target):
        print(f"错误: 目标文件不存在: {target}")
        sys.exit(1)

    # 确认桌面目录存在
    if not os.path.exists(desktop):
        print(f"错误: 桌面目录不存在: {desktop}")
        sys.exit(1)

    success = create_shortcut_powershell(target, shortcut, desc, work_dir)
    if success:
        print(f"\n✅ 桌面快捷方式已创建: {shortcut}")
    else:
        print("\n❌ 创建快捷方式失败")
        sys.exit(1)
