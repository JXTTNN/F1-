"""创建F1OPT桌面快捷方式"""
import os
import struct
import sys

def create_shortcut(target_path, shortcut_path, description="", working_dir=""):
    """用Python标准库创建Windows .lnk快捷方式文件。

    .lnk文件格式参考：MS-SHLLINK规范。
    这里使用最小化二进制格式创建可用的快捷方式。
    """
    # 使用ctypes调用Windows Shell API
    import ctypes
    from ctypes import wintypes

    # 加载shell32
    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32

    # 初始化COM
    ole32.CoInitialize(0)

    # CLSID for ShellLink
    clsid = ctypes.c_buffer(b'\x01\x14\x02\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x46', 16)

    # IID for IShellLinkW
    iid = ctypes.c_buffer(b'\x41\x14\x02\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x46', 16)

    # 定义IPersistFile接口的IID
    iid_persist_file = ctypes.c_buffer(
        b'\x03\x01\x00\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x46', 16
    )

    # 使用CoCreateInstance创建IShellLinkW对象
    # 这个方法太复杂了，改用更简单的方式

    ole32.CoUninitialize()
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
    target = r"D:\F1OPT-Test\dist\F1OPT.exe"
    desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
    shortcut = os.path.join(desktop, "F1OPT.lnk")
    work_dir = r"D:\F1OPT-Test\dist"
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
        print(f"\n❌ 创建快捷方式失败")
        sys.exit(1)