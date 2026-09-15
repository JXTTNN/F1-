"""安全清理C盘临时文件"""
import os
import shutil
import sys
from pathlib import Path

def clean_temp():
    """清理TEMP目录中的安全可删除文件"""
    temp_dir = Path(os.environ.get("TEMP", ""))
    if not temp_dir.exists():
        print("TEMP目录不存在")
        return 0

    total_freed = 0
    cleaned_count = 0
    failed_count = 0

    # 安全清理目标：Nuitka onefile_* 临时目录
    # 这些是打包过程中产生的临时解压目录，打包完成后不再需要
    targets = list(temp_dir.glob("onefile_*"))

    # WinGet 下载缓存
    winget = temp_dir / "WinGet"
    if winget.exists():
        targets.append(winget)

    # I-Menu-Downod（下载器临时文件）
    imenu = temp_dir / "I-Menu-Downod"
    if imenu.exists():
        targets.append(imenu)

    print(f"找到 {len(targets)} 个清理目标")

    for target in targets:
        try:
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target, ignore_errors=True)
            if not target.exists():
                freed_mb = size / (1024 * 1024)
                total_freed += size
                cleaned_count += 1
                print(f"  ✅ 已清理: {target.name} ({freed_mb:.1f} MB)")
            else:
                failed_count += 1
                print(f"  ⚠️ 部分清理: {target.name} (部分文件被锁定)")
        except Exception as e:
            failed_count += 1
            print(f"  ❌ 失败: {target.name} ({e})")

    # 清理 *.tmp 文件
    tmp_files = list(temp_dir.glob("*.tmp"))
    for tmp_file in tmp_files:
        try:
            size = tmp_file.stat().st_size
            tmp_file.unlink()
            total_freed += size
            cleaned_count += 1
        except Exception:
            failed_count += 1

    # 清理 pip 缓存目录
    pip_cache = Path(os.environ.get("LOCALAPPDATA", "")) / "pip" / "cache"
    if pip_cache.exists():
        try:
            size = sum(f.stat().st_size for f in pip_cache.rglob("*") if f.is_file())
            shutil.rmtree(pip_cache, ignore_errors=True)
            if not pip_cache.exists():
                freed_mb = size / (1024 * 1024)
                total_freed += size
                print(f"  ✅ 已清理: pip缓存 ({freed_mb:.1f} MB)")
        except Exception:
            pass

    freed_mb = total_freed / (1024 * 1024)
    print(f"\n📊 清理完成: {cleaned_count} 项已清理, {failed_count} 项失败")
    print(f"   释放空间: {freed_mb:.1f} MB ({freed_mb/1024:.2f} GB)")
    return total_freed

if __name__ == "__main__":
    clean_temp()