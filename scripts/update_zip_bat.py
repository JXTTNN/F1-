"""更新zip包中的bat文件。"""
import os
import zipfile

zip_path = r"D:\F1OPT-Test\dist\F1OPT-portable.zip"
bat_path = r"D:\F1OPT-Test\dist\一键启动.bat"

# 读取修复后的bat内容
with open(bat_path, encoding="utf-8") as f:
    bat_content = f.read()

# 创建临时zip，替换bat文件
temp_zip = zip_path + ".tmp"
with zipfile.ZipFile(zip_path, "r") as zin:
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if "一键启动.bat" in item.filename:
                zout.writestr(item.filename, bat_content.encode("utf-8"))
                print(f"已替换: {item.filename}")
            else:
                zout.writestr(item, zin.read(item.filename))

os.replace(temp_zip, zip_path)
print(f"zip包已更新: {zip_path}")
print(f"zip大小: {os.path.getsize(zip_path)} 字节")

# 验证zip中的bat文件
with zipfile.ZipFile(zip_path, "r") as z:
    for name in z.namelist():
        if "一键启动" in name or name.endswith(".bat"):
            content = z.read(name).decode("utf-8")
            has_bug = "LSS :20" in content
            has_fix = "LSS 20 " in content
            print(f"  {name}: bug={has_bug}, fixed={has_fix}")