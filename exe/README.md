# EXE 打包

两种方案任选其一：

## 方案 A：PyInstaller onedir (release.yml)

- 产物：`f1opt-windows.zip` (≈ 220 MB)
- 命令：`build.bat` 或 `pyinstaller exe/f1opt.spec --noconfirm`
- 双击 `dist/f1opt/f1opt.exe` 即可运行

### 使用

```bat
build.bat
```

产物: `dist/f1opt/f1opt.exe`

## 方案 B：Embeddable Python 便携包 (release-portable.yml)  **[推荐给闪退用户]**

- 产物：`f1opt-portable-windows.zip` (≈ 600 MB)
- 触发：推 `p*` 标签 → CI 构建
- 优势：**零 PyInstaller 冻结**，彻底避免「解压失败 => 闪退」


### 使用

```bat
# 手动构建 (仅调试用)
powershell -ExecutionPolicy Bypass -File exe/portable-build.ps1

# 正式版构建 (CI)
git tag p1.4.2
git push origin p1.4.2
```

产物解压后：
- 双击 `f1opt.bat` 即可运行
- 体积 `~600MB` (包含 torch)

## 文件说明

| 文件 | 用途 |
|---|---|
| `f1opt.spec` | PyInstaller 打包配置 |
| `portable-build.ps1` | 便携包构建脚本 |
| `version_info.txt` | Windows EXE 版本信息资源 |
| `build.bat` | onedir 一键构建 |
