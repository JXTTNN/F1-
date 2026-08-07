# EXE 打包

## 构建

### Windows

双击 `build.bat` 或命令行运行:

```bat
build.bat
```

产物: `dist/f1opt/f1opt.exe`

### 手动构建

```bash
pip install -e ".[dev]"
pip install pyinstaller
pyinstaller exe/f1opt.spec --noconfirm
```

## 文件说明

| 文件 | 用途 |
|---|---|
| `f1opt.spec` | PyInstaller 打包配置 |
| `version_info.txt` | Windows EXE 版本信息资源 |
| `build.bat` | 一键构建脚本 |
