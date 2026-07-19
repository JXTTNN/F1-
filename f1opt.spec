# -*- mode: python ; coding: utf-8 -*-

# Iter-171: 添加 slowapi / opentelemetry hidden imports (新工厂级依赖).
# PyInstaller 静态分析可能漏掉这些动态导入的子模块.

a = Analysis(
    ['/workspace/f1opt/cli.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # slowapi (rate limiting) — 动态导入 middleware
        'slowapi',
        'slowapi.errors',
        'slowapi.util',
        'slowapi.middleware',
        # opentelemetry (tracing) — best-effort, 可选
        'opentelemetry',
        'opentelemetry.api',
        'opentelemetry.sdk',
        'opentelemetry.trace',
        # f1opt 新模块 (Iter-170/171)
        'f1opt.observability.audit',
        'f1opt.observability.tracing',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='f1opt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='f1opt',
)
