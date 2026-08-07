# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置.

使用方法:
    cd 项目根目录
    pyinstaller exe/f1opt.spec
    → 产物在 dist/f1opt/
"""

import os
import sys

IS_WINDOWS = sys.platform == 'win32'

# 项目根目录 (spec 文件在 exe/ 下, 所以往上一级)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__spec_file__)))

# --- 静态资源 (UI 仪表盘) ---
_ui_static = os.path.join(_ROOT, 'f1opt', 'ui', 'static')
_ui_datas = []
if os.path.isdir(_ui_static):
    for root, _dirs, files in os.walk(_ui_static):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.relpath(src, os.path.dirname(_ui_static))
            _ui_datas.append((src, os.path.dirname(dst)))

a = Analysis(
    [os.path.join(_ROOT, 'f1opt', 'cli.py')],
    pathex=[_ROOT],
    binaries=[],
    datas=_ui_datas,
    hiddenimports=[
        # --- f1opt 内部 ---
        'f1opt.observability.audit',
        'f1opt.observability.tracing',
        # --- 第三方库动态导入 ---
        'slowapi', 'slowapi.errors', 'slowapi.util', 'slowapi.middleware',
        'opentelemetry', 'opentelemetry.api', 'opentelemetry.sdk', 'opentelemetry.trace',
        'structlog', 'structlog.dev', 'structlog.processors', 'structlog.stdlib',
        'httpx', 'httpcore', 'h11',
        # --- pydantic ---
        'pydantic.deprecated.decorator',
        'pydantic.annotated_handlers',
        'pydantic.functional_validators',
        'pydantic.functional_serializers',
        # --- starlette / fastapi / uvicorn ---
        'starlette.middleware', 'starlette.middleware.base', 'starlette.middleware.cors',
        'starlette.middleware.gzip', 'starlette.middleware.errors',
        'starlette.middleware.httpsredirect', 'starlette.middleware.trustedhost',
        'starlette.middleware.wsgi', 'starlette.middleware.sessions',
        'starlette.responses', 'starlette.routing', 'starlette.staticfiles',
        'starlette.templating', 'starlette.concurrency', 'starlette.convertors',
        'starlette.datastructures', 'starlette.exceptions', 'starlette.formparsers',
        'starlette.requests', 'starlette.status', 'starlette.types', 'starlette.websockets',
        'fastapi.middleware', 'fastapi.middleware.cors', 'fastapi.middleware.gzip',
        'fastapi.middleware.trustedhost', 'fastapi.middleware.httpsredirect',
        'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto',
        'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off',
        # --- numpy / scipy / torch ---
        'numpy.core._multiarray_umath', 'numpy.linalg._umath_linalg',
        'numpy.random._common', 'numpy.random.bit_generator',
        'numpy.random._bounded_integers', 'numpy.random._mt19937',
        'numpy.random._pcg64', 'numpy.random._philox', 'numpy.random._sfc64',
        'numpy.random._generator',
        'scipy.spatial.transform._rotation_groups',
        'scipy.spatial.transform._rotation',
        'scipy.sparse.csgraph._validation', 'scipy.sparse.csgraph._tools',
        'scipy.sparse.csgraph._traversal', 'scipy.sparse.csgraph._min_spanning_tree',
        'scipy.sparse.csgraph._flow', 'scipy.sparse.csgraph._matching',
        'scipy.sparse.csgraph._reordering', 'scipy.sparse.csgraph._shortest_path',
        'scipy.sparse.linalg._dsolve.linsolve',
        'scipy.sparse.linalg._eigen.arpack',
        'scipy.sparse.linalg._isolve.iterative',
        'torch', 'torch.utils', 'torch.utils.data', 'torch._C',
        'torch.nn', 'torch.nn.functional', 'torch.optim',
        # --- pyarrow ---
        'pyarrow', 'pyarrow.lib', 'pyarrow._parquet', 'pyarrow.compute',
        'pyarrow.csv', 'pyarrow.dataset', 'pyarrow.feather', 'pyarrow.fs',
        'pyarrow.json', 'pyarrow.parquet', 'pyarrow.types',
        # --- Windows 运行时 ---
        'asyncio.windows_events', 'asyncio.windows_utils',
        'multiprocessing', 'multiprocessing.popen_spawn_win32', 'multiprocessing.spawn',
        'ctypes', 'ctypes.wintypes',
        'encodings', 'encodings.utf_8', 'encodings.gbk', 'encodings.ascii',
        'encodings.latin_1', 'encodings.idna', 'encodings.aliases',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest', 'setuptools', 'pip', 'wheel',
        'tkinter', 'unittest', 'test', 'tests',
        'docutils', 'sphinx', 'IPython', 'jupyter', 'notebook',
        'matplotlib', 'PIL', 'cv2',
        'tensorflow', 'keras', 'tensorboard',
        'sqlalchemy', 'alembic',
        'pandas.tests', 'numpy.tests', 'scipy.tests', 'pyarrow.tests',
    ],
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
    uac_admin=False,
    uac_uiaccess=False,
    icon=None,
    version=os.path.join(_ROOT, 'exe', 'version_info.txt') if IS_WINDOWS else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'python3*.dll', 'vcruntime*.dll', 'msvcp*.dll',
        'api-ms-win-*.dll', 'ucrtbase.dll',
        '_socket.pyd', '_asyncio.pyd', 'select.pyd', '_overlapped.pyd',
    ],
    name='f1opt',
)
