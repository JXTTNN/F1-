# -*- mode: python ; coding: utf-8 -*-

import sys

block_cipher = None

datas = []

hidden_imports = [
    'f1opt.api', 'f1opt.api.app', 'f1opt.api.extended_app', 'f1opt.api.routes',
    'f1opt.model', 'f1opt.model.surrogate', 'f1opt.model.optimizer',
    'f1opt.model.bayesian', 'f1opt.model.train', 'f1opt.model.validation',
    'f1opt.model.physics', 'f1opt.model.aerodynamics', 'f1opt.model.brake_model',
    'f1opt.model.cache', 'f1opt.model.confidence', 'f1opt.model.diagnostics',
    'f1opt.model.feature_importance', 'f1opt.model.fuel_model', 'f1opt.model.pareto',
    'f1opt.model.pit_crew', 'f1opt.model.pirelli_2026', 'f1opt.model.drs_2026',
    'f1opt.model.energy_budget', 'f1opt.model.ers_model', 'f1opt.model.active_aero',
    'f1opt.model.active_aero_coupling', 'f1opt.model.batch', 'f1opt.model.championship',
    'f1opt.model.cost_cap', 'f1opt.model.development_planner',
    'f1opt.model.driver_fatigue', 'f1opt.model.driver_morale',
    'f1opt.model.drs_coupling', 'f1opt.model.drs_train', 'f1opt.model.grid_penalty',
    'f1opt.model.lap_simulator_2026',
    'f1opt.telemetry', 'f1opt.telemetry.aggregator', 'f1opt.telemetry.aligner',
    'f1opt.telemetry.analytics', 'f1opt.telemetry.gap_filler', 'f1opt.telemetry.listener',
    'f1opt.telemetry.packets', 'f1opt.telemetry.packet_loss',
    'f1opt.telemetry.quality_score', 'f1opt.telemetry.rate_monitor',
    'f1opt.telemetry.replay', 'f1opt.telemetry.safety_car',
    'f1opt.telemetry.sector_times', 'f1opt.telemetry.simulator',
    'f1opt.telemetry.summary', 'f1opt.telemetry.validation',
    'f1opt.driver', 'f1opt.driver.profile',
    'f1opt.feedback', 'f1opt.feedback.engine', 'f1opt.feedback.intent',
    'f1opt.feedback.causal', 'f1opt.feedback.comparison',
    'f1opt.feedback.conversation', 'f1opt.feedback.language',
    'f1opt.feedback.nlg', 'f1opt.feedback.prompts', 'f1opt.feedback.quality',
    'f1opt.observability', 'f1opt.observability.audit',
    'f1opt.observability.logging', 'f1opt.observability.metrics',
    'f1opt.observability.profiler', 'f1opt.observability.tracing',
    'structlog', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.protocols',
]

if sys.platform == 'win32':
    hidden_imports += [
        'asyncio.windows_events', 'asyncio.proactor_events',
        '_overlapped', '_winapi',
    ]

a = Analysis(
    ['f1opt/cli.py'],
    pathex=[],
    binaries=[],
    data=datas,
    hiddenimports=hidden_imports,
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
    a.binaries,
    a.datas,
    [],
    name='f1opt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
