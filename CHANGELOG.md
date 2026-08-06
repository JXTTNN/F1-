# CHANGELOG

## 0.1.1 (Unreleased)

### Added
- Windows compatibility: `SelectorEventLoop` policy for asyncio UDP listener
- Cross-platform CI matrix (ubuntu + windows, Python 3.10/3.11/3.12)
- `llm_model` config field for flexible LLM model selection
- `is_windows` property on Settings
- Benchmark CI workflow (packet parsing + model prediction)
- Pipeline initialization module with Windows event loop patch
- PyInstaller spec: full hidden imports and Windows-specific modules

### Changed
- Reduced minimum Python to 3.10 for broader platform support
- Restructured README with badges, project tree, build instructions
- Expanded `.gitignore` with Windows and PyInstaller entries
- Added `pyproject.toml`: classifiers, keywords, URLs, entry_points, optional deps

## 0.1.0 (2026-07-19)

### Added
- Initial release: F1 2026 Setup Optimizer
- UDP telemetry: F1 25 binary protocol (16 packet types), async listener, 60Hz unified frames
- Performance model: 37-feature multi-task model predicting lap time and sector responses
- Setup search: Differential Evolution + Bayesian Optimization across 24 tracks
- Driver feedback: 11-dimension feedback engine with three precision levels
- API server: FastAPI with 30+ REST endpoints + WebSocket real-time streaming
- CLI: argparse-based interface with 9 subcommands
- Production quality: structured logging, metrics, audit trail, rate limiting, CI/CD
