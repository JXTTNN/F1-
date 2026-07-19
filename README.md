# F1OPT — F1 2026 Setup Optimizer

> Real-time telemetry, driver feedback, and setup search for EA Sports F1.

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Download

**[Download the latest release](https://github.com/JXTTNN/F1-/releases)** — pre-built binary, no Python or setup required.

```bash
./f1opt --help
```

---

## Project Structure

```
├── f1opt/                  Source code
│   ├── api/                  FastAPI server (30+ endpoints + WebSocket)
│   ├── cli.py                Command-line interface
│   ├── config.py             Configuration
│   ├── data/                 Track data, setup schema, benchmarks
│   ├── driver/               Driver profile analysis
│   ├── feedback/             Feedback engine, intent recognition, causal explanations
│   ├── model/                Performance model, bayesian search, strategy
│   ├── observability/        Metrics, logging, profiling, audit, tracing
│   └── telemetry/            UDP listener, packet parser, alignment, aggregation
│
├── tests/                  Tests
├── docs/                   Documentation
├── examples/               Usage examples
├── .github/workflows/      CI (lint, type check, test, coverage)
├── pyproject.toml          Project configuration
└── f1opt.spec              PyInstaller build spec
```

## Quick Start

### Driver Feedback

```bash
# Corner-level feedback
f1opt feedback --track suzuka --question "Why am I understeering into T1?"

# Sector-level feedback
f1opt feedback --track suzuka --question "S2 feels slow through the esses"

# Overall feedback
f1opt feedback --track bahrain --question "How much time can I still find?"
```

Three levels of precision: **corner**, **sector**, and **overall** — automatically detected.

### Setup Search

```bash
f1opt search --track suzuka --iterations 100        # Differential evolution
f1opt bayesian --track monza --iterations 15          # Bayesian optimization
f1opt predict --track suzuka --setup-json '...'       # Predict lap time
```

### Training & Server

```bash
f1opt train --iterations 500                         # Train the performance model
f1opt serve --host 0.0.0.0 --port 8000               # Start API server
```

## Features

### Telemetry
- F1 25 binary protocol (all 16 packet types)
- Async UDP listener, 60Hz unified frames
- Parquet persistence + WebSocket real-time streaming

### Performance Model
- Predicts lap time from setup parameters
- 19 setup + 10 track + 8 driver features
- Held-out MAE < 0.4s, p99 latency < 1ms

### Driver Feedback
- 11 feedback dimensions + usage examples (Chinese/English)
- Three precision levels (corner, sector, overall)
- Pluggable chat backend (local AI or cloud API)

### Setup Search
- Differential Evolution + Bayesian Optimization
- Online residual correction
- 24 tracks, aligned with EA F1 2026 benchmarks

### Production Quality
- Observability: metrics, structured logging, profiling, audit trail
- Health checks: /api/health, /api/livez, /api/readyz
- Rate limiting (per-IP)
- CI: GitHub Actions + pytest + ruff + mypy

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `F1OPT_UDP_HOST` | `127.0.0.1` | UDP listen address |
| `F1OPT_UDP_PORT` | `20777` | UDP listen port |
| `F1OPT_DATA_DIR` | `./data_store` | Data directory |
| `F1OPT_CHAT_BACKEND` | `none` | Chat: `none` / `openai` / `local` |
| `F1OPT_CHAT_API_KEY` | (empty) | API key |
| `F1OPT_CHAT_MODEL` | `gpt-4o-mini` | Model name |
| `F1OPT_OTEL_EXPORT` | (empty) | OpenTelemetry endpoint (optional) |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ --cov=f1opt --cov-report=term-missing
ruff check f1opt tests
mypy f1opt
```

## License

Proprietary. © F1OPT Team.