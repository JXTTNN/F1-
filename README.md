# F1OPT - F1 2026 Setup Optimizer

> Real-time telemetry, driver feedback, and setup search for EA Sports F1.

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](https://github.com/JXTTNN/F1-)

---

## Features

- **Real-time Telemetry**: F1 25 binary protocol (16 packet types), async UDP listener, 60Hz unified frames
- **Performance Model**: Predicts lap time from 19 setup + 10 track + 8 driver features, MAE < 0.4s
- **Setup Search**: Differential Evolution + Bayesian Optimization across 24 tracks
- **Driver Feedback**: 11 feedback dimensions with three precision levels (corner, sector, overall)
- **AI Chat**: Pluggable chat backend (OpenAI, Ollama, or local LLM)
- **API Server**: FastAPI with 30+ REST endpoints + WebSocket real-time streaming
- **Cross-Platform**: Windows x64 and Linux support

## Quick Start

```bash
pip install -e .
f1opt --help
```

### Driver Feedback

```bash
# Corner-level feedback
f1opt feedback --track suzuka --question "Why am I understeering into T1?"

# Sector-level feedback
f1opt feedback --track suzuka --question "S2 feels slow through the esses"

# Overall feedback
f1opt feedback --track bahrain --question "How much time can I still find?"
```

### Setup Search

```bash
f1opt search --track suzuka --iterations 100
f1opt bayesian --track monza --iterations 15
f1opt predict --track suzuka --setup-json '...'
```

### Training & Server

```bash
f1opt train --iterations 500
f1opt serve --host 0.0.0.0 --port 8000
```

## Project Structure

```
f1opt/
  api/           FastAPI backend (30+ REST endpoints + WebSocket)
  cli.py         CLI entry point
  config.py      Environment-based settings
  data/          Track data, setup schema, benchmarks
  driver/        Driver profile & behavior analysis
  feedback/      AI feedback engine, intent recognition, causal explanations
  model/         Performance model, Bayesian & DE search, physics sim
  observability/ Metrics, logging, profiling, audit, tracing
  pipeline/      Training and inference orchestration
  telemetry/     UDP listener, packet parsers, alignment, aggregation, replay
  ui/            Frontend dashboard (D3/ECharts), static assets
```

## Configuration

Set via `F1OPT_` prefixed environment variables:

| Variable | Default | Description |
|---|---|---|
| `F1OPT_UDP_HOST` | `0.0.0.0` | UDP listen address |
| `F1OPT_UDP_PORT` | `20777` | UDP listen port |
| `F1OPT_API_HOST` | `127.0.0.1` | API host |
| `F1OPT_API_PORT` | `8000` | API port |
| `F1OPT_DATA_DIR` | `data_store` | Data directory |
| `F1OPT_LLM_BACKEND` | `none` | LLM: `none`/`openai`/`ollama`/`local` |
| `F1OPT_LLM_API_KEY` | (empty) | API key for cloud backend |
| `F1OPT_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `F1OPT_LOG_LEVEL` | `INFO` | Log level |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ --cov=f1opt --cov-report=term-missing
ruff check f1opt tests
mypy f1opt
```

## Build

```bash
pip install pyinstaller
pyinstaller f1opt.spec
```

## License

Proprietary. © F1OPT Team.
