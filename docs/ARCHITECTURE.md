# F1OPT 系统架构

## 数据流 (Data Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                    F1 25 Game (UDP port 20777)                  │
│  16 packet types: Motion, Session, LapData, Event, ...          │
│  60Hz per frame, 22 cars (11 teams × 2, including Cadillac)     │
└───────────────────────────────┬─────────────────────────────────┘
                                │ binary packets (29B header + body)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  f1opt.telemetry                                                │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ packets  │ → │ listener │ → │ aligner  │ → │aggregator│     │
│  │ parser   │   │ UDP recv │   │ 60Hz sync│   │ Parquet  │     │
│  │ 16 types │   │ async    │   │ 47 fields│   │ storage  │     │
│  └──────────┘   └──────────┘   └────┬─────┘   └──────────┘     │
│                                     │                            │
│                                     │ unified frames             │
└─────────────────────────────────────┼────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  f1opt.model                                                    │
│                                                                  │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│  │ surrogate  │  │ optimizer    │  │ online_correction    │     │
│  │ 37→10 pred │  │ scipy DE     │  │ observation buffer   │     │
│  │ sector+resp│  │ Bayesian GP  │  │ residual learning     │     │
│  └────────────┘  └──────────────┘  └──────────────────────┘     │
│                                                                  │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│  │ train      │  │ diagnostics  │  │ bayesian             │     │
│  │ synthetic  │  │ calibration  │  │ GP + EI/UCB/PI       │     │
│  │ 24×2K data │  │ bootstrap    │  │ search               │     │
│  └────────────┘  └──────────────┘  └──────────────────────┘     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ predictions + responses
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  f1opt.feedback                                                 │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ engine   │  │ prompts  │  │ intent   │  │ causal   │       │
│  │ rules+AI│  │ 11 dims  │  │ 8 types  │  │ WhatIf   │       │
│  │ feedback │  │ 12 sample│  │ granular │  │ analysis │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ conversation (multi-turn memory)                      │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ feedback + recommendations
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│  f1opt.api (FastAPI)                                            │
│                                                                  │
│  REST (30+):                        WebSocket:                  │
│  /api/health  /api/livez  /api/readyz  /ws/telemetry            │
│  /api/predict  /api/search  /api/feedback                       │
│  /api/whatif  /api/causal/explain                               │
│  /api/analytics/lap  /api/analytics/anomalies                   │
│  /api/metrics  /api/tracks  /api/setup  /api/audit              │
│                                                                  │
│  Middleware: rate limiting / tracing                             │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### telemetry

| Module | Purpose |
|---|---|
| `packets.py` | F1 25 binary protocol parser, 16 packet types, 29B header |
| `listener.py` | Async UDP listener (port 20777), drop-oldest backpressure |
| `aligner.py` | Multi-source alignment, 60Hz unified frames, 47 fields |
| `aggregator.py` | Clean lap detection, Parquet persistence |

### model

| Module | Purpose |
|---|---|
| `surrogate.py` | Performance model, 37→10 prediction, predict_batch() |
| `train.py` | Physics-inspired synthetic data training, 24 tracks × 2000 setups |
| `optimizer.py` | Differential Evolution search (scipy, vectorized) |
| `bayesian.py` | Bayesian Optimization (GP + EI/UCB/PI) |
| `diagnostics.py` | Calibration curves, bootstrap metrics, residual analysis |
| `online_correction.py` | Observation buffer, residual learning |

### feedback

| Module | Purpose |
|---|---|
| `engine.py` | Feedback engine with rules and AI enhancement |
| `prompts.py` | Prompt templates (11 dimensions, 12 examples, 3 precision levels) |
| `intent.py` | Intent classification (8 types) + granularity detection (corner/sector/overall) |
| `causal.py` | WhatIf analysis + causal explanation |
| `conversation.py` | Multi-turn conversation memory |

### observability

| Module | Purpose |
|---|---|
| `metrics.py` | Metrics registry (latency histograms) |
| `logging.py` | Structured logging |
| `profiler.py` | Performance profiler |
| `audit.py` | Audit logger (append-only JSONL) |
| `tracing.py` | Distributed tracing (best-effort) |

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | PyTorch |
| API | FastAPI + Uvicorn + WebSocket |
| Optimization | SciPy (Differential Evolution) + scikit-learn (Gaussian Process) |
| Data | PyArrow (Parquet), NumPy |
| Observability | structlog, slowapi, OpenTelemetry |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Code Quality | ruff, mypy |
| CI/CD | GitHub Actions |
| Packaging | PyInstaller |