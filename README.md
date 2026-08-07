# F1OPT — F1 2026 赛车调校优化系统

> 实时遥测采集 · 车手反馈分析 · 调校参数搜索 · 本地 AI 增强

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 下载

**[下载最新版本](https://github.com/JXTTNN/F1-/releases)** — 预编译 EXE，无需安装 Python 或任何依赖。

```bash
./f1opt --help
```

---

## 项目目录

```
├── f1opt/                   核心源代码
│   ├── api/                   FastAPI 服务 (30+ 接口 + WebSocket)
│   ├── cli.py                 命令行入口
│   ├── config.py              配置管理
│   ├── data/                  赛道数据、调校参数、F1 2026 基准
│   ├── driver/                车手画像分析
│   ├── feedback/              反馈引擎、意图识别、因果解释
│   ├── model/                 性能模型、贝叶斯搜索、策略优化
│   ├── observability/         监控、日志、性能分析、审计追踪
│   ├── telemetry/             UDP 监听、数据包解析、数据对齐、聚合
│   └── ui/                    前端界面 (仪表盘)
│
├── tests/                   测试代码
├── docs/                    文档
│   ├── ARCHITECTURE.md       架构设计
│   └── F1_2026_Research_Summary.md  技术研究摘要
├── examples/                使用示例
├── 半成品/                   未完成/实验性内容
│   ├── R10_迭代训练/          R10 LLM 训练成果
│   └── .trae/                迭代开发笔记
├── .github/workflows/       CI 流水线 (代码检查、类型检查、测试、覆盖率)
├── pyproject.toml           项目配置
└── f1opt.spec               PyInstaller 打包配置
```

## 快速开始

### 车手反馈

```bash
# 弯道级反馈
f1opt feedback --track suzuka --question "为什么 T1 入弯总推头？"

# 扇区级反馈
f1opt feedback --track suzuka --question "S2 连续弯那段指向性太差"

# 整体反馈
f1opt feedback --track bahrain --question "还能再快多少？"
```

系统自动识别问题精确度，支持三级粒度：**弯道**、**扇区**、**整体**。

### 调校搜索

```bash
f1opt search --track suzuka --iterations 100        # 差分进化搜索
f1opt bayesian --track monza --iterations 15          # 贝叶斯优化
f1opt predict --track suzuka --setup-json '...'       # 预测圈速
```

### 模型训练与 API 服务

```bash
f1opt train --iterations 500                         # 训练性能模型
f1opt serve --host 0.0.0.0 --port 8000               # 启动 API 服务
```

## 功能特性

### 遥测系统
- F1 25/26 二进制协议（全部 16 种数据包类型）
- 异步 UDP 监听，60Hz 统一帧对齐
- Parquet 持久化存储 + WebSocket 实时推送
- 自适应队列、连接健康监控、延迟统计

### 性能模型
- 从调校参数预测圈速
- 21 维调校参数 + 10 维赛道特征 + 8 维车手特征
- 留出法 MAE < 0.4s，p99 延迟 < 1ms

### 车手反馈
- 12 个反馈维度 + 使用示例（中/英双语）
- 三级精度（弯道 / 扇区 / 整体）
- 可插拔 LLM 后端（本地 AI 或云端 API）
- 车手反馈模板系统（结构化输入辅助）
- LLM 生命周期管理（游戏期间不加载，跑完后才启动）

### 调校搜索
- 差分进化 + 贝叶斯优化
- 帕累托前沿（圈速 vs 轮胎磨损 多目标优化）
- 在线残差校正
- 24 条官方赛道，对齐 EA F1 2026 基准

### 2026 新特性
- 主动空气动力学（X-mode / Z-mode）
- 50:50 混合动力分配
- MGU-K 增强 + DRS 替换为主动空力模式
- Pirelli 2026 轮胎模型

### 工程质量
- 可观测性：指标、结构化日志、性能分析、审计追踪
- 健康检查：`/api/health`、`/api/livez`、`/api/readyz`
- 速率限制（按 IP）
- CI：GitHub Actions + pytest + ruff + mypy
- Windows EXE 打包（PyInstaller + UPX 压缩）

## 配置

通过环境变量或 `.env` 文件设置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `F1OPT_UDP_HOST` | `127.0.0.1` | UDP 监听地址 |
| `F1OPT_UDP_PORT` | `20777` | UDP 监听端口 |
| `F1OPT_DATA_DIR` | `./data_store` | 数据存储目录 |
| `F1OPT_CHAT_BACKEND` | `none` | LLM 后端：`none` / `openai` / `local` |
| `F1OPT_CHAT_API_KEY` | (空) | API 密钥 |
| `F1OPT_CHAT_MODEL` | `gpt-4o-mini` | 模型名称 |
| `F1OPT_OTEL_EXPORT` | (空) | OpenTelemetry 端点（可选） |

## 开发

```bash
pip install -e ".[dev]"
pytest tests/ --cov=f1opt --cov-report=term-missing
ruff check f1opt tests
mypy f1opt
```

## 更多文档

| 内容 | 位置 |
|---|---|
| 快速开始 | [QUICKSTART.md](QUICKSTART.md) |
| 架构设计 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 技术研究 | [docs/F1_2026_Research_Summary.md](docs/F1_2026_Research_Summary.md) |
| 使用示例 | [examples/](examples/) |
| API 文档 | 启动 `./f1opt serve` 后访问 http://127.0.0.1:8000/docs |
| 半成品/WIP | [半成品/](半成品/) |

## 许可证

私有项目。© F1OPT Team.