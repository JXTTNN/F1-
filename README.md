# F1OPT — F1 2026 赛车调校助手

> 接遥测 → 问问题 → 出调校。中英文自然语言反馈，本地 AI 驱动。

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 下载即用

[下载 EXE](https://github.com/JXTTNN/F1-/releases) — 无需 Python，双击运行。

---

## 三句话上手

```bash
# ① 问问车哪里不对
f1opt feedback --track suzuka --question "T1 入弯总推头怎么办？"

# ② 搜一搜最优调校
f1opt search --track suzuka --iterations 100

# ③ 边跑边分析
f1opt serve          # 启动服务 → 开游戏跑圈 → 停遥测自动出报告
```

系统自动识别你的问题是 **弯道级**、**扇区级** 还是 **整体级**，回答匹配对应精度。

---

## 能做什么

| 模块 | 干什么 |
|---|---|
| `feedback` | 车手反馈 — 你说问题，它分析遥测 + 给出调校建议，中英双语 |
| `search` | 调校搜索 — 差分进化 / 贝叶斯优化，24 条赛道，帕累托前沿 |
| `serve` | API 服务 — FastAPI + WebSocket，实时遥测流 + 跑后自动分析 |
| `telemetry` | 遥测采集 — F1 25/26 UDP 协议全解析，60Hz 对齐，自适应队列 |
| `model` | 性能模型 — 21 维调校参数预测圈速，Pirelli 2026 轮胎模型 |

---

## 目录

```
f1opt/         源代码
├── api/         FastAPI 服务
├── cli.py       命令行入口
├── data/        赛道 & 调校数据
├── driver/      车手画像
├── feedback/    反馈引擎 + LLM
├── model/       优化器 & 轮胎物理
├── telemetry/   UDP 监听 & 解析
└── ui/          仪表盘
tests/         测试
docs/          文档
半成品/         实验性内容
```

---

## 配置

环境变量或 `.env` 文件：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `F1OPT_UDP_HOST` | `127.0.0.1` | 游戏遥测地址 |
| `F1OPT_UDP_PORT` | `20777` | 游戏遥测端口 |
| `F1OPT_CHAT_BACKEND` | `none` | LLM 后端：`openai` / `local` |
| `F1OPT_CHAT_API_KEY` | — | API 密钥 |
| `F1OPT_DATA_DIR` | `./data_store` | 数据目录 |

---

## 开发

```bash
pip install -e ".[dev]"
pytest tests/ --cov=f1opt
ruff check f1opt tests
```