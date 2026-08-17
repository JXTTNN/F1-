# F1OPT — F1 2026 赛车调校助手

> 接遥测 → 问问题 → 出调校。中英文自然语言反馈，本地 AI 驱动。

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 下载即用（单文件 EXE）

- **Windows EXE**：[下载 f1opt.exe (v1.2.0)](https://github.com/JXTTNN/F1-/releases/download/v1.2.0/f1opt.exe) — 单个文件，无需 Python，**双击即可用**：自动打开实时面板（可切换智能分析中心）；也可命令行运行。
- 全部版本：[GitHub Releases](https://github.com/JXTTNN/F1-/releases)

> 双击 `f1opt.exe` 会看到启动横幅，随后自动在默认浏览器打开 `http://127.0.0.1:8000/`（实时面板，首次解压约几秒）。右上角可切换「智能分析中心」。包含遥测采集、车手反馈、调校优化、模型训练全功能。关闭控制台窗口或按 Ctrl+C 即退出。

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

### 游戏内设置（接遥测）

在 F1 2026 游戏里打开遥测输出：

- **UDP Telemetry** → `On`
- **UDP Format** → `2026`
- **UDP Port** → `20777`
- **UDP IP** → `127.0.0.1`（本机）

设置好后双击 `f1opt.exe`（或命令行运行 `f1opt serve`），跑圈即可实时采集、停后自动出分析报告。

---

## 能做什么

| 模块 | 干什么 |
|---|---|
| `feedback` | 车手反馈 — 你说问题，它分析遥测 + 给出调校建议，中英双语 |
| `search` | 调校搜索 — 差分进化 / 贝叶斯优化，24 条赛道，帕累托前沿 |
| `serve` | API 服务 — FastAPI + WebSocket，实时遥测流 + 跑后自动分析 |
| `telemetry` | 遥测采集 — F1 25/26 UDP 协议全解析，60Hz 对齐，自适应队列 |
| `model` | 性能模型 — 23 维调校参数预测圈速（含发动机制动/配重/主动空力），Pirelli 2026 轮胎模型 |

---

## F1 2026 完整支持

严格贴合 EA Sports F1 2026 Season Pack：

- **11 车队 / 22 车手**：完整阵容（含 Audi、Cadillac 新车队；UDP 预留 24 车位）
- **主动空力**：X-Mode（直道低阻）/ Z-Mode（弯道下压力）分析
- **Pirelli C0-C5 轮胎**：2026 轮胎模型（新增 C0 最硬配方）与温度梯度
- **50-50 动力单元**：MGU-K 350kW + Overtake 模式
- **24 条赛道**：含马德里 Madring 新赛道

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
exe/           EXE 打包 (spec + 构建脚本)
tests/         测试
docs/          文档
半成品/         实验性内容
```

---

## 配置

环境变量或 `.env` 文件：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `F1OPT_UDP_HOST` | `0.0.0.0` | 游戏遥测地址 |
| `F1OPT_UDP_PORT` | `20777` | 游戏遥测端口 |
| `F1OPT_LLM_BACKEND` | `none` | LLM 后端：`openai` / `local` |
| `F1OPT_LLM_API_KEY` | — | API 密钥 |
| `F1OPT_LLM_MODEL` | `gpt-4o-mini` | 模型名称 |
| `F1OPT_DATA_DIR` | `data_store` | 数据目录 |

---

## 详细用法

完整使用方法见 [docs/使用方法.md](docs/使用方法.md) — 涵盖所有命令、参数、API 接口、常见问题.

---

## 开发

```bash
pip install -e ".[dev]"
pytest tests/ --cov=f1opt
ruff check f1opt tests
```

## 打包 EXE

```bash
# Windows: 双击 exe/build.bat
# 或手动:
pyinstaller exe/f1opt.spec --noconfirm
# → 产物为单个 dist/f1opt.exe（one-file 模式）
```