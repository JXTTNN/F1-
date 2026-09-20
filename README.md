# F1OPT — F1 2026 赛车调校助手

> 接遥测 → 问问题 → 出调校。中英文自然语言反馈，本地 AI 驱动。

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 下载即用（单文件 EXE）

> **最新版 v1.3.1** — Windows 单文件，无需安装 Python，**双击即可用**。云端构建（Actions），**内置小模型开箱即用**：收集车手反馈 → 针对性改进调教，离线运行、随反馈在线学习；EXE 本体已通过全量真人式点击审计。

| 版本 | 平台 | 下载 |
|------|------|------|
| **v1.3.1**（最新，推荐） | Windows | [⬇️ 下载 f1opt.exe](https://github.com/JXTTNN/F1-/releases/download/v1.3.1/f1opt.exe) |
| 全部历史版本 | 跨平台 | [GitHub Releases](https://github.com/JXTTNN/F1-/releases) |

**v1.3.1 校验**：SHA256 与文件大小见 [Release 页说明](https://github.com/JXTTNN/F1-/releases/tag/v1.3.1)（随云端构建自动生成）。

> 双击 `f1opt.exe` 会看到启动横幅，随后自动在默认浏览器打开实时面板（首次解压约几秒）。实时遥测 + 车手反馈 + 调教编辑 + 最优搜索全部在一个页面。关闭控制台窗口或按 Ctrl+C 即退出。
>
> **内置小模型（默认启用，开箱即用）**：`F1OPT_LLM_BACKEND=builtin`（默认）。跑完停车后在反馈框描述问题（推头/甩尾/打滑/锁死…），内置模型即时给出带数值的针对性调教修正，并把反馈样本持久化到本地、随使用不断学习；跑圈实时反馈始终走规则引擎（零延迟），停表后分析自动启用。也可切换 `local`（Ollama）/ `openai` 后端获得自由文本增强。

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
| `model` | 性能模型 — 20 维调校参数预测圈速（含配重/主动空力；F1 2026 车库无引擎制动），Pirelli 2026 轮胎模型 |

---

## F1 2026 完整支持

严格贴合 EA Sports F1 2026 Season Pack：

- **11 车队 / 22 车手**：完整阵容（含 Audi、Cadillac 新车队；UDP 预留 24 车位）
- **主动空力**：X-Mode（直道低阻）/ Z-Mode（弯道下压力）分析
- **Pirelli C0-C5 轮胎（+C6 待核）**：2026 轮胎模型（新增 C0 最硬配方；C6 存废待核）与温度梯度
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

## 什么新变化 (2026-09-19)

- **神经网络模拟优化真正生效（不再需要 PyTorch）**：旧分支依赖 PyTorch + 权重文件，
  在本机静默降级成"纯规则"，UI 还提示"缺 PyTorch/权重"。现已改为
  **遥测锚定训练 + 纯标准库推理**：
  - 训练数据：80k+ 条真实 2026 逐弯遥测做锚点 + 文档化的准稳态车辆动力学，
    构建「调教 → 圈速增量」18,200 条样本（`scripts/build_setup_sim_dataset.py`）；
  - 模型：纯标准库 MLP，验证集 **MAE 53 ms / R² 0.996**（`data/models/setup_sim_nn.json`）；
  - 管线：**参数矩阵给方向 → 神经网络不断模拟优化**（坐标上升，候选调教由 NN
    预测圈速，保留更优且不牺牲车手需求满足度的候选），报告输出迭代轨迹与预计提升。
- **反馈生命周期**：生成建议后自动清除本赛道反馈，避免跨圈串味（点击"生成调教"后
  反馈即被消费，下一圈请重新录入）。
- **逐弯通过时间**：`LapAggregator` 新增按弯计时（`corner_times_s`），使
  "车手没反馈的问题"能由遥测自动发现并给出方案。
- **假绿/失效检查修正**：参数项数（21→20）、强度区间（0-5→1-3）、过时断言与
  非法测试载荷全部修正；`pytest -q` **2092 passed / 3 skipped**，`ruff` 全绿。
- **优化性能验证**（用户要求"用现有遥测验证"）：6 赛道 × 3 场景端到端，NN 判定
  平均提升 **+367 ms**、解析仿真器独立复核 **+631 ms**，18/18 场景均有提升
  （`scripts/verify_optimization_telemetry.py` → `docs/Optimization_Verification.md`）。

> 架构与复算步骤详见 [`docs/Setup_Simulation_Optimizer.md`](docs/Setup_Simulation_Optimizer.md)。

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