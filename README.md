# F1OPT — F1 2026 赛车调校助手

> 接遥测 → 说问题 → 出整体性调校。**本地运行，确定性输出，零第三方 AI 依赖**。
>
> 参数矩阵给方向，神经网络不断模拟优化 —— 模型真的在跑，不是贴标签。

[![CI](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml/badge.svg)](https://github.com/JXTTNN/F1-/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-2114%20collected-brightgreen.svg)](#开发与测试)
[![Runtime AI deps](https://img.shields.io/badge/runtime%20AI%20deps-none-success.svg)](#为什么不需要-pytorch)

---

## 这是什么

F1OPT 读取 **EA F1 2026** 游戏的 UDP 遥测，结合你在赛道图上点出的问题（或纯遥测自动发现的问题），
给出**整套调校建议**：哪些参数、改多少、为什么这么改、代价是什么。

它不猜。整条链路是确定性的、可复现的、可验证的：

```
游戏 UDP 遥测 + 车手反馈（点赛道图）
        │
        ├─ 遥测自动诊断（车手没反馈的问题：逐弯压路肩 / 锁死 / 刮底 / 胎温…）
        ▼
   诊断向量 Dx（9 维需求）
        ▼
   耦合矩阵 C —— ★ 参数矩阵给方向
        │  Dx × C → 圈级优化（逐弯类别权衡）→ 整体性收口（配平 / 预算）
        ▼
   神经网络模拟优化 —— ★ 由神经网络不断模拟优化
        │  遥测锚定仿真训练的「调教性能 NN」在坐标上升循环里
        │  对候选调校逐一预测圈速，保留更优且不牺牲你反馈需求的解
        ▼
   建议报告：20 项参数（含悬挂几何/防倾杆/离地/胎压）+ 依据链 + 模拟轨迹
```

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **遥测解析** | F1 2025/2026 UDP 协议；核心包逐字段解析，非支持包**无损留存**（可从 `.f1rec` 重放补解析） |
| **车手反馈** | 中英自然语言问题 → 弯道级 / 扇区级 / 整体级自动识别精度 |
| **遥测自动发现** | 你没反馈的问题也会被发现：逐弯通过时间残差、按弯路肩冲击、锁死、刮底、胎温/胎压异常、损伤 |
| **整体性调教** | 20 项参数一次给全（含**悬挂几何 / 防倾杆 / 离地间隙 / 四轮胎压**），并给每项的联动依据与代价权衡 |
| **神经网络模拟优化** | 遥测锚定仿真训练的性能模型（MAE 53ms / R² 0.996）在优化循环里做模拟精修，报告给出迭代轨迹与预计提升 |
| **24 条赛道** | 含 Madrid 新赛道；每条赛道的弯道数据来自 80k+ 真实 2026 逐弯遥测 |
| **录制与训练导出** | 一键录制（`.f1rec` 原始字节）→ 导出逐圈训练样本；**录制永久保留** |

### 实测数据（可复现）

| 指标 | 数值 |
|------|------|
| 调教性能模型 | 验证集 **MAE 53.3 ms / R² 0.9959**（零改动基线 2758 ms）；相邻候选差异方向正确率 **90%**（>30ms 显著移动 **95%**） |
| 端到端优化效果 | 6 赛道 × 3 场景：模型判定平均 **−367 ms**，解析仿真器独立复核 **−631 ms**（18/18 场景均提升） |
| 赛道签名 | 最优翼角：Monaco **+12 档**（最想加翼）↔ Spa **−6 档**（最想减翼），与 F1 调教共识一致 |
| 测试 | `pytest` **2114 项**（含深切片 / 端到端）；`ruff` 全绿 |

> 复现命令见「开发与测试」与 [`docs/Setup_Simulation_Optimizer.md`](docs/Setup_Simulation_Optimizer.md)；
> 验收报告：[`docs/Optimization_Verification.md`](docs/Optimization_Verification.md)。

---

## 快速开始

### 方式 A：从 GitHub 安装（推荐）

需要 **Python 3.11+**。

```bash
pip install "git+https://github.com/JXTTNN/F1-"
f1opt
```

启动后自动在浏览器打开面板（`http://127.0.0.1:8000`）。参数：

```bash
f1opt --help              # 查看全部参数
f1opt --port 8123         # 换端口
f1opt --no-browser        # 不自动开浏览器
f1opt --keep-telemetry    # 退出时保留派生遥测数据（见「数据与隐私」）
```

从源码安装（含开发依赖）：

```bash
git clone https://github.com/JXTTNN/F1-.git
cd F1-
pip install -e ".[dev]"
python -m setup_tuner.cli
```

> **装完即完整**：训练好的模型（调教性能 NN + 遥测代理）随包分发
> （`setup_tuner/resources/models/`），UI 与 24 条赛道数据同样在包内 ——
> `pip install` 之后不需要再训练、不需要额外下载，和本地开发环境能力一致。
>
> **为什么不需要 PyTorch**：神经网络是项目自带的**纯标准库** MLP
> （`setup_tuner/engine/pure_nn.py`，手写前向 / 反传 + Adam）。安装体积小、无需编译、
> Windows/Linux 行为一致；模型权重**随包分发**（`setup_tuner/resources/models/`），
> 装完即用 —— 不会出现"缺依赖 → 静默降级"。

### 方式 B：免安装便携包（Windows）

从 [Releases](https://github.com/JXTTNN/F1-/releases) 下载
[`F1OPT-portable.zip`](https://github.com/JXTTNN/F1-/releases/download/v1.6.11/F1OPT-portable.zip)（约 22 MB），
解压后双击「一键启动.bat」即可，无需安装 Python、无需任何配置。
>
> 便携包每次发布都会自动通过**打包产物测试**（6 阶段 / 100+ 检查）才会公开：
> 包含「模型确实随包内嵌并在跑（`model_type=nn`）」与「数据落在解压目录内」两项硬断言。

### 游戏内设置（接遥测）

F1 2026 → Settings → Telemetry：

| 项 | 值 |
|---|---|
| UDP Telemetry | `On` |
| UDP Format | `2026` |
| UDP Port | `20777` |
| UDP IP | `127.0.0.1`（本机） |

启动 F1OPT 后跑圈即可实时采集；停表自动出分析报告。

---

## 使用流程

1. **启动** `f1opt` → 打开面板（实时遥测 + 赛道图）
2. **选赛道**（或由遥测自动识别）
3. **导入当前调校**（可选：游戏内导出 Car Setup，面板一键导入）
4. **跑一圈**
5. **点赛道图上的问题弯**录入反馈（推头 / 甩尾 / 打滑 / 锁死 / 刮底 / 压路肩…），
   或**什么都不点** —— 遥测会自动发现问题
6. **点「生成建议」** → 得到 20 项调整 + 每项依据 + 模拟优化轨迹
7. 应用到游戏里，再跑一圈对比

> 反馈是"每圈一次"的消耗品：生成建议后本赛道反馈会被清空，避免与下一圈混淆
> （可用 `clear_feedback_after_suggest: false` 保留）。

---

## 神经网络模拟优化（模型真的在用）

「参数矩阵给方向」这条线早就有；**模型驱动**这条线在 2026-09-19 做了大改：

| 环节 | 实现 | 说明 |
|------|------|------|
| 数据 | `scripts/build_setup_sim_dataset.py` | **80,983 条真实 2026 逐弯遥测**做锚点（逐弯速度剖面 / 弯时 / 刹车占比 / 赛道中位圈速 / 直道占比）+ 文档化的准稳态车辆动力学 → 18,200 条「调教 → 圈速增量」样本 |
| 模型 | `scripts/train_setup_sim_nn.py` | 纯标准库 MLP（`setup_tuner/resources/models/setup_sim_nn.json`），验证 MAE 53 ms / R² 0.996 |
| 优化 | `setup_tuner/engine/sim_optimizer.py` | 坐标上升的**模拟精修**：逐参数试步 → NN 预测候选圈速 → 更优且"付得起"目标代价才接受 |
| 报告 | `report.holistic.simulation` | 模型描述 / 迭代轮数 / 采纳次数 / 模拟提升 / 逐步采纳轨迹 —— **面板上直接可见** |

**降级是诚实的**：模型不可用、赛道不在覆盖范围、或你没有任何反馈且遥测也没发现问题时，
自动走纯规则路径，并在报告里写明原因（不假装模型在跑）。

---

## 数据与隐私

**所有数据都在安装文件夹内**（`<安装文件夹>/data/`）—— 不跟随启动目录，
也不会散落到用户主目录或 `%APPDATA%`。安装文件夹指含 `setup_tuner/` 的那一层：
便携包 / 仓库开发就是解压/克隆出来的那个目录，`pip install` 则是虚拟环境的
`Lib/site-packages/`。

| 数据 | 位置（安装文件夹内） | 关闭应用时 |
|------|--------------------|-----------|
| **录制**（`.f1rec` + 逐圈样本） | `data/recordings/` | **永久保留**（你的原始数据） |
| 应用数据（反馈 / 建议 / 迭代历史） | `data/f1opt.db` | 保留 |
| 派生遥测数据（训练集 / 模拟包流） | `data/training/`、`data/sim_telemetry/` | **自动删除**（可随时重新生成） |
| 训练好的模型 | 随包分发（`setup_tuner/resources/models/`） | 保留 |

- 一切都在本机：不联网、不上传、无遥测回传。
- 想换数据位置（高级用法）：设 `F1OPT_DATA_DIR`；默认无需配置。
- 想保留派生数据（例如复用训练集）：`f1opt --keep-telemetry` 或 `F1OPT_KEEP_TELEMETRY=1`。
- 清理是**白名单**操作：只删上面列出的派生目录，**绝不触碰录制**，
  且拒绝在家目录 / 文件系统根下执行（见 `setup_tuner/telemetry/cleanup.py`）。

---

## 配置

环境变量（`F1OPT_` 前缀可选，两种写法都认）或项目根目录的 `.env`：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `F1OPT_UDP_HOST` | `0.0.0.0` | 遥测监听地址（绑所有网卡才能收广播包） |
| `F1OPT_UDP_PORT` | `20777` | 遥测监听端口 |
| `F1OPT_API_HOST` | `127.0.0.1` | 面板 / API 监听地址 |
| `F1OPT_API_PORT` | `8000` | 面板 / API 端口 |
| `F1OPT_DATA_DIR` | `<安装文件夹>/data` | 数据目录（录制 / 数据库 / 派生数据）；默认固定在安装文件夹内，无需设置 |
| `F1OPT_KEEP_TELEMETRY` | 未设置（关闭时清理） | 设为 `1` 保留派生遥测数据 |
| `F1OPT_LOG_LEVEL` | `INFO` | 日志级别 |

命令行参数优先级高于环境变量（如 `f1opt --port 8123`）。

---

## 项目结构

```
setup_tuner/            运行时包（pip 安装的就是它）
├── app.py              FastAPI 应用装配 + 生命周期（关闭时清理遥测）
├── cli.py              命令行入口（f1opt）
├── config.py           配置（环境变量 / .env）
├── api/                REST 端点 + WebSocket
├── db/                 SQLite（反馈 / 建议 / 迭代 / 调教快照）
├── domain/             领域模型：调教参数、24 赛道、弯道弧长表、症状
├── engine/             ★ 调校引擎
│   ├── engine.py         主流水线（Dx → C → 圈级优化 → 收口 → 模拟优化）
│   ├── coupling.py       耦合矩阵 C（诊断维度 → 参数）
│   ├── optimizer.py      圈级坐标上升（残差 + 代价）
│   ├── holistic.py       整体性收口（机械抓地参与度 / 配平 / 预算）
│   ├── setup_sim.py      调教性能 NN（纯标准库推理）
│   ├── sim_optimizer.py  ★ 神经网络模拟优化循环
│   └── pure_nn.py        自带的纯标准库 MLP（训练 + 推理）
├── feedback/           反馈录入与迭代闭环
├── report/             报告组装
├── resources/models/   ★ 随包分发的模型产物
├── telemetry/          UDP 监听 / 解析 / 聚合 / 录制 / 关闭清理
└── ui/                 面板（index.html + app.js + 赛道 SVG）

scripts/                训练、校验、验收、审计脚本
tests/                  2114 项测试（含 deep_slice / deep_integration）
docs/                   设计文档、变更日志、验证报告
assets/                 一键启动.bat 等
legacy/                 重构前的旧实现存档（只读参考，不参与运行与 lint）
```

---

## 开发与测试

```bash
pip install -e ".[dev]"

pytest -q                       # 全量测试（含深切片 / 端到端）
ruff check setup_tuner tests scripts
node --check setup_tuner/ui/app.js

# 数据隔离下的「CI 等价」自检（提交前推荐）
DATA_DIR=$(mktemp -d) pytest -q
```

模型与验收：

```bash
python scripts/build_setup_sim_dataset.py --calibrate-only   # 标定表 / 赛道签名
python scripts/retrain_all.py                                # 一键重训全部模型
python scripts/verify_optimization_telemetry.py              # 端到端效果验收（含真值复核）
python scripts/api_endpoint_test.py                          # API 契约 + 模型语义校验
```

### 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/Setup_Simulation_Optimizer.md`](docs/Setup_Simulation_Optimizer.md) | ★ 神经网络模拟优化：数据 / 模型 / 管线 / 降级契约 / 复算步骤 |
| [`docs/Optimization_Verification.md`](docs/Optimization_Verification.md) | 端到端效果验收报告（模型判定 + 解析仿真真值复核双列） |
| [`docs/Telemetry_Trained_Setup_Model.md`](docs/Telemetry_Trained_Setup_Model.md) | 遥测代理模型（逐弯重要度 / 残差诊断） |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 系统架构与模块职责 |
| [`docs/使用方法.md`](docs/使用方法.md) | 完整使用说明（面板 / API / 常见问题） |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | 变更日志（每次迭代都有可验证的提升） |

---

## 免责声明

本项目是**非官方的社区工具**，与 EA、Codemasters、Formula 1 及其关联公司无关，
未获其授权或背书；所有商标归各自所有者。建议基于公开遥测规范、公开调教工程常识
与本机实测数据生成，**仅供参考**；实际效果受游戏版本、车手风格与赛道条件影响。
