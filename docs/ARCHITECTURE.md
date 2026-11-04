# F1OPT 系统架构

> 本文件按**当前仓库代码**描述。历史版本里写的 `f1opt` 包、Parquet 落盘、
> scipy 差分进化、GP 贝叶斯优化、`observability/` 子包，都是早期废弃方案，
> 代码里一处都不存在 —— 别再据此理解系统。

## 1. 数据流

```
┌──────────────────────────────────────────────────────────────┐
│ EA F1 2026 游戏（UDP 20777，29B 报文头，60Hz）                │
└────────────────────────────┬─────────────────────────────────┘
                             │ 原始字节
                             ▼
   telemetry/packets.py     解析（纯 struct）
            │
            ▼
   telemetry/listener.py    后台线程收包（丢旧保新）
            │
            ├─────────────► telemetry/stream.py      最新帧缓存（供 WS 推送）
            │
            ├─────────────► telemetry/recorder.py    无损录制 .f1rec（可选 zstd）
            │
            ▼
   telemetry/lap_aggregator.py   逐帧 → 整圈统计量
            │
            ▼
   telemetry/style_extractor.py  逐帧 → 12 维车手风格
            │
            ▼
   domain/*                      20 参数调教 / 22 症状 / 24 赛道几何
            │
            ▼
   engine/engine.py              generate_suggestion（见第 3 节）
            │
            ▼
   report/builder.py             报告 JSON（参数明细 / 收益 / 代价 / 置信度）
            │
            ├─────────────► api/routes.py   REST（/api/v1，26 个端点）
            └─────────────► api/ws.py       WebSocket 实时推送
                             │
                             ▼
   ui/  单文件前端（index.html + app.js + style.css + 24 张赛道 SVG）
```

## 2. 模块地图

### domain —— 领域模型（无 IO、无网络、全静态）

| 模块 | 职责 |
|---|---|
| `setup.py` | 调教参数全集 **20 项 / 6 大类**（含 range / default / step / max_delta）+ `CarSetup` |
| `symptoms.py` | 症状枚举 **22 项 / 4 类**（entry / apex / exit / global） |
| `track.py` | **24 条** F1 2026 赛历静态数据 + `Corner` / `Track` |
| `_turn_data.py` / `_track_anchors.py` / `_track_arcs.py` | 逐弯元数据、赛道图锚点像素坐标、弯道圈内距离占比（自动生成） |
| `_track_official.py` | 弯道数据官方来源登记表 |
| `corner_locator.py` / `corner_groups.py` | 圈内进度 → 弯道编号的唯一实现；连续弯分组 |
| `style_coefficients.py` | 12 维车手风格 → 参数敏感度倍数（`gain_for_style`）+ 离散偏好动作（`style_trims`，±1 档、同参数只认领一次） |
| `track_coefficients.py` | 赛道类型 → 参数敏感度倍数 |

### engine —— 建议生成

| 模块 | 职责 |
|---|---|
| `engine.py` | 主流水线 `generate_suggestion`（7 步，见第 3 节）+ 风格定向 `apply_style_targeting` |
| `diagnostic.py` | 诊断向量 `Dx`（9 维 = 8 维能力需求 + 底盘离地） |
| `coupling.py` | 耦合矩阵 `C`（9 × 20） |
| `holistic.py` | 整体权衡层：改动预算 + 归一化幅度排序（`abs(Δ)/max_delta`） |
| `optimizer.py` | 确定性坐标上升（真实单位） |
| `sim_optimizer.py` | 模拟优化：坐标上升 + **三道上限**（单步 80ms / 累计 120ms / 最多 3 步） |
| `setup_sim.py` | 调教性能预测封装：**差分校准** `raw(候选) − raw(默认)` |
| `pure_nn.py` | 纯标准库 MLP（前向 + 训练，权重 JSON） |
| `lap_model.py` | 圈级 QSS 圈速模型（让建议「算」出来而不是查表） |
| `surrogate.py` | 遥测代理模型接入 |
| `telemetry_diagnosis.py` | 车手没反馈到的问题的自动诊断 |
| `confidence.py` | 置信度（证据充分度 + 反馈明确度） |
| `model_io.py` | 模型产物路径解析：工作目录 → 仓库根 → **随包资源** |

### telemetry / api / db / feedback / report

| 模块 | 职责 |
|---|---|
| `telemetry/packets.py` | F1 26 UDP 解析，`HEADER_FORMAT` 断言 **29 字节** |
| `telemetry/listener.py` | 标准库 socket 后台线程收包 |
| `telemetry/recorder.py` | 无损录制 + 可选 zstd 压缩 + 回放 |
| `telemetry/importer.py` / `simulator.py` | JSON 导入 / 模拟回放 |
| `telemetry/training_export.py` | `.f1rec` → 逐圈训练样本 JSONL |
| `api/routes.py` | REST，前缀 `/api/v1`，统一 `{code, message, data}` 信封 |
| `api/ws.py` | 单连接多事件 WebSocket 网关 |
| `api/envelope.py` | 信封工具 |
| `db/store.py` + `db/schema.sql` | SQLite 持久化（反馈 / 建议 / 迭代 / 录制元数据） |
| `feedback/service.py` / `iteration.py` | 反馈业务逻辑；迭代闭环与前后对比 |
| `report/builder.py` | 报告组装（design 2.7.7 格式） |

## 3. 建议生成链（`generate_suggestion`）

1. **症状 → `Dx`**：车手反馈（22 症状 × 强度）+ 遥测自动诊断 → 9 维诊断向量。
2. **`Dx × C` → 参数方向**：耦合矩阵给出 20 项参数的方向与相对幅度。
3. **整体权衡**（`holistic`）：按 `abs(Δ)/max_delta` 归一化排序，套改动预算
   （一次别让车手改一车），保证跨量纲可比。
4. **风格定向**（`apply_style_targeting`）：风格向量（≥3 圈）→ 离散 ±1 档动作，
   逐条用性能模型评代价，单条 ±30ms 可信区间、整层 25ms 预算；采纳的叠在已采纳
   结果上，拒绝的写进 `holistic.style_targeting.rejected`。无诊断输入时整层跳过。
5. **模拟优化**（`sim_refine`）：以第 4 步结果为起点做坐标上升；每步都要模型
   认可才采纳，收益按**可信上限**记账（单步 ≤80ms、累计 ≤120ms、最多 3 步），
   被截断的步骤在轨迹里标 `capped`。
6. **差分校准**（`setup_sim`）：`predict_delta_s = raw(候选) − raw(默认)`。默认点
   恒为 0，消除每赛道常数偏置；两套调教的差值不受偏置影响。
7. **组装报告**（`report/builder.py`）：参数明细、`gain_ms`（记账值）与
   `raw_gain_ms`（模型原始预测）、代价、置信度、赛道锚点来源，以及
   `holistic.simulation` / `holistic.style_targeting` 两段可核对的执行记录。

> 为什么收益要封顶：代理模型「差值方向可信、绝对量级不可信」。不封顶时
> 一版 24 站模型能报出中位 733ms、平均改 9.37 个参数的「建议」—— 物理上
> 不可能，等于让车手重写调教。详见 `docs/Setup_Simulation_Optimizer.md`。

## 4. 模型产物

| 文件（`setup_tuner/resources/models/`） | 说明 |
|---|---|
| `setup_sim_nn.json` | 调教性能 MLP：**53 维输入**（24 赛道 one-hot + 5 赛道特征 + 20 调教 + 4 工况），`[48,32]` 隐层，9,600 基础样本，24 站（遥测锚定 19 + 几何推定 5） |
| `telemetry_surrogate.json` | 遥测代理模型 |
| `setup_sim_eval.json` / `.md` | 模型离线评测报告（MAE / R² / 分片误差） |
| `usage_flow_eval.json` / `.md` | **使用流程**评测（走真实 `generate_suggestion` 路径） |
| `eval_rejected/` | 未过发布门禁的评测留档（线上仍用旧模型） |

模型随包分发（`pyproject.toml` 的 `package-data` + Nuitka `--include-data-dir`），
运行时由 `engine/model_io.py` 解析，找不到就**显式降级**并在报告 `model_type`
里如实标注，不会假装在用 NN。

## 5. 运行形态

| 形态 | 启动 | 数据落点 |
|---|---|---|
| 源码 | `python -m setup_tuner.cli` / `uvicorn setup_tuner.app:create_app --factory` | `<仓库根>/data` |
| pip 安装 | `f1opt` | `<安装根>/data`（`F1OPT_DATA_DIR` 可覆盖） |
| Nuitka 便携包 | 双击 `F1OPT.exe` | 安装文件夹内 `data/` |

## 6. Tech Stack

> 这张表以前写的是「设计时打算用什么」（PyTorch / SciPy / scikit-learn /
> PyArrow / OpenTelemetry），与实际代码不符 —— 仓库里没有任何一处 import
> 它们。下表按**当前代码真实依赖**重写，可作为 `pip install` 清单的对照。

| Layer | Technology |
|---|---|
| Language | Python 3.11+（运行时只用标准库 + FastAPI 全家桶） |
| ML | 纯标准库 MLP（`engine/pure_nn.py`，权重存 JSON，运行时零 AI 依赖） |
| Optimization | 规则层（耦合矩阵 + 症状权重）+ 坐标上升（`engine/sim_optimizer.py`）+ 差分校准 |
| API | FastAPI + Uvicorn + WebSocket（`/api/v1/ws` 实时遥测） |
| Data | JSON / JSONL 数据集 + SQLite（`db/schema.sql`）；录制可选 zstandard 压缩 |
| Observability | 标准库 `logging` + 自建审计日志（append-only JSONL） |
| Testing | pytest（+ pytest-asyncio / pytest-cov / pytest-timeout），2,100+ 项 |
| Code Quality | ruff（CI 门禁：`ruff check setup_tuner/ tests/`） |
| CI/CD | GitHub Actions（ubuntu 测试 + windows-latest 打包） |
| Packaging | Nuitka（onefile 便携包 + `F1OPT-portable.zip`） |

## 7. 相关文档

- `docs/Telemetry_Trained_Setup_Model.md` —— 模型怎么训、可信边界在哪
- `docs/Setup_Simulation_Optimizer.md` —— 模拟优化器与三道上限
- `docs/Optimization_Verification.md` —— 历史验证快照（方法学参考）
- `docs/CHANGELOG.md` —— 逐版本变更与验证数据
