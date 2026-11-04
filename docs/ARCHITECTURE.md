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
6. **差分校准 + 真实校准**（`setup_sim`）：先做差分
   `sim_delta = raw(候选) − raw(默认)`（默认点恒为 0，消除每赛道常数偏置）；
   若加载了真实圈速校准，再按该站权重融合
   `(1−w)·sim_delta + w·(coef·n + sim_coef·sim_delta)`。真实层是**堆叠**的：
   `sim_coef≈1、其余≈0` 就等价于完全信任仿真，因此它只能修正、不会更差。
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
| `real_lap_calibration.json` | **真实圈速校准层**（v1.6.16）：来自 F1Laps 公开调教库的「20 项调教 + 实测圈速」配对，站内变系数岭回归（仿真预测 + 调教主效应 + 赛道特征交互），逐站 20 个等效系数 + 1 个仿真系数 + 融合权重；权重 0 = 该站退回纯仿真 |
| `telemetry_surrogate.json` | 遥测代理模型 |
| `setup_sim_eval.json` / `.md` | 模型离线评测报告（MAE / R² / 分片误差） |
| `usage_flow_eval.json` / `.md` | **使用流程**评测（走真实 `generate_suggestion` 路径） |
| `eval_rejected/` | 未过发布门禁的评测留档（线上仍用旧模型） |

### 4.1 真实圈速校准（外部真值层）

`setup_sim_nn.json` 的标签全部来自项目自写的准静态仿真器，
`setup_sim_eval.json` 里的 `real_setup_laps: 0` 就写着这件事。v1.6.16 引入
`scripts/train_real_lap_model.py`，用**外部真实配对**校正仿真：

| 环节 | 做法 | 为什么 |
|---|---|---|
| 数据 | `scripts/fetch_setup_laps.py`：F1Laps 公开库，3 个游戏年版 × 干/湿 = 6 个切面，24 站，库内 2,538 条（入模 1,667 条 / 3,188 对）；robots.txt 明确 `ai-train=yes`，只走公开 HTML、限速 | 2026 官方遥测只有速度/油门/刹车，不含调教，圈级头长期不可用 |
| 版本隔离 | 某年版若过半样本的参数超出当前滑杆区间，**整版跳过**（记 `version_incompatible:<年>`） | F1 24 的后离地间隙是 40~83、后倾角 −2.2~−1.2，与当前区间不同；裁剪或按比例映射会给该年度注入系统偏移（同一个 n 值对应的真实角度变了） |
| 去混淆 | 交替投影剔除 `赛道 / 车队 / 会话 / 年份 / 切面 / TC` 主效应 | 车队性能差 0.3~1.5 s，比调教差异还大；不去掉就学到「谁的车快」 |
| 评估口径 | 指标一律用**同队同会话配对**（同站同队同会话内两套调教谁更快） | 站内混比把 1~2 s 车手差距混进来，会把 58.6% 的有效模型误判成 50.3% 的无效模型 |
| 结构 | 变系数岭回归：`α·仿真 + Σβ_k·n_k + Σγ_{k,j}·n_k·track_feature_j` | 每站仅 32~100 条，逐站硬拟合 20 个参数只能学到噪声（实测留出集 45.8%，低于随机） |
| 选参 | 按**作者分组**的 5 折交叉验证；λ 网格 `1e0…1e7`，在最优平台（±0.002）内取**更小**值 | 同一作者多套调教跨折会泄漏；λ 取平台最小值而非边界最优，数值条件更好、不依赖「最优恰好落在边缘」 |
| 尺度校准 | 逐站打分**等比**缩放到标准差 0.40 s | 大 λ 解出的系数只有 0.003 s 量级，排序全对但与仿真融合等于静音；等比缩放不改变任何排序 |
| 逐站权重 | 按该站留出准确率派生（`≥55%` 满权重 / `50~55%` 减半 / `≤50%` 归零） | 规则是**事先固定的单调映射**，不看这一站挑中了哪套调教，所以不算在留出集上调参；hungaroring 留出 43.4%（114 对）显著低于随机，继续融合等于用噪声改车。边界含不含等号必须与引擎对齐：`SetupSimModel` 把 `≤50%` 当「没有可用精度」不对外显示，训练侧若仍给半权重就会出现「有权重、没准确率」（shanghai 恰好 0.500 曾踩中） |
| 门禁 | 真实层打不过仿真时 `blend_weight=0`，行为与旧版完全一致 | 宁可不用，也不能用噪声改车 |

**零泄漏验收**（`scripts/eval_real_lap_model.py`，与训练分开跑）：

| 检验 | 结果 | 说明 |
|---|---|---|
| 同队同会话配对（5 折） | 58.6%（仿真 48.1%，3,188 对） | 主指标 |
| **留一赛道** | **59.1%**（95% 区间 57.4%–60.8%，18/24 站 >55%） | 该站完全不参与训练，只能靠赛道特征外推 |
| 跨年 2025 干→2026 干 | 57.4%（626 对；反向 2026→2025 仅 50.9%） | 车与规则都变了；**单向**外推成立，反向不成立 |
| 跨切面 2025 干→湿 | 55.5%（482 对） | 干湿共用同一套参数响应 |
| 跨切面 2026 干→湿 | 54.4%（240 对） | 同一结论在新一年度复现 |

留一赛道与 K 折几乎重合 → K 折没有水分。

**已知边界**：配对预测对样本顺序是对称的（互换 i/j 时 `d_actual` 与 `d_pred` 同时取反），
所以指标不受抓取顺序影响；但并列必须记半分（AUC 的 tie 处理）—— 早期版本把并列判错，
让一个恒定预测的退化模型刷到最高分，这个坑已在 `matched_pair_counts` 里写死注释。

报告字段：`holistic.simulation.real_calibration`（`weight` / `accuracy` / `tracks` / `samples`）。
`weight` 是「听谁的」，`accuracy` 是「有多准」，面板显示后者（「真实校准 NN% 准」），
权重 0 时不显示 —— 把 0.9 写成「真实校准 90%」会被读成准确率。

模型随包分发（`pyproject.toml` 的 `package-data` + Nuitka `--include-data-dir`），
运行时由 `engine/model_io.py` 解析，找不到就**显式降级**并在报告 `model_type`
里如实标注，不会假装在用 NN。

## 5. 运行形态

| 形态 | 启动 | 数据落点 |
|---|---|---|
| 源码 | `python -m setup_tuner.cli` / `uvicorn setup_tuner.app:create_app --factory` | `<仓库根>/data` |
| pip 安装 | `f1opt` | `<安装根>/data`（`F1OPT_DATA_DIR` 可覆盖） |
| Nuitka 便携包 | 双击 `F1OPT.exe` 或 `一键启动.bat` | 安装文件夹内 `data/` |

便携包是**控制台程序**（`--windows-console-mode=force`），不是静默的 GUI 程序：
双击后那个窗口就是运行日志，关掉即停止。1.6.17 及以前用 `disable`，双击后既无
窗口也无提示，首次解压的 3–10 秒里用户只看到「没反应」，出错更是一个字都没有。
日志同时写 `<数据目录>/logs/f1opt.log`（2 MB × 3 轮转），启动器失败时 tail 它。

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
