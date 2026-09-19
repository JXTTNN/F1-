# 遥测训练的调教优化模型（2026 赛季）

> 本文记录"用真实 2026 遥测训练模型并接入本地 F1OPT 调教优化"的完整链路、
> 实测指标与验证方式。所有数字均来自本仓库实际运行输出，可复现。

## 1. 为什么之前是坏的

| 问题 | 事实 |
|---|---|
| "神经网络参与调教"从未发生 | `engine/nn_model.py` 依赖 PyTorch；venv 中 **没有 torch / numpy / sklearn**，且 PyPI 装不上（实测 5 分钟超时）→ `available` 恒为 `False`，`model_type="hybrid"` **静默降级为纯规则引擎** |
| 上一轮的"模型"是空壳 | `train_setup_model_advanced.py` 只写了 `{version, samples_count}`，没有权重；`telemetry_implicit_feedback.py` 是 `{"placeholder": true}` |
| 悬挂几何在最终建议里消失 | 优化器**单调拆掉所有改动**（见 §5） |

清零这些假产物后重建整条链路。

## 2. 数据

| 来源 | 规模 | 用途 |
|---|---|---|
| TracingInsights 2026 逐点遥测（`fetch_official_2026.py`，2026 年检） | 13 赛道 / 80,983 个**逐弯样本** | 弯速代理训练 |
| 本地游戏 UDP 跑圈（`data/recordings/*_laps.jsonl`） | 14 圈（含 22 项 setup + 11 维风格） | 整圈配速代理训练 |

逐弯样本由 `scripts/build_corner_dataset.py` 用官方弯道锚点的弧长占比把每一切成 N 段，
目标量 = **该弯用时占整圈时间的比例**（无量纲，消除"慢圈全线变慢"的噪声）。

## 3. 模型（`data/models/telemetry_surrogate.json`）

| 头 | 学习器 | 留出集指标 |
|---|---|---|
| 弯速代理（遥测训练） | 岭回归（闭式解） | **MAE 0.0062（占圈速比）/ R² 0.8909** |
| 弯速代理 | 纯标准库 MLP（`engine/pure_nn.py`） | MAE 0.0173 / R² 0.5546（本轮落败，引擎按 MAE 择优取岭回归） |
| 整圈配速（含 setup + 风格） | 岭回归 + 留一验证 | LOO MAE 4831 ms vs 同赛道均值基线 4810 ms → **skill = 0.0（不参与）** |

关键结论：

- **赛道×弯号 one-hot 是决定性特征**：只给"赛道"身份 R²≈0.11，给到弯级身份后 R²=0.907
  （实测 Monaco T1 期望弯时由 0.28 s 修正为 2.16 s）。
- 整圈配速头在 14 个样本上**学不到东西就不参与**（skill=0），
  不编造"高精度模型"去覆盖物理方向（C 矩阵）。
- MLP 落败是如实上报的结果，不是隐藏；`--force-model mlp` 可强制切换。

## 4. 模型怎么被用（这是"真的生效"的部分）

1. **逐弯重要度**：`lap_model.corner_evaluations` 优先采用模型给出的"该弯占整圈时间比例"，
   取代旧的 `120 / 参考速度` 启发式。实测 **Monza 全部 11 个弯的权重都与启发式不同**。
   重要度直接决定优化器"愿意为这个弯付出多少代价"。
2. **自动发现车手未反馈的问题**：`engine/telemetry_diagnosis.py` 把遥测信号翻译成
   与车手反馈**同构**的症状，并可选地使用弯速模型残差（实测用时 vs 模型期望）。
3. **C 给方向、模型给评价**：耦合矩阵 C 负责初始方向（`initial_delta`），
   优化器在该方向附近搜索；搜索用的目标函数权重来自遥测训练的模型。

### 4.1 接管了哪些"引擎规则没覆盖"的盲区

逐个 grep 确认 `engine.py` 的 17 条聚合规则**完全没覆盖**以下信号，因此它们必须由
隐式诊断接管（否则车手没反馈时永远进不了优化）：

| 遥测信号 | 症状 | 依据 |
|---|---|---|
| `m_antiLockBrakes` | `lockup` | 防抱死介入 = 锁死倾向 |
| `m_tyresWear` | `tyre_wear` / 前后胎耗失衡 | 无任何规则消费胎耗 |
| `m_tractionControl` | `exit_wheelspin` | 牵引控制介入 = 出弯打滑 |
| `m_floorDamage` | `high_speed_instability` / `bottoming` | 底板损伤 → 扩散器失效 |
| `m_frontLeftWingDamage` 等 | `understeer` | 前翼损伤 → 前轴失压（速度越高越明显） |
| `kerb_corners` | `kerb_instability`（带弯道归因） | 规则只有全圈判定、无弯道号 |
| 模型残差 | 按弯型判定 | 实测用时 vs 模型期望 |

已有规则覆盖的信号（胎温过高/不均、胎压、刹车过热、转向、直道速度、刮底）
**只进报告展示、不进反馈路径** —— 否则同一问题会被重复计入 Dx。

### 4.2 「问题 → 优化方案」映射

报告里 `holistic.implicit_plan` 给出每条遥测发现对应的参数改动（走
`症状 → Dx 维度 → 耦合矩阵 C → 参数`，与引擎同源）。没有对应改动的会显式标
`resolved: false`，不假装已解决。实测 13/13 条发现全部落地。

## 4.3 API 层的致命闸门（"没生效"的真正原因）

`POST /api/v1/suggest` 原本**要求必须先有车手反馈**，没有就 400 拒绝 ——
于是"遥测自动发现的问题"永远到不了用户面前。现已放宽：

- 有车手反馈 → 照常；
- **没有反馈但遥测诊断出具体问题 → 仍然出方案**，报告标 `telemetry_only: true`
  并在 `summary` 里说明；
- 既没有反馈也没有遥测发现 → 仍然 400（不会变成"什么都出方案"）。

## 4.4 反馈生命周期：生成后"消费"（跨圈不串味）

**问题**（2026-09-19 用户反馈）："前一圈的车手反馈在点击生成优化后调教要清除，
要不然杂着后一圈吗"。`/suggest` 从 SQLite 读**累积**反馈，跨圈时第 2 次生成
会同时受第 1 圈 F1 + 第 2 圈 F2 影响 —— 而 F1 针对的问题应已被上一轮调教
修正/替换，混入即是噪声。

**修复**：`POST /api/v1/suggest` 新增 `clear_feedback_after_suggest`（默认 `True`）：

- 生成**成功后**清除本赛道反馈，并如实返回"已清除 N 条"；
- **时序**：落库成功之后、WS 推送之前 —— WS 事件后前端刷新反馈列表不会读回旧数据；
- 生成/落库失败时不清除（异常提前抛出），反馈保留供重试；
- 清除失败非致命（建议已落库）：记日志 + 返回成功，不把已完成的生成打成 500；
- `False` 保留累积行为（连续重新生成、性能基准等场景）；
- 前端：生成成功后自动刷新反馈摘要与赛道图"已反馈"标记；WS 推送同样触发刷新。

**验证**（`tests/test_feedback_lifecycle.py`，7 条，含负向验证）：

- 默认路径：生成后反馈清空、无新反馈再次生成 400（不复用已消费输入）；
- 对照路径（`False`）：反馈保留、可连续重新生成；
- **跨圈纯净性**（核心断言）：两轮不同反馈时，第二轮 delta 与"全新环境仅含
  第二轮反馈"逐位一致 —— 第一轮反馈零贡献。把清除逻辑停掉后该用例必红；
- 清除只作用于目标赛道；空赛道清除返回 0（幂等）。

修复过程中还发现并修正了两处**假绿/失效检查**（与生命周期同一批排查）：

- `test_increasing_strength_increases_delta_magnitude` 用 `strength=5`（超出合法
  区间 [1,3]），`POST /feedback` 实际 422 从未入库 —— "strong >= weak" 比较的是
  同一份输入。现改为合法上限 3 并**断言 POST 状态码**，杜绝静默失败。
- `scripts/api_endpoint_test.py` 3 处过时断言（21 项参数 → 实际 20 项），以及
  `strength: 4` 非法载荷；`run_local_test.py`、`perf_*` 基准脚本的同类问题一并修正。
  修正后端点全检 **23 passed / 0 failed**。

## 4.5 优化质量

- **二级精修**：粗搜（网格 0.125）收敛后再用半格长（0.0625）重启一轮，
  只接受更优解。实测 Suzuka 圈级目标 0.005377 → 0.003924（**降 27%**），
  Spa / Silverstone 亦有改善；Monza / Monaco 已在格点最优、不变。
- **需求满足度**：报告 `holistic.optimization.satisfaction` 给出各弯道类别
  "诊断出的需求被覆盖了多少"（0..1）+ 优化前后对比。它与目标函数互补 ——
  目标函数含阻力/刮底/胎温代价项，只看它会漏掉"代价很低但留有缺口"的解。
  注意：由于规则引擎初值本身已按 `ΣC²` 归一化到接近满供，该指标常在 0.99~1.00，
  更适合用来发现"缺口"而不是比较优劣。

## 5. 悬挂几何为什么之前"没有"

`scripts/trace_setup_pipeline.py` 逐步追踪后发现：优化器在**单调拆掉所有改动**
（后翼 −0.62、前束角 0.05→0.04→0.03→0、后防倾杆 1.00→0.75→0.50→0.25→0），
Monza 最终只剩 3 项。原因：改动惩罚 `_K_EFFORT = 0.02` 的量级远大于残差项
（≈0.8·deficit²），"少改/不改"在数学上最划算。

**尝试把 `_K_EFFORT` 降到 0.002 → 打坏别处**（实测回归）：

- Monza 与 Monaco 收敛到**逐位相同**的解（赛道特性失效）；
- 湿地总调整幅度不变量被打破（`deep_integration::test_multi_packet_to_telemetry_summary` 变红）。

→ 已回滚 `0.02`，改用**定向规则**修复：`holistic_coherence` 新增第 9 条
「机械抓地参与度」——牵引型赛道（`traction_index ≥ 0.35`）且存在机械抓地需求、
却没有任何悬挂/几何/防倾杆改动时，从规则层原始建议里恢复，且**预算中性**
（从非机械项等量回收，保证总改动幅度不增长）。

## 6. 验证

```
python scripts/verify_api_telemetry_only.py  # 真实 API 链路五项验证（全部通过）★
python scripts/verify_model_in_engine.py     # 引擎层四项验证（全部通过）
python scripts/api_endpoint_test.py          # 端点全检 23 passed / 0 failed（2026-09-19 修复后）
python scripts/smoke_boot_server.py          # 真实 HTTP 启动 + 模型加载冒烟
python scripts/trace_setup_pipeline.py       # 逐阶段追踪参数是被谁改掉的
python scripts/validate_track_data.py        # 赛道图/锚点/弧长（24 赛道，0 错误）
pytest -q                                    # 2110 passed / 7 skipped（隔离 DATA_DIR）
ruff check setup_tuner tests scripts         # All checks passed
```

`verify_api_telemetry_only.py`（**用真实 app + 真实 UDP 包处理链 + 真实 HTTP 路由**）
的实测结论：

1. 零车手反馈 + 有遥测发现 → **HTTP 200**（原来 400）；
2. 报告 `telemetry_only = true`、`telemetry_discovered = 13`；
3. 自动发现 13 条问题，含 T1/T4/T5 重度压路肩、底板触地、底板损伤 45%、
   前胎比后胎热 18 °C、刹车均温 559 °C、牵引控制介入、防抱死介入；
4. `implicit_plan` **13/13 条已落地**到具体参数改动；
5. 建议共 12 项非零改动，其中**悬挂/几何 6 项**（前/后悬挂 +2、前/后防倾杆 +1、
   前/后离地 +2）；
6. 回归：零反馈 + 零遥测 → 仍然 HTTP 400。

`smoke_boot_server.py`：真实 uvicorn 启动 + `/api/v1/health` 200，
模型 `data/models/telemetry_surrogate.json` 已被引擎加载（R² 0.8909）。

## 7. 重训

```
python scripts/retrain_all.py                # 逐弯样本 → 特征融合 → 代理模型
```

引擎（`lap_model` / `generate_suggestion`）在启动时读取
`data/models/telemetry_surrogate.json`，跑完即生效，无需改代码；
模型文件缺失或损坏时**中性降级**（不抛错，回到启发式 + 纯物理规则）。

## 8. 官方口径核对

对照 EA F1 UDP **Packet 5 CarSetups** 字段表与 F1 25（2026 规则）实际调教值：

- 本项目 20 项参数与包内字段一一对应；
- 取值范围与社区实测一致（前翼 0–50、外倾 −3.5/−2.0、束角 0–0.2 / 0.1–0.35、
  悬挂 1–41、防倾杆 1–21、前后离地 15–35 / 40–60、胎压 22.5–29.5 / 20.5–26.5）；
- `m_engineBraking` 确实存在于包内，但车库不可调 → 不建模（既有决策正确）。

## 9. 已知限制

- 整圈配速头样本仅 14 圈 → skill=0，暂不参与；需要真实跑圈反馈持续积累
  （约 100 圈/赛道可进入可用区间）。
- 弯速代理覆盖 13/24 赛道；新增赛道后重跑 `retrain_all.py` 即扩展。
- 逐弯**实际用时**（`corner_times`）目前只有外部遥测有；本地跑圈要启用模型残差诊断，
  需要把逐点遥测回放成逐弯用时后传进 `generate_suggestion(corner_times=...)`。
