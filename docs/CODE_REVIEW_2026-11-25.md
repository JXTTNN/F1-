# F1OPT 全量代码审查报告

- 审查对象：`JXTTNN/F1-` @ `main`（2026-11-23 最后一次推送，v1.6.20）
- 审查日期：2026-11-25
- 审查方式：全仓静态扫描 + 依赖安装后实跑 lint/测试/覆盖率 + 运行时代码深读 + 关键路径实测

---

## 0. 结论摘要

| 维度 | 结果 |
|------|------|
| ruff（仓库自带配置） | **0 告警**（exit 0） |
| pytest | **2253 passed, 3 skipped, 0 failed**，89 s |
| 覆盖率（`setup_tuner`） | **85.98%**（6654 statements / 759 miss） |
| 建议生成延迟（实测，suzuka + 3 条反馈） | rule **46.8 ms** / nn **106.5 ms** |
| 安全基线 | 无 `eval/exec/pickle/shell=True`；SQL 全参数化；XSS 统一 `esc()`；`session_id` 有穿越校验 |

**必须优先处理 2 个 P1 问题**：一是反馈路径与症状路径的 Dx 量纲差 3 倍，直接影响所有
`/suggest` 输出强度与置信度；二是「停止回放」接口是空操作，点了不会停。

总体判断：这是一个成熟度明显高于个人项目的工程——分层清晰（domain / engine /
telemetry / api）、模型降级链路完整、隐私口径写进了 `.gitignore` 和清理器、CI 覆盖
lint/pytest/便携包。问题集中在**两套口径并存的过渡期代码**（新反馈链路 vs 旧症状链路）
和**本地文件/线程边界**上，而不是主算法。

---

## 1. 审查范围与边界

| 范围 | 规模 | 审查深度 |
|------|------|----------|
| 全仓文件 | 642 个文件 / 约 19.5 万行 | 静态模式扫描（安全、异常、死代码、断言、SQL、前端转义等 19 类规则） |
| `setup_tuner/`（运行时） | 约 1.9 万行 / 6654 可执行语句 | engine 逐文件深读；api / db / telemetry 关键路径深读；ui 抽样 |
| `tests/` | 2.2 万行 | 实跑 + 覆盖率热点，不逐条读 |
| `scripts/` | 2.0 万行 | 数据管道脚本按需读；新增脚本按仓库规范实现并过 ruff |
| `legacy/` | 约 9.4 万行（占仓库近一半） | **未逐行审**（README 声明只读归档、不参与 lint/运行）；仅确认无运行时引用 |

> 说明：19.5 万行不可能逐行人工阅读。本报告的结论来自「全量自动扫描 + 核心模块全读 +
> 关键路径实测」，覆盖率数字给出了未读部分的客观边界。

---

## 2. 质量基线（可复现）

```
ruff check . --statistics          -> All checks passed! (exit 0)
pytest -q                          -> 2253 passed, 3 skipped, 17 warnings in 89.16s
pytest --cov=setup_tuner           -> TOTAL 85.98% (6654 stmts, 759 miss)
```

覆盖率短板（按模块，来自 `--cov-report=term`）：

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| `telemetry/packets.py` | **53.56%** | 大量分支是 2026 新包字段（`CarTelemetry2`、`MotionEx` 等）解析，单测未覆盖；建议用真实 `.f1rec` 样本做参数化回放测试 |
| `engine/surrogate.py` | 78.31% | 空模型/缺赛道分支未覆盖 |
| `engine/telemetry_diagnosis.py` | 79.14% | 遥测症状推断的阈值分支缺口大 |
| `telemetry/lap_aggregator.py` | 84.58% | 路肩归因、损伤极值、历史圈过滤均有未覆盖分支 |
| `telemetry/recorder.py` | 84.36% | 断线/停录竞态路径未覆盖（见 P2-4.4） |
| 其余 engine 模块 | 92%–99% | 良好 |

值得肯定的工程细节：

- **无随机性**：优化器是确定性搜索（`optimize_setup` 不用 `random`），同输入同输出，可回归。
- **降级设计**：模型缺失/版本不符时 `_FAILED_MODEL` 让 `available=False`，接口不 500 而是回退 rule。
- **单位统一**：domain 用「档位 / 0.1 bar / mm」等游戏单位，NN 内部归一化，边界清楚。
- **隐私**：`cleanup.py` 默认删派生遥测、录制永久保留；`.gitignore` 明确个人遥测不入库。

---

## 3. P1 问题（建议本迭代内修）

### 3.1 反馈路径与症状路径的 Dx 量纲相差约 3 倍

**位置**：`setup_tuner/engine/holistic.py:387`（`strength / 3.0`）、
`setup_tuner/engine/engine.py:1209-1214`（两条路径二选一）

**现象**：`generate_suggestion` 在 `effective_feedbacks` 非空时用
`class_weighted_dx()`，否则用 `compute_dx()`。前者把强度归一化除以 3，后者直接
`coef * strength`。同一组症状、同一强度，两条路径产出差 3 倍以上。

**实测**（suzuka，三条反馈：entry understeer 3 / exit oversteer 2 / global tyre_overheat 2）：

| 路径 | \|Dx\| 合计 | 置信度 | 非零调教动作 |
|------|------------|--------|--------------|
| feedback（生产 `/suggest` 走这条） | 2.244 | **low** | 7 项，最大 1 档 |
| symptom（CLI / 单测 / 直接调引擎） | 7.600 | **high** | 11 项，含前翼 3 档、off-throttle diff 10 |

比值 3.39×。影响：

1. 线上用户拿到的建议系统性偏弱（翼片 1 档 vs 3 档），且更容易被判「低置信度」；
2. 优化器 `objective` 的 `needs_by_class` 也小 3 倍，NN 二次优化的改进空间被压缩，
   `sim_optimizer` 更容易判定「无改进」而回退初始值；
3. 两套入口的行为不一致，调参时无法用 CLI 复现线上结果。

**建议**：确定唯一口径。若以生产路径（归一化 0–1）为准，则 `compute_dx` 同样除以 3；
若以 `coef × strength` 为准，则删掉 `class_weighted_dx` 的 `/3.0`。无论选哪个，补一条
回归测试：同输入下两条路径的 `dx` 逐项相等。

---

### 3.2 「停止回放」是空操作，回放线程无法中断

**位置**：`setup_tuner/api/routes.py:1650`（只置位 `replay_stop_requested`）、
`setup_tuner/telemetry/recorder.py:704-736`（`ReplayReader.replay()` 无停止检查）、
`routes.py:1567-1600`（`_replay_target` 也不检查该标志）

**现象**：`POST /telemetry/replay/stop` 只写了一个状态位，整个回放链路没有任何地方读取它。
`replay()` 按 `time.sleep(1/speed)` 一路跑到文件结尾。1× 回放 45 分钟的会话就要占线程
45 分钟；期间 `POST /telemetry/replay` 的「已有回放正在进行」守卫会拒绝新任务，用户只能
重启应用。

**建议**：`ReplayReader.replay(..., stop_event: threading.Event)`，循环内每包检查；
`_build_replay_target` 把 `app.state.replay_stop_requested` 转成 Event 传入；`stop` 接口
同时 `join(timeout)` 并复位标志。补一条「启动→停止→2 秒内退出」的集成测试。

---

## 4. P2 问题（下一迭代）

### 4.1 `/telemetry/import-file` 可读任意本地文件

`routes.py:1794-1828` 直接把请求体里的绝对路径交给 `Path.read_text()`，未做目录白名单、
扩展名、大小上限。虽然 API 默认只绑 `127.0.0.1` 且无鉴权，但本机任意进程（含浏览器里的
恶意页面在无预检的简单请求场景）都可能把它当「本地文件读取器」用。

建议：限定 `data/` 或用户显式选择的目录；扩展名白名单 `.jsonl/.json`；读前 `stat().st_size`
限 50 MB；对目录导入限制递归深度与条目数。

### 4.2 `_invoke_generate_suggestion` 用 `except TypeError` 做版本兼容

`routes.py:856-880`：先按新签名调用，捕获 **任意 TypeError** 后用旧签名重跑。问题在于
引擎内部真实的 `TypeError`（例如 `None` 参与算术）会被当成「旧版本不支持新参数」，静默降级
成纯规则建议，既掩盖 bug 又重复计算一次。

建议：用 `inspect.signature(generate_suggestion).parameters` 判断是否支持 `feedbacks`，
不要用异常做特性探测；或显式传 `version` 标记。

### 4.3 跑一次 `pytest` 会删掉本地 `data/training/`

**实测**：本次构建了 `data/training/corner_dataset.jsonl`（4.5 MB）后跑全量测试（含
`--cov`），测试结束后该目录**整体消失**。

根因：`cleanup.py:43` 的 `DELETE_DIRS` 含 `"training"`，应用 `lifespan` 在启动与关闭时都会执行
`cleanup_telemetry(Config.data_dir)`（`app.py:391`、`app.py:417`）；部分测试用默认 `./data`
起 app，于是清掉了真实派生数据。对「2253 项测试」作为卖点的仓库，这是很容易踩的坑：用户跑完
测试，训练数据没了。

建议：测试统一注入 `F1OPT_DATA_DIR` 到 `tmp_path`（或 conftest 里全局 monkeypatch）；
`cleanup_telemetry` 在 `PYTEST_CURRENT_TEST` 环境下默认 dry-run；README 里明确写
`F1OPT_KEEP_TELEMETRY=1`。

### 4.4 `recorder.stop()` 与工作线程的关闭竞态

`recorder.py:232-258`：`worker.join(timeout)` 超时后仍继续 `close()` 文件与 DB 连接，而
worker 线程可能还在写。慢盘 + 高包速率（60 Hz × 多包）时可复现「写已关闭文件」的异常，
表现为该段录制静默丢包（只有一条 warning）。

建议：超时后先设 `_stopping`，由 worker 自己 flush/close；或改为「投递哨兵 + 无限 join」
（写入线程不应被强杀），并把丢包计数写进录制元数据。

### 4.5 「效果闭环」字段是死接口

`db/store.py:526` 的 `update_iteration_outcome()` 仅被测试调用，`grep` 全仓无生产
调用点。这意味着 `iteration` 表的 `outcome_*` 五列永远为空，「建议→实测圈速→验证」的
闭环在产品里没有落地。要么接进 `/iteration/{id}/outcome`，要么删列并在文档里去掉该承诺。

### 4.6 服务版本号不一致

`app.py:595` 硬编码 `version="0.1.0"`，而包版本是 `1.6.20`（README/CHANGELOG 一致）。
`/docs`、`/openapi.json` 展示的版本是错的。建议 `create_app` 用
`from . import __version__`。

### 4.7 异常详情直接回给客户端

`app.py:504`：`"message": f"服务内部错误：{exc}"`；`routes.py` 多处 `建议生成失败：{e}`。
本地应用危害有限，但会把内部路径、SQL 片段、第三方堆栈暴露到 UI/日志复制场景。
建议：对外返回固定文案 + `trace_id`，详情只进日志。

### 4.8 反馈强度迁移会留下违反 CHECK 的旧行

`schema.sql:45` 约束 `strength BETWEEN 1 AND 3`；`store.py:137-146` 的迁移只处理
`strength >= 4`（→3）和 `strength >= 2`（→2），**strength = 0 原样保留**（注释称「0 保持
0」）。SQLite 不会回溯校验 CHECK，于是库里可长期存在 0 强度行，聚合时权重为 0，语义不明。
建议：把 0 归一到 1（或删除该行）并加一条迁移测试。

### 4.9 `pyproject.toml` 的 license 写法已弃用

`license = { text = "Proprietary" }` 是 TOML 表写法，setuptools 在本次安装时已告警
（要求 2027-02 前改为 SPDX 字符串 `license = "LicenseRef-Proprietary"`）。不改会在下一次
构建工具升级时直接失败。

### 4.10 CI 工作流冗余

`.github/workflows/` 下 13 个 workflow，`ci.yml`、`full-ci.yml`、`cloud-audit.yml`、
`deep-slice.yml`、`audit-*.yml` 都包含 ruff + pytest，push 时重复跑多遍。建议收敛为
「1 条主流水线（ruff + pytest + 打包冒烟）+ 2 条手动/定时（审计、深度回放）」。

---

## 5. P3 问题（随手清理）

| # | 位置 | 问题 |
|---|------|------|
| 1 | `engine/pure_nn.py:214-216` | 死代码：`m_w = [[...] for _ in range(1)]` 建完立即被下一行正确结构覆盖 |
| 2 | `engine/setup_sim.py:584-585` | `_FAILED_MODEL.real_sim_coef = {}` 连写两遍 |
| 3 | `engine/engine.py:_align_to_step`、`lap_model.units_to_delta`、`holistic._snap_to_step` | 用 `round()` 银行家舍入：step=1 时 raw=2.5 变 2（「半步」消失）。建议 `floor(x+0.5)` 或 `Decimal` |
| 4 | `config.py:202` | `.env` 只从 `Path.cwd()` 读；便携包从桌面快捷方式启动（cwd=系统目录）时配置不生效，与「数据目录按安装根」不一致 |
| 5 | `config.py:int(_get("UDP_PORT"))` | 填错直接 `ValueError` 崩在启动早期，报错不指向 `.env`；建议 `try/except` + 友好提示 + 端口范围校验 |
| 6 | `routes.py:1290` | 访问 `listener._host/_port` 私有属性；应在 Listener 上暴露 `host/port` 只读属性 |
| 7 | `routes.py:368-383`、`525-552` | `_TRACKS_LIST_CACHE` / `_SETUP_FIELDS_CACHE` 全局可变缓存无失效机制；赛道数据热更新后不刷新（当前是静态数据，可接受，但需注释约束） |
| 8 | `engine/engine.py:668` | `_compute_param_raw_delta(..., None)` 传 `None` 让 `style_gain` 形参实际闲置，风格在下一段另算；建议删形参或合并 |
| 9 | `engine/holistic.py:325` | `_stage_of` 用 `next(iter(table))` 依赖 dict 插入顺序，与 `diagnostic._resolve_stage` 的 `SYMPTOM_DEFAULT_STAGE` 是两份逻辑；表顺序一改就分叉 |
| 10 | `engine/engine.py:_is_wet_weather` | `telemetry.get("weather") or telemetry.get("m_weather")`：`m_weather=0`（晴天）被 `or` 吞掉，靠默认 `False` 兜底，语义脆弱 |
| 11 | `api/routes.py:143,164` | 描述写「12 症状标识之一」，实际 `SYMPTOM_STAGE_TO_DX` 有 **22** 个症状 |
| 12 | `ui/app.js:687` | `tmp.innerHTML = svgText` 信任服务端静态 SVG；若未来赛道文件可被用户替换，会变成存储型 XSS。建议用 `DOMParser` 只取 `<path>` |
| 13 | 仓库根 | `legacy/` 9.4 万行约占仓库一半体积；建议拆到独立归档仓库或 Release 附件，避免 clone/搜索污染 |
| 14 | `engine/coupling.py:validate_matrix()` | 模块导入时用 `assert` 做自检；`python -O` 下失效（内部不变量，可接受，但建议改为显式 `raise ValueError`） |

---

## 6. 数据侧发现（本次实抓验证）

本次按 `scripts/fetch_official_2026.py` 抓了 6 站 2026 Race 会话（385 个逐点文件，
`dirty=0, missing=0, 年检 0 条非 2026`），并用 `build_corner_dataset.py` 切出 6380 条
逐弯样本，过程中发现一个**上游数据质量问题**：

- **1 号弯窗口包含记圈点零速帧**：`min_speed` 或 `avg_speed` 为 0 的样本 **142 / 6380
  （2.2%）**，其中 **135 条集中在 `corner=1`**（各站分布均匀，20–27 条/站）。
  后果是 1 号弯的 `entry/min/avg speed` 的 p10 被拉成 0，若直接进模型会教出「起步弯速度
  为 0」的错误先验。
- 建议修复：`build_corner_dataset.py` 在切窗口前过滤 `speed <= 0` 的帧，或把 1 号弯窗口
  起点从 lap distance 0 改为 `> 30 m`。

本次交付的数据包对这类样本做了剔除并在 manifest 里计数（未改上游脚本，避免影响既有模型
训练口径）。详见 `data/f1_2026/README.md`。

---

## 7. 修复优先级建议

| 优先级 | 事项 | 预估 | 风险 |
|--------|------|------|------|
| P1 | 统一 Dx 量纲 + 回归测试 | 0.5 d | 会改变线上建议强度，需同步 CHANGELOG 与快照测试 |
| P1 | 回放停止真正生效 | 0.5 d | 低 |
| P2 | import-file 白名单 | 0.5 d | 低 |
| P2 | 测试隔离 data_dir / cleanup 保护 | 0.5 d | 低，但需改 conftest 与若干用例 |
| P2 | recorder 关闭竞态 | 1 d | 中，涉及线程模型 |
| P2 | 异常详情脱敏 + 版本号 | 0.5 d | 低 |
| P2 | 反馈 0 强度迁移 | 0.5 h | 低 |
| P2 | 闭环接口接线或删列 | 1 d | 需产品确认 |
| P2 | license / CI 收敛 | 0.5 d | 低 |
| P3 | 上表 14 项 | 1–2 d | 低 |
| 数据 | 1 号弯零速帧上游修复 | 0.5 d | 会改变训练样本分布，需重训并对比 MAE |

---

## 8. 复现命令

```bash
pip install -e ".[dev]" ruff pytest pytest-cov
ruff check . --statistics
pytest -q
pytest -q --cov=setup_tuner --cov-report=term

# 数据（本次实抓口径）
python scripts/fetch_official_2026.py --race "Italian Grand Prix" --session Race --max-laps 3
python scripts/build_corner_dataset.py --races "Italian Grand Prix" ...
python scripts/build_f1_2026_bundle.py
```