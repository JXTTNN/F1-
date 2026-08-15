# 变更日志 (Changelog)

> 本文件记录 f1opt 系统的优化迭代历史。每次迭代都有可验证的提升。

## 2026-08 优化迭代

### 文档一致性: 移除硬编码维度计数 (Iter-262)
- 反馈引擎/quality/__init__ 多处注释仍写死 "18/12/10 个维度", 与实际
  `FEEDBACK_DIMENSIONS` (现 20 维) 脱节。改为引用 `FEEDBACK_DIMENSIONS` 常量,
  避免后续新增维度时文档再度陈旧。

### 车手画像 DRS 使用效率修正 (Iter-261)
- **`_drs_usage_efficiency` 改用 `drs_active` (实际 DRS 激活, m_drs)**：原用
  `drs_allowed` (仅"此区段允许 DRS") 作代理, 会把"允许但未使用"也计为使用,
  高估 DRS 效率。现优先用激活状态, 无数据时回退 `drs_allowed`。

### analytics ERS 累计语义修正 (Iter-260b)
- **修复 `ers_sector_analysis`**：`ers_deployed_this_lap` 是累计能量(单调递增),
  原 `np.sum(dep_s[mask])` 会把扇区内所有累计值相加(数量级错误)。改为 `末-首`。
- **修复 `ers_recovery_efficiency_analysis`**：制动区部署量原 `np.sum(deploy)` 同样
  误作速率求和, 改为 `deploy[end-1]-deploy[start]`。
- **修复 `ers_overdeploy` 异常检测**：原读累计能量字段并与阈值 0.9 比较,
  会在几帧后恒为真(累计值>0.9 MJ)。改为读 `ers_deploy_mode` (0=none/1=medium/
  2=hotlap/3=overtake) 并判定 `>=2` (hotlap/overtake 持续窗口)。
- 测试：`test_detects_ers_overdeploy` 改用 `ers_deploy_mode=2`。

### 接入 ERS 扇区效率维度 (Iter-260)
- **修复 `_dim_ers_sector_efficiency` 死代码**：该维度 (函数 + NLG + 标签) 早已实现
  但从未接入 `rule_based_feedback` 构建器、也不在 `FEEDBACK_DIMENSIONS` 中, 且其
  依赖的 `values["ers_sector_efficiency"]` 从未被计算 → 永远无法出现。
- `extract_metrics` 现按 `lap_distance` 拆 3 扇区, 用 `ers_deployed_this_lap` 累计
  差 (部署) ÷ brake 代理 (回收) 计算各扇区效率, 供维度输出 `S1/S2/S3_eff`。
- 维度计数 19→20；测试 `test_ers_sector_efficiency_dimension_wired`。

### 模块衔接修复: ERS/DRS 字段对齐 + OpenAPI + EXE 静态资源 (Iter-259)
- **修复 ERS/DRS 字段命名错位**：aligner 产出 `ers_deployed_this_lap` /
  `ers_harvested_this_lap`, 但反馈引擎与 analytics 读的是幻影字段
  `ers_deployed` / `ers_harvested` / `ers_mgu_k_deploy` / `drs_zone`(从不产出),
  导致 ERS 部署/回收、DRS 区段分析**永远"数据不足"**。现已对齐并移除幻影字段。
- `ers_analysis` 修正累计语义: `m_ersDeployedThisLap` 是累计能量(单调递增),
  deploy_total 改为 `末值-首值`, 事件数用差分上升沿(原误作速率积分)。
- **修复 `/openapi.json` 500 (Swagger 不可用)**：`BatchFeedbackRequest` 原为
  `create_app` 内局部类, 在 `from __future__ import annotations` 下成为无法
  解析的 ForwardRef。提升到模块级。
- **移除 `/api/feedback/batch` 重复路由**：extended.py (Iter-183) 与 app.py
  (Iter-206) 各定义一次, 触发 Duplicate Operation ID 且旧版被遮蔽。保留新版权重。
- **修复 EXE 静态资源 404**：冻结环境下 UI 打包到 `sys._MEIPASS/static`,
  代码却按 `f1opt/ui/static` 定位 → EXE 中 dashboard/index 404。已按 frozen 分支定位。
- 测试：`test_ers_deployment_dimension_reads_aligned_field`。

### 内置 LLM 多反思自评 (Iter-258)
- **新增反射式 (reflective) 第二轮自评修正**：`llm_enhance` / `llm_enhance_async`
  在首轮回答后, 若 `F1OPT_LLM_REFLECTION=true` (新增 `Settings.llm_reflection`,
  默认关闭保持轻量), 会把首轮回答连同遥测证据回传给模型做"逐条核验数字是否
  有据可依 / 建议是否安全 / 粒度是否匹配"的自评, 返回修正后的最终答案。
- 新增 `REFLECTION_PROMPT_TEMPLATE` (prompts.py) + 第二轮 token 用量记录。
- 测试：`test_llm_enhance_reflection_refines_summary` (两轮调用)、
  `test_llm_enhance_no_reflection_by_default` (默认单轮)。

### 内置 LLM 尊重用户自定义模型 (Iter-257)
- **修复 `llm_model` 配置被静默忽略**：4 个 `llm_enhance*` 入口 + `preload_llm`
  一律硬编码 `_LLM_DEFAULT_MODEL`，用户通过 `LLM_MODEL` 环境变量或
  `Settings.llm_model` 指定的模型名从未生效。现改为 `config.llm_model or 默认`,
  云后端 (openai) 可切换到任意兼容模型 (如 gpt-4.1-mini / 本地代理模型)。
- 测试：`test_llm_enhance_honors_config_llm_model` (捕获 POST payload 断言 model)。

### 车手反馈接入 F1 2026 主动空力 (Iter-256)
- **新增 `active_aero_usage` 反馈维度 (第 19 维)**：反馈引擎此前对 2026 赛季
  招牌玩法主动空力 (X-Mode 低阻直道 / Z-Mode 高下压弯道) 完全无感——
  `col_multi` 未抽取 `active_aero_x/z`, 无任何维度反映切换节奏。现抽取并计算
  `active_aero_x/z_fraction` (激活帧占比), 按 F1 2026 规则 (Z 默认弯道 / X 仅直道)
  给出切换时机建议。
- `extract_metrics` 新增 `active_aero_mean_x/z` 与证据引用 (`sources`)。
- NLG 新增 `_narrate_active_aero_usage` 中文叙事 + `_DIM_LABEL_ZH` 标签；
  同时补上 `brake_temp` / `tyre_temp_gradient` / `grip_consistency` 三个此前缺失的中文标签。
- 测试：`test_active_aero_usage_dimension` (30%/70% 占比) + 无数据回退用例；
  维度计数 18→19。

### 遥测聚合分母修正 (Iter-255)
- **修复 `avg_ers_deploy` 分母错误**：ERS 累计值来自 CarStatus (20Hz)，却按
  CarTelemetry 的 `num_samples` (60Hz) 求平均，导致被稀释约 3×（与 Iter-254
  主动空力同源 bug）。现与 `avg_active_aero_x/z` 统一用独立的 `car_status_count`。
- `_LapState.active_aero_count` 更名 `car_status_count`（语义更清晰：CarStatus 样本数，
  同时服务于 ERS 与主动空力三字段）。
- 测试 `test_active_aero_averaged_and_exported` 扩展覆盖 `avg_ers_deploy`。

### Windows 兼容性
- **修复 CLI 入口点缺失**：添加 `[project.scripts]` 与 `cli.py` 的 `__main__` 块，
  修复 `f1opt` 命令不存在、以及 **EXE 打包后启动即退出** 的严重 bug。
- **移除已弃用的 ProactorEventLoop 显式设置**（Python 3.14+ 弃用）。
- **修复 `app.py` 的 `STARTUPINFO.dwFlags` 类属性错误**：该错误曾导致整个 API 层无法导入。
- **MemoryTracker 跨平台内存测量**：Windows 使用 `GetProcessMemoryInfo` (psapi)。
- **修复 `f1opt.spec` 的 `__spec_file__`**：改用 PyInstaller 正确的 `SPECPATH` 全局变量。

### F1 2026 阵容严格对齐 (10 队/20 车手 → 11 队/22 车手)
- 新增 **Cadillac** 车队 + Perez/Bottas 车手。
- Kick Sauber 正名 **Audi**。
- 修复 Racing Bulls 车手 ID 错位 (law/had)。
- 排位赛 Q1/Q2 淘汰数 5→6；赛季模拟器 22 车手 / 11 车队。

### 内置 LLM 车手反馈
- 同步反馈引擎维度计数 12→18（引擎已扩展但 docstring/测试停留在 12）。
- 修复 CLI `feedback` 命令调用不存在的 `engine.generate_feedback()` 方法。

### 遥测收集
- **架构重构**：recv 回调只解析 header（~1µs），body 解析移入异步 dispatch 循环，
  修复"recv 回调阻塞事件循环"的设计违规。
- 记录 Windows UDP 洪泛性能限制（25k pps 人工洪泛 ~60%，真实 60Hz 无影响）。

### 遥测分析
- 修复 `compute_all` 覆盖 bug：17 个分析方法（sector_timing/gear_usage/
  downforce_balance/brake_temp_balance/tyre_temp_gradient/fuel_per_sector 等）
  从未被全量入口调用，补齐为全部 28 个。

### 代码质量
- 修复 F821 未定义名（conversation/strategy 缺 Any、train 缺 Path、cli 缺 DriverProfile）。
- ruff 自动修复 32 处（未用 import/f-string/类型注解现代化）。
- 清理 F841 死代码（brake_model track_load、analytics fields/dt、corners med_frac）。

### EXE 构建
- `build.bat` 改为只装运行时依赖（不再为 EXE 安装 pytest/ruff/mypy）。

---

## 2026-08 第二轮优化 (整体性 + 工厂级 + 内置 LLM + UI)

### 内置轻量 LLM
- 修复 local 后端 (Ollama) 错误要求 API key 的门控 bug (内置 LLM 永远无法启用)。
- 添加 LLM 门控回归测试 (local 无需 key / openai 需 key)。

### F1 2026 严格对齐
- 新增主动空力 (X/Z-Mode) 分析：aligner 接入 m_activeAeroX/Z，analytics 新增
  active_aero_usage_analysis，compute_all 补齐为 29 项。
- 实时遥测 UI 新增主动空力指示器 (X-Mode 青 / Z-Mode 紫)。

### 调教优化
- 修复 search_setup 同 seed 非确定性：精英保留路径逐代跑 DE 但仅首代传 seed，
  改为每代派生 seed+gen，8 轮结果完全一致。

### 模块衔接
- 修复 f1opt train 硬编码 save=False (训练结果丢弃)，改为 --save/--no-save (默认保存)。
- 新增 f1opt teams list 命令 (暴露 11 车队 / 22 车手)。

### 工厂级质量
- mypy 配置 python_version 3.11 -> 3.12 (numpy 2.x 存根用 3.12 type 语法)。
- 全量回归：核心套件 3172 通过 / 0 失败。
- 安全扫描：无硬编码密钥/密码/令牌。

---

## 2026-08 第三轮优化 (点开就能用)

### 模块衔接 (关键修复)
- **修复 `f1opt serve` 遥测监听器未启动**：`cmd_serve` 此前硬编码
  `start_listener=False`，导致启动 API 后 UDP 遥测端口 (20777) 从未绑定、
  `/api/health` 恒报 `udp_listening:false`，F1 2026 游戏遥测无法接入。
  改为 `start_listener=True`，监听失败仍优雅降级 (API 保持可用)。
  实测 `f1opt serve` + `GET /api/health` 返回 `udp_listening:true`。

### 遥测收集 (性能关键修复)
- **修复 UDP 洪泛吞吐瓶颈 (唯一失败的压力测试)**：Motion body 解析从 ~86µs
  降至 **~13µs (6.6x)**。根因是每包 eager 构造 22 个 per-car 字典 (~58µs,
  GIL-bound)，而热路径 (aligner) 只读玩家单车。改为 `_LazyCarList` 惰性物化
  (按需构造单个车字典，保留 len/索引/迭代/变异契约)。
  `test_stress_udp_flood_6000_motion` 从 ~60% 投递 (阈值 75%) 提升为 **通过**。
  全量遥测套件 359 passed。

### 测试可靠性 (工厂级)
- **修复 e2e UDP 真实 socket 测试的发送时序**：`test_e2e_udp_listener_real_socket`
  此前同步连发 10 包无 yield，内核丢包导致只收到 1 包 (预存失败)。改为每包
  `await asyncio.sleep(0)` 让监听器 recv 回调排空 socket，测试稳定通过。

### 分发 (EXE 可下载)
- **发布 GitHub Release v1.1.0 (Windows EXE)**：从最新源码重建并上传
  `f1opt.exe` (55MB, PyInstaller, 含 serve 遥测修复 + 惰性解析) 至 Release，
  直达下载链接：
  https://github.com/JXTTNN/F1-/releases/download/v1.1.0/f1opt.exe
  (SHA256: 4A284EFDC0F9385FF2BAC9A51CD82EE0239A6D46E41EB05062E1D1476B627884)。
- README 补全 Windows/Linux 双平台下载入口。

### 遥测监听器稳健性 (工厂级, 修复测试套件挂起)
- **修复 `TelemetryListener.stop()` 挂起 bug**：stop 原用 sentinel 排在队列尾部
  等待 drain；当存在慢订阅者 (如 10s sleeper) 时，每个排队包都要等
  `_SUBSCRIBER_TIMEOUT`(5s)，导致 `test_stress_comprehensive.py` 整体挂起
  (~95s+)。改为直接 `cancel()` dispatch 任务即时退出。
  修复后 `test_stress_comprehensive.py` 从**挂起**变为 **187s 完成**
  (113 passed / 6 预存失败, 与本次改动无关)。
- `_unpack_body` 避免精确长度时的冗余字节拷贝 (工厂级微优化)。

### 输入校验 + 压力测试修复 (工厂级)
- **修复未知赛道被静默接受**：`f1opt predict/search/bayesian/validate` 与
  `POST /api/predict` 此前对未知 track_id 静默回退到默认圈速 (71.37s)，
  CLI 返回 0 / API 返回 200。现在 CLI 返回 1、API 返回 400。
- **修复 6 个预存压力测试失败**：
  - `test_concurrent_session_read_write` 使用已重命名的旧 API (add_message → add)。
  - `test_asyncio_with_subprocess` 用 `echo`(Windows 非可执行文件) → 改用 `sys.executable`。
  - `test_concurrent_search/feedback` 未考虑 20/min 限流 → 断言改为
    "全部响应 (200/429) 且至少 1 个成功"。
  - `test_cli_invalid_args_graceful_exit` / `test_api_404_on_unknown_track` 由输入校验修复。
  修复后 `test_stress_comprehensive.py` 125 项全绿 (无失败)。

### 实时遥测热路径性能 (关键)
- **`latest_unified_frame` 51x 提速**：该函数在 60Hz WS 广播路径上每包调用，
  原实现每次对 5 个源的 buffer 排序 (O(N log N)) 并为每字段构建全量样本列表
  (O(N·F))，随会话进行 buffer 增长到 ~5000 样本时单次耗时 **51.7ms**
  (已超 60Hz 16.7ms 预算，导致 UI 实时性随圈数恶化)。
  改为直接取每个源的最大时间样本 (O(S·F))，**1.0ms**，输出逐位等价。
  实测 5000 样本: 51.7ms → 1.0ms (51x)。

### 模块衔接 (F1 2026 主动空力 UI 修复)
- **修复 WS 帧投影缺失主动空力字段**：aligner 已产出 `active_aero_x/z`,
  UI 指示器也读取 `active_aero_x/z`, 但中间的 `_frame_to_ws` 投影漏掉了这两个
  字段, 导致 UI 的 X-Mode/Z-Mode 指示器**永远显示 "—"**。补上后 UI 指示器正常。
  新增 `test_frame_to_ws_includes_active_aero` 回归测试。
- **修复 lap 广播消息 `track_id` 硬编码 None**：`_emit_lap` 此前固定
  `track_id=None`, 现按 aggregator row 的 int8 track_id 解析, 未知时回退
  `state.current_track_id`, 与 `_feed_observation_buffer` 的解析逻辑一致。

### UI 点开即用 (智能分析中心打通)
- **默认 `f1opt serve` 挂载完整 App (核心 + 扩展路由)**：智能分析中心
  (`/dashboard.html`) 调用的 `/api/bayesian-search` / `/api/pareto-search` /
  `/api/compare/*` / `/api/weather/impact` / `/api/health/extended` 均在扩展
  路由内, 此前仅 `--extended` 才挂载, 导致默认 serve 下这些 tab **全部 404**。
  现在默认即完整 App, 实测默认 serve 下以上端点全部 200。
- `index.html` 头部新增「智能分析中心」链接, 与 `dashboard.html` 的「返回实时面板」
  形成双向导航。

### 调教搜索 (Pareto 单目标崩溃修复)
- **修复 `/api/pareto-search` 单目标崩溃**：`MultiObjectiveOptimizer.evaluate()`
  始终返回 `[lap_time, tire_wear_proxy]` (2 值), 但 `search()` 用用户传入的
  `objectives` 构造 ParetoFront; 当 `objectives=['lap_time']` (1 目标) 时崩溃
  `ValueError: values length 2 != objectives 1`。现归一化为规范双目标, 单目标
  请求返回 200。新增 `test_search_single_objective_does_not_crash` 回归测试。

### 车手反馈性能 (接线死代码优化)
- **接线 `col_multi` 批量字段提取 (Iter-229 死代码激活)**：`extract_metrics`
  此前用 `col()` 逐字段遍历帧 (33 次调用 → 33 趟 O(n) 扫描), 而 Iter-229 已实现
  的 `col_multi` (一趟 O(n) 提取全部字段) 从未被调用, 是死代码。现改为单次
  `col_multi()` 提取 32 字段, 移除死掉的 `col()`。600 帧实测 6.64ms → 5.65ms。
  feedback 315 + smoke 109 = 424 passed。

### 类型安全 (工厂级 mypy 收敛)
- **engine.py 轮胎分析段 mypy 错误 12 → 0**：`wears`/`temps`/`inner_temps`/
  `outer_temps` 列表此前混入 `None` (`list[float | None]`), 触发 ~12 处
  `float | None` 类型错误。改为「仅追加 float + 用 `len(list)==N` 判断」的
  类型收窄写法, 移除全部 `# type: ignore` 注释。feedback 315 passed。
- **cli.py mypy 错误 10 → 0**：`cmd_search` 复用了 `result` 变量承载两种返回
  类型 (bayesian dict vs SearchResult), 改为独立变量 `bayesian_result`/
  `search_result`; `grouped` 字典补上 `dict[str, list[dict[str, Any]]]` 注解。
  全项目 mypy 80 → 70 错误。test_cli 25 passed。

### 内置 LLM 状态准确性
- **`preload_llm` 对 local (Ollama) 后端做可达性检查**：此前 preload 在未验证
  Ollama 是否运行的情况下直接返回 `loaded=True`, 用户以为 LLM 已就绪, 实际每次
  反馈请求都 10s 超时后静默回退。现在对 `http://localhost:11434/api/tags` 做
  2s 健康探测, 不可达时返回 `loaded=False` + 明确 reason。
  新增 `test_preload_llm_local_backend_unreachable` 回归测试。

### 类型安全 (mypy 继续收敛 70 → 66)
- `config.py`：`udp_port`/`api_port` 的 `_env` 返回 str 却标注 int → lambda 内
  `int(_env(...))` 显式转换。
- `pareto.py`：`MultiObjectiveOptimizer.bounds` 标注放宽为 array-like。
- `aligner.py`：`latest_unified_frame` 复用 `it` 变量 (tuple vs Optional) → 改
  独立变量 `entry`。
- `setup_schema.py` / `setup_physics_bridge.py`：`CarSetup(**dict)` 动态拆包
  无法被 mypy 验证字段类型 → 加 `# type: ignore[arg-type]`。全项目 66 → 63。

### UI 设计 (实时遥测状态可见性)
- **health badge 显示实时遥测统计**：`index.html` 的 health badge 此前只显示
  "UDP 监听中/未监听", 无法判断游戏是否真的在流式传输。现每 5s 额外拉取
  `/api/telemetry/stats`, 显示「收包 N · 圈 M」, 用户一眼确认 F1 2026 游戏
  遥测已接入 (收包递增 = 正常流式, 圈数 = 已完成圈)。

### 类型安全 (mypy 继续收敛 63 → 60)
- `ea_f1_2026_benchmark.py` / `quality_score.py`：`min/max(..., key=dict.get)`
  的 `.get` 返回 `T | None` 触发 overloaded 类型错误 → 改用 `lambda k: d[k]`
  (直接索引, 返回 `T`)。

### UI 设计 (新手接入引导)
- **收包为 0 时显示 UDP 设置提示**：health badge 现在当 `listener_received==0`
  时显示琥珀色警告「收包 0 — 请在 F1 2026 游戏设置开启 UDP 遥测(端口 20777)」,
  新增 `.badge.warn` 样式。新用户开箱即可知道如何让游戏遥测接入, 而非面对
  「收包 0」无从下手。

### 类型安全 (mypy 继续收敛 60 → 52)
- `race_weekend_2026.py`：`_report` 字段标注 `WeekendReport2026 | None` 导致
  6 处 `union-attr` 错误。改为 `field(init=False)` + `__post_init__` 初始化非
  None 报告容器, 消除 transient None 状态 (正确性改进, 非仅标注)。
  race_weekend 29 passed。

### 类型正确性 (mypy 继续收敛 52 → 50)
- `lap_simulator_2026.py` / `tire_curve.py`：`tire_age_laps` 标注为 `int` 但
  实际为 float (SC/VSC 期间可分数磨损), 导致 `int`/`float` 类型不一致。
  改为 `float` 贯穿 lap simulator + tire_curve 两个 `lap_time_delta_s` 签名
  (函数内部本就 `int(...)` 收窄, 运行时等价)。另 `p` 循环变量复用 → 改名
  `period`。tire_curve + lap_simulator 105 passed。

### 遥测热路径微优化 (itemgetter)
- `aligner.py`：`_sorted_items` / `latest_unified_frame` 用 `lambda x: x[0]`
  做排序/取最大值 key, 每元素一次 Python 函数调用。改为 C 级
  `operator.itemgetter(0)`, 实测 sorted **3.3x**、max **1.76x** 更快
  (20000 样本)。60Hz WS 广播热路径 (latest_unified_frame) 进一步提速。
  遥测 359 + e2e smoke 10 = 369 passed。

### 类型安全 (mypy 继续收敛 50 → 46)
- `gap_filler.py`：线性插值前补 `assert prev_val/curr_val 非 None` (帮助
  mypy 收窄 4 种 None 组合)。
- `nlg.py` / `deep_profile.py`：`float(after)` / `float(min(lts))` 的
  `Any | None` 误报加定向 `type: ignore`。gap/nlg/deep_profile 91 passed。
- `strategy.py`：燃油节约策略循环 `mode[...]` 返回 `object` → 显式 `float()/int()`
  转换 + 定向 `type: ignore`。strategy 21 passed。全项目 46 → 43。
- `analytics.py`：`high_g / max(low_g, 1)` 中 `np.int64` 与 int 混合类型触发
  overloaded 错误 → 显式 `float(high_g) / max(float(low_g), 1.0)`。
  analytics 31 passed。全项目 43 → 41。

### 类型/死代码修复 (mypy 继续收敛 41 → 35)
- `pareto.py`：`_mutate` 的 `mask` 加 `np.ndarray` 显式注解 (mypy 误判为 bool)。
- `diagnostics.py`：**移除重复定义的 `_predict_lap`** (第一个无 `model.eval()`,
  被第二个遮蔽, 是死代码)。
- `strategy_optimizer.py`：`compounds_pool` 注解 `tuple[str, ...]` (干地 3 元素
  vs 湿地 2 元素长度不一致)。pareto+diagnostics+strategy_optimizer 69 passed。
- `bayesian.py`：`_compute_factor` 用局部 `X`/`y`/`L` 变量替代 `self._X`/`_y`/`_L`
  (帮助 mypy 收窄 None), 语义等价。bayesian 24 passed。全项目 35 → 33。

### 打包/插件 (安装技能)
- `pyproject.toml` 补全可选依赖：dev 加 `pytest-timeout` (支持测试超时保护,
  防止未来压测挂起), 新增 `build` extra (含 `pyinstaller`, 一键 EXE 构建)。

### 类型安全 (mypy 继续收敛 33 → 32)
- `train.py`：`_held_out_mae` 的 `model` 参数标注放宽为 `SurrogateModel |
  EnsembleSurrogateModel` (实际两种模型均接受, 且均有 `predict` 方法)。
  train 26 passed。
- `surrogate.py`：`EnsembleSurrogateModel` 新增类型化 `_members` 并行列表
  (`list[SurrogateModel]`), 循环改用 `self._members` 替代 `nn.ModuleList` 迭代
  (mypy 把 ModuleList 元素推断为 Tensor, 导致 `m.predict()` 报 "Tensor not
  callable")。surrogate 23 passed。全项目 32 → 29。
- `bayesian.py`：`predict` / `log_marginal_likelihood` / `_optimize_hyperparams`
  的 `neg_lml` 闭包改用局部 `X`/`y`/`L`/`alpha`/`V`/`w` 变量 (替代可选属性),
  彻底消除 `ndarray | None` 收窄问题。bayesian 24 passed。全项目 29 → 22。

### 类型正确性 (mypy 继续收敛 22 → 18)
- `app.py`：`_emit_lap` 的 `track_id` 注解 `str | None` (current_track_id 可为 None)。
- `race_simulator.py`：`_compute_lap_time` 捕获局部 `sim` + None 兜底。
- `season_simulator.py`：`position`/`driver_id` 显式 `int()`/`str()` 转换。
  race+season+api 65 passed。
- `pareto.py`：`_mutate` 的 `mask` 改用 `np.asarray(..., dtype=bool)`
  (显式产出 ndarray, 规避 mypy 把 `Generator.random(n) < prob` 误判为 bool)。
- `strategy_optimizer.py`：`candidates`/`out` 组合列表注解 `list[tuple[int/str, ...]]`
  (消除变长元组长度不一致错误)。pareto+strategy_optimizer 50 passed。全项目 18 → 12。
- `surrogate.py`：`state_dict`/`load_state_dict` 覆写签名加 `# type: ignore[override]`
  (有意覆写: 返回含版本号的富字典, 与 torch `Module` 签名不同)。全项目 12 → 8。
- `app.py`：`lifespan` 返回类型 `None` → `AsyncIterator[None]` (修复
  asynccontextmanager 类型)；`add_exception_handler` 加 `# type: ignore[arg-type]`。
  api 25 passed。全项目 8 → 5。
- **mypy 全项目清零 (里程碑)**：修复最后 5 处 —
  `surrogate.py` tensor 变量重命名 (`sec_res_t`/`resp_res_t`)、
  `strategy_optimizer.py` 去重注解、`season_simulator.py` `pos` → `race_pos`。
  **`mypy f1opt/` → Success: no issues found in 117 source files**。
  season+strategy_optimizer 33 passed。全项目 5 → 0。

### 采集数据正确性 (模块衔接 bug 修复)
- **修复主动空力数据在 parquet 导出时丢失**：`_SCHEMA` 缺少
  `avg_active_aero_x/z` 字段, `pa.Table.from_pylist(rows, schema=_SCHEMA)` 会
  静默丢弃这两列。补上 schema 字段。
- **修复主动空力平均值分母错误**：`avg_active_aero_x` 原用 `num_samples`
  (CarTelemetry 60Hz 计数) 作分母, 但主动空力来自 CarStatus (20Hz), 平均
  值被错误稀释 3x。新增独立 `active_aero_count` (CarStatus 计数)。
  新增回归测试。telemetry 360 passed。

### UI 设计 (ERS/DRS 状态显示)
- `_frame_to_ws` 补传 `ers_deploy_mode` 与 `drs_active` (aligner 已产出但 UI 未显示)。
- 实时面板：DRS 显示区分「DRS 开」(绿) / 「DRS 允许」(灰)；新增 ERS 部署模式
  (无/低/中/Hotlap/超车)。ui+api 54 passed。mypy 仍全绿。

---

## 已知限制 (Known Limitations)

- ~~**UDP 洪泛吞吐**~~：**已修复**。惰性 per-car 物化使 Motion 解析 6.6x 提速，
  人工洪泛测试现已通过。剩余上限为真实 UDP 内核缓冲 + 事件循环调度，远高于
  F1 2026 实际 60Hz 速率。
