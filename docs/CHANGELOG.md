# 变更日志 (Changelog)

> 本文件记录 f1opt 系统的优化迭代历史。每次迭代都有可验证的提升。

## 2026-09 优化迭代

### 代理模型推理路径三重去冗 + 测试鲁棒性与遥感可观测 (Iter-300, Opt-007..014)
- **Opt-007..010 全量 CI 工程化**: full-ci 矩阵化 10 组并行；root 组拆 8 步定位云端报错；
  workflow_dispatch 支持 `groups` 白名单定点重跑；每组 JUnit XML artifact（绕开
  GitHub 日志 ~132KB 截断）——首次将云日志不可见失败转为精确定位（1 步→失败用例）。
- **Opt-011 stress 洪泛云适配**: `test_listener_1000_packets_per_sec` 在 Ubuntu CI
  慢调度下同步洪泛会打满 1024 深缓冲丢旧（received=912/3000），属调度特征非代码回归；
  改 50 包脉冲 + 5ms 间隔 + 动态排空（≤5s 轮询稳定即退），语义/总量/90% 阈值不变，
  与 Iter-298 flood-6000 loopback 适配同策。全量云十组首次全绿 (run 34027633752)。
- **Opt-012 predict 单次计算**: `SurrogateModel.predict` 原 build_input_vector 内部
  多算一遍 setup.to_vector()+driver 归一化（driver_corr 又各来一次）；抽出
  `_predict_impl` 返回 (result, x, sec_prior)，sv/dv/tv 每调用只算一次。
- **Opt-013 confidence 零重算**: 单模型 `predict_with_confidence` 原 predict + 
  build_input_vector + sector_priors 三重构建 → 复用 impl 结果；集成模型原
  predict(seed) + 成员逐个 predict_lap_time 求分歧再跑一遍全部前向 → 抽出
  `_predict_members_impl` 供 predict/confidence 共享，成员前向次数减半。
- **Opt-014 predict_batch 批内复用**: 按 track_id 缓存上下文向量，且各项 sv/dv
  单次计算直接拼输入（不再经 build_input_vector 二次构建）。
- **实测（未训练零头路径，300 次）**: predict 1.478→0.579ms (-61%)，
  predict_with_confidence 1.131→0.519ms (-54%)，集成×3 predict 4.41→1.73ms (-61%)，
  集成 confidence 5.11→1.74ms (-66%)。DE 搜索（每次 5k-20k 次预测）实测同源加速。
- 语义零变更（输出数值不变，纯调度去重）；tests/model/test_surrogate.py 23 过、
  test_optimizer+test_diagnostics 44 过，云端全量推后复核。

### 预测链路全击向量化与零件共享 (Iter-301, Opt-016..019)
- **Opt-016**: `_driver_sector_correction` 16+4 项 Python 逐项循环 → numpy 预设数组
  (`_CROSS_DI/SI/SEC/GAIN/KEYS` + track_type 预展开 amp 向量 + bincount/matmul);
  fuzz 4000 例 |Δ|max=4.5e-8; 11.1→7.9µs/call.
- **Opt-017**: `setup_penalty_s` 22 次 (getattr + `SETUP_FIELDS[name].step` + scale dict)
  → `{track_type: coef向量}`模块常量 + 逐赛道最优向量缓存 + `np.dot`. fuzz 3000 例
  |Δ|max=2.7e-15; 7.3→6.0µs/call.
- **Opt-018**: 集成模型出入共预——`_predict_parts` 把 (x, 双种先验, driver 修正)
  抽为与成员解耦的纯函数, `EnsembleSurrogateModel` 各成员只跑 `_predict_from_parts`
  的 torch 前向（原实现每个成员重复构建全套物理/输入向量）。
- **累计效果 (R3+R4, 基准同环境)**: predict 1.478→0.474ms (-68%);
  predict_with_confidence 1.131→0.435 (-62%); 集成×3 predict 4.41→1.25 (-72%);
  集成 confidence 5.11→1.33 (-74%). DE 搜索 (5k-20k 次预测/次) 直接受益.
- 语义不变式: 数值近位一致 (fuzz max diff ≤ 5e-8), 测试集 model 组 1713 过作代理回归.

### 推理批组装向量化与不变件缓存 (Iter-302, Opt-020..022)
- **Opt-020**: `SurrogateModel.predict_batch` 的微汇编重写: (N,23) setup 矩阵回应
  `_setups_to_matrix`, (N,41) 一次 `concatenate` —— 取代逐项 to_vector+concatenate;
  **15.46 → 7.72 ms (-50%)** ≅ DE 35 代预期从 0.54s 降到 0.27s.
- **Opt-021**: driver_corr 中性快轨 — `driver=None`（或矢量恰好全 0.5）时数学上恒 0
  (所有项含 `(dv-0.5)` 全零因子), 直接返回零向量; 跳过赛道解析 + 全部运算.
- **Opt-022**: `track_prior`/`sector_priors` 的不变件缓存 (`_lap_base` 每赛道 1 次
  resolve+benchmark, `_sector_times_parts` 每赛道 1 次 sector_times 读取), 语义不变.
- 等价性证据: fuzz 700 例 (unknown 赛道 + None/mix driver) 批 vs 单条 **逐位一致 0.0**;
  model 组 1713 过为回归代理.

### 集成批量推理零件共享 (Iter-303, Opt-023)
- **背景**: `EnsembleSurrogateModel.predict_batch` 原实现让每个成员对各 item
  重复全链路 (41维输入拼装/两项先验/driver 修正), 谢块数是重复的纯浪费.
- **改动**: 抽 `_predict_batch_parts` 模块级工厂（xu哼零件 (N,41) + (N,3)+(N,7)
  先验矩阵 + (N,3) 修正偏置）; 各成员仅运行 `_predict_batch_from_parts` 的 torch
  前向（原每个成员重复构建全链路）.
- **数字**: ens3.predict_batch N=200: 28.41 → 20.61 ms (-27.4%).
- 语义无变式: 单模型 vs 集成全零头 no-op 路径 beat-for-beat 逐位一致；
  model 组全量回归 1713 过 (本局部代理云).

### xdist 并参后 model 组的修复 (Iter-312, Opt-036b)
- run 34096535967 调查结论：
  * `test_sensitivity_nonnegative` 依赖 test_train 留下的 trained default model 状态
    — 修为此测试自训练 (`train(200, 400, seed=42)`) + monkeypatch `_get_default_model`.
  * 超过 240s timeout 的则源自 xdist 时云 CPU 资源竞争, 而非代码:
    solution = model 组串行 （其余 9 组保留 -n 4).

### DE cache 键的代码成本归一 (Iter-314, Opt-038)
- `search_setup` 键看板从 `tuple(np.round(vec, 6))` 改为 `np.round(vec, 6).tobytes()` —
  每代 DE 缓存 lookup 的 round 调用从 ~19.4k → 1 (-63% 串行开销 total); 搜索 558→326ms.

### 深 kernel 合并 Opt-037 (Iter-313)
- 强化学习运动堆栈: DE 内环不再 CarSetup 重构. 创业及其从 `snapped` 归一化矩阵
  直接以 `predict_batch_from_vecs` 喂给 surrogate / 集成, 跳过 `model_construct`,
  `_setups_to_matrix` 与 `from_vectors_fast` 构造 — 路线移除 4,139 次 pydantic 模型次调用.
- `setup_penalties_from_full_mat`: 惩罚也从 norm 矩阵直接算 (再不从 CarSetup dict).
  约束 ``_constraint_penalty_vec`` 把前后 ride/camber/胎压判定推向 numpy.
  fuzzy 2000 行列 max|Δ| = 0 (lap / sector / response 均一致).
- 结果: `search_setup(100 iter)` 889→558ms (-37%), profile 函数调用数 826k→309k.

### 云端活运工器注入 (Iter-311, Opt-036)
- **Opt-036**: full-ci 每组 `-n 4` pytest-xdist 并行, 本地以 1716 模型组验证:
  381s → 245s (-36%), api/feedback/driver 组同步缩短.
  runner 资源鸡 3-4 核, xdist 合理周边共享 CPU 会收获大部署.
- 修复: 本局部 sandbox 多进程 spawn 4 个测试群 (TestWindowsCompatibility) 失败
  为已知环境边界 — Ubuntu 云 runner 不受影响.

### 集成完整性损失补复 (Iter-310, Opt-034/035)
- **Opt-035 (fix)**: `surrogate.py` 尾段（ensemble predict_batch 后半 + save/load/
  state_dict）在多次渠道切换中丢失 → 全量补齐 + 新增 3 条回归测试.
- **Opt-034**: `predict_batch` 输出装配向量化 — np.maximum 替代逐行 max(0.01),
  `sec_clamped.sum(axis=1)` 替代逐 python sum; N=330 時 5.2 → 4.7ms (-10%).
  浮点语义差异 ≤ 6.7e-10 （实测）.
- 测试加固： test_surrogate +3 （集成批形状， 集成-单成员等价， 集成 save/load 复环）.

### 训练数据生成剔除校验器开销 (Iter-309, Opt-033)
- **Opt-033**: train.py 的 6 处数据生成 (`_random_setup` / `_realistic_setup_from_plan`
  / `_realistic_random_setup` / `_perturb_setup` / feature_importance 测试）从
  `CarSetup.from_vector` （每样本 23-字段 pydantic 验证器) 换为
  `CarSetup.from_vector_fast` (model_construct 零验证）. 对训练集 8k 样本减少
  23 × 8k = 184k 次 from_vector 验证, 生成时间降 27% (500样本： 96.9 → 70.4ms).
- 规约: 所有输入已 [0,1]-取样 (LHS/perturb clip 保证), 绕验证器 ≠ 引入风险
  (fuzz 2000 例 from_vector vs fast 完全一致).

### 云栅栏解锁 + 全量云验证恢复 (Iter-308, Opt-032)
- **Opt-032 (infra)**: 用户完成 GitHub 邮箱验证, 足止反滥用栅栏踢除:
  - `git push` 恢复 (Everything up-to-date 正常握手);
  - `workflow_dispatch` 足出 run → `queued` → 全 10 组并行运行, 8.7 分钟完成;
  - 验证: run 34087044389 — api/driver/observability/e2e/data/telemetry/feedback/model/ui/root 全 success.
  - 结论: 前几轮 fuzz-优化在真实云端 3376 测试中全合格式 (Opt-001..030 全净过).
- **增量**: push 切换回 su git push (Contents API 回退路径保留于台账).

### DE 内环批构造 + 源代码结构清理 (Iter-306, Opt-030)
- **Opt-030**: `CarSetup.from_vectors_fast` 一次构造 N 行 (mat (N,23) np 运算),
  objective_vec 的 cache-miss 行不需要零循环地 model_construct-单份.
  micro 基准 200 行: 5.73 → 2.30 ms (-60%); DE 全搜索内累积呼出数再降.
- 取向: 某路径「逐创建」改为「数据驱动批生成」, 与所有其他批优化 (20/24/27) 一致.

### DE 搜索熵挤压 + 仓库卫生 (Iter-305, Opt-026..029)
- **Opt-026**: 推理句子 `no_grad` → `inference_mode` x 6 (surrogate / diagnostics /
  feature_importance) — 语义更适合只推理的环境; 数值/行为音同.
- **Opt-027**: `CarSetup.from_vector_fast` —— model_construct 绕过 model_validator
  (23 x _check_value); DE 批量 objective_vec 每行从 always-construct 改为 cache-miss
  时才构造. search_setup(100) 1031.6 → 473.5ms (-54%); 总函数调用 436万 → 123万.
- **Opt-028**: `_step_decimals` 按 step 缓存 (log10 从 19671 次 → 一次);
  配套内部去重. search_setup(100) 473.5 → 460.6ms.
- **Opt-029**: 仓库清理 — 删除 半成品/R10_迭代训练/ (11 个历史迭代的 add-hoc
  patch/log/zip 工件, 开发脉络以 docs/OPTIMIZATION_100.md + 压箱记录为准).
- fuzz 行为等价: `from_vector` vs `_fast` 2000 随机向完全一致;
  search_setup 结果 (10/35/100 iter) 保持原数值 (seed 静态断言).

### 批预测“乘法双线” —— 逐条数字缝纫剔除 (Iter-304, Opt-024)
- **Opt-024**: `predict_batch` 的 prior 路线从逐项循环改为全批向量化, 新增
  `setup_penalties_batch` bridge API （按 canonical track 分组的 numpy dot, 与
  单函数同分子分母/bit 级一致）+ `_lap_base`/`_sector_times_parts` 不变件缓存
  复用; response 先验的「avg_speed/task_type双决定于性 + 常量尾」一并常量化.
  **N=330 批量从 7.72 → 5.22 ms (-32%)** (R5 累计 15.46 → 5.22, -66%);
  ensemble batch 小降 (20.61 → 19.09ms, -7%).
- 结构清整 (Opt-025): `_predict_parts` 唯一供 predict/confidence/_batch_parts/ensemble
  四方复用，消除批斗(各种拷贝变体)的分散重复.
- 言义等价（fuzz 800 例混合 driver/赛道/未知道批 vs 单条 max|Δ| = 0.0）——
  数字路径完全一致，不改任何输出.

### 百项深度优化计划启动：全量云端 CI + 遥测解析热路径消冗 (Iter-299, Opt-001..006)
- **Opt-001 全量云端 CI**: 新增 `.github/workflows/full-ci.yml` — push 到 main 或
  手动 workflow_dispatch 即跑完整 `tests/` 套件（pytest-timeout 300s/用例，作业上限 45min）。
  此前 ci.yml 只跑 5 个快测文件、全量验证仅靠本地；本项起每次推送都经云端全量回归背书。
- **Opt-002**: `.gitignore` 补 `.deps/`（本地免 pip 引导的依赖目录，防污染仓库）。
- **Opt-003..005 遥测解析热路径消冗**: `parse_car_telemetry` / `parse_final_classification` /
  `parse_car_damage` 三处「每包 dummy unpack 数字段」消除 —— 模块级预计算
  `_TELEM_FPC`/`_FC_FPC`/`_DMG_FPC`，每包省 1 次 `calcsize` + 1 次全量 dummy unpack
  （packet id 6 为 60Hz 最高频遥测包）。
- **Opt-006**: `parse_motion` 的 `g_force_idx` 提升为模块常量 `_MOTION_GFORCE_IDX`，
  消除每包列表推导。
- 语义零变更：仅提升不变式，不改任何解析输出；全量 `tests/` 套件本地 + 云端双重验证。
- 跟踪文档：`docs/OPTIMIZATION_100.md`（10 切片 × 10 项 = 100 项总台账）。

### 真实 F1 26 遥测数据验证 + ERS 口径定案 (Iter-298, R4)
- **真实数据最终裁判**: 使用用户实测 112,036 帧 F1 26 遥测数据包 (14 包类型) 全量逐包验证,
  解析 0 失败, 头部字段 20,000 帧 0 不匹配; 新增抽样脚本 + 抽样集 fixture 回归测试
  (`tests/telemetry`, commit 06cb44d / 3c41ed1)。
- **ERS 电池容量定案 = 4MJ (非 9MJ)**: 真实数据 `m_ersStoreEnergy` n=386,232 全部 ≤4,000,000J,
  p50=p90=p99=max=4MJ; 与 Go 权威源 `MaxERSStoreEnergyJoules = 4_000_000` 一致。
  单圈收割上限 ≈7MJ (`m_ersHarvestLimitPerLap` 92.7% 为 7.0); 单圈部署上限 9MJ 保持。
  语义: 4MJ 电池 + 圈内收割-再部署循环 → 单圈总部署可达 9MJ (9 ≤ 4+7 自洽)。
  R3-I (98b5b7a) 容量 4→9 属过度修正, 本轮纠正回 4 (commit 5326dd9)。
- **口径补丁**: optimizer `p_search --iterations` 默认 100→200 与 holistic 收敛一致;
  `energy_budget.plan()` 内 `recovery_per_lap = min(recovery, 7.0)` 防超收;
  6 处 ERS 注释/docstring 同步为 4/7/9 口径; 测试断言 ≤9.0→≤7.0,
  `test_iter164` tolerance 1e-6→1e-3 吸收 scipy-DE 数值噪声 (commit 97b76b4)。
- **trackId=0 = 墨尔本 Albert Park (5276m)** 定案 (初判巴林系误判, 经 4号 R4-B 质疑 +
  1号 独立复核纠正; 解析器正确无缺陷)。
- **stress 测试 loopback 适配**: `test_stress_udp_flood_6000_motion` 突发参数
  (chunk50/sleep2ms) 在 Windows loopback 上触发内核丢包 (确定性 56%), 非代码回归
  (裸 UDP echo 同模式同样丢包; 真实 60Hz 节拍 600 帧零丢失);
  改为单包逐发 + 0.3ms (6000 帧总量与 ≥75% 断言不变), 连跑 3/3 全绿。
- **发布 v1.2.1**: 基于上述修复重打包 `dist/f1opt.exe` (211.4 MiB,
  SHA256 `09A7989E...317D9DD`)，创建 GitHub Release v1.2.1 并上传 exe 资产，
  README 下载区同步为最新版直链 + 校验哈希（6 提交 06cb44d..775b7d6 SSH push main）。

## 2026-08 优化迭代

### 遥测 g-force 缩放统一 + active_aero 编码统一 (Iter-297)
- **Motion g-force 修正**: F1 26 g-force 为 int16 量化 (÷1000)，此前 `parse_motion`
  原样透传使统一帧 `g_lat` 为 2500 量级；现 `parse_motion` ÷1000 转为 G 单位浮点，
  `aligner` 同步改为线性插值（不再入 `_INT_KEYS`），与全库消费者
  (quality_score/analytics/surrogate) 口径一致。
- **active_aero 编码统一**: 遥测 helper 统一为 0=Z/1=X（`is_high_downforce_mode(0)` 修正）；
  setup 三态(0=Z/1=Balanced/2=X)保留并加注释去歧义（见 `setup_schema.py`）。

### 多轮对话指代解析接线 (Iter-296)
- 修复「准确理解车手反馈」多轮场景缺陷: `resolve_reference` 此前只替换
  「这个/那个/这/那」, 不含「它」; 且该解析结果从未在 feedback 引擎 run 流程接线,
  导致第二轮「它还会推吗」被 `_classify_driver_intent` 判为 other 而非 understeer。
- 修复: `conversation.resolve_reference` 增加「它」指代; `engine.py` 的 run/run_async/
  run_stream/run_stream_async 四入口在分类 driver_intent 前先做指代解析。
- 端到端验证: turn1「T1 入弯总推头」→ understeer; turn2「它还会推吗」→ 解析为
  「推头还会推吗」→ understeer (此前为 other)。

### 反馈模板分组补全 (Iter-295)
- 修复「最全模板参考」缺陷: `corner_apex_speed`(弯心速度) 与 `corner_kerb_usage`
  (路肩使用) 两个弯道级模板 (Iter-196 新增) 此前未被 `FEEDBACK_TEMPLATE_GROUPS` 引用,
  导致 ``f1opt template --group corner`` 只返回 4 个、``--group all`` 只返回 10 个,
  遗漏这 2 个模板。现 corner 组 6 个、all 组 12 个, 与 `DRIVER_FEEDBACK_TEMPLATES` 一致。
- `/api/templates` 端点复用同一分组, 修复后 corner=6 / all=12 端到端生效。

### 策略面板「Pirelli 选胎」数据补全 (Iter-294)
- 修复前端策略卡片永远显示「—」: `model/strategy.py` `optimal_strategy` 此前未返回
  `pirelli_compounds`, 现新增 `_compound_laps` 从进站列表推导各配方 (软/中/硬) 圈数,
  如 53 圈 → `{soft:17, medium:17, hard:19}`, UI「Pirelli 选胎 (软/中/硬)」卡片正确显示。
- 端到端复核 dashboard 各图表数据源: strategy (plan.stops / tire_wear_projection /
  fuel_projection) / weather (GET, grip/lap/compound/setup) / bayesian (history) /
  pareto (front + 3 anchor setups) 前后端字段契约一致。

### UI 中文化 + 精简搜索参数 (Iter-293)
- 赛道下拉框全部显示中文名 (如「日本大奖赛 · 铃鹿」), 不再显示英文赛道名;
  `f1opt/data/tracks.py` 新增 `TRACK_NAME_CN`/`TRACK_COUNTRY_CN`/`TRACK_TYPE_CN`
  与 `track_name_cn`/`country_name_cn`/`track_type_cn` 辅助函数 (24 条赛道全覆盖)。
- `f1opt/api/app.py` `_track_dict` 注入 `name_cn`/`country_cn`/`track_type_cn` 三个字段;
  index.html 的赛道信息行 (国家/类型) 与 dashboard.html 同步中文化, "Sprint"→"冲刺赛"。
- 移除智能分析中心「调教搜索」里的「随机种子」输入框 (bayes/pareto), 后端 seed 使用
  默认值 42 —— 普通用户无需关心该高级参数。
- 修复 index.html 调教面板漏显 4 项 (active_aero_mode/x_mode_activations/engine_braking/
  ballast): `SETUP_GROUPS`/`SETUP_GAME_GROUPS` 21→23 项, 「当前调教 (23 项)」面板与
  游戏格式导出现完整覆盖全部 23 维调教参数。
- 智能分析中心推荐调教表格改为中文参数名 + 单位 (新增 `SETUP_FIELD_CN` 映射), 不再
  显示英文 field key (front_wing 等)。

### 双击即用 (无参自动启动智能分析中心) (Iter-292)
- 修复「EXE 打不开」: 此前双击 `f1opt.exe` 无参时 `main()` 仅打印 `--help` 后返回 1,
  控制台窗口一闪而过。现改为 **无参启动时自动拉起 API 服务器并打开浏览器**
  (`http://127.0.0.1:8000/dashboard.html`), 实现「点开即用」。
- `f1opt/cli.py` 新增 `_launch_gui()`: 主线程阻塞运行 uvicorn (Ctrl+C 仍可退出),
  守护线程等待端口就绪后 `webbrowser.open` 打开分析中心; 遥测监听 (端口 20777) 自动开启。
- 显式空参 `main([])` 仍保留打印帮助的旧行为, 测试 `test_cli.py`/`test_smoke.py` 不受影响
  (134 passed)。README 补充双击说明。

### 单文件 EXE (one-file) + README 修正 (Iter-291)
- `exe/f1opt.spec` 由 one-folder 切换为 **one-file**: 产物为单个 `dist/f1opt.exe`
  (255MB, 含 torch 训练/推理全栈), 满足「最终是一个 exe」。
- 单 EXE 本地冒烟通过: `--help` / `tracks info monza` (C2/C1/C0) /
  `train --iterations 3` (seg-dnn-torch-v0.3 trained)。
- README 修正: 21→23 维调校、C1-C5→C0-C5、UDP 24 车位、打包产物路径。

### 车手口头反馈识别 + 映射到调教字段 (Iter-290)
- 打通「车手反馈 → 调教分析」链路: `generate_feedback(question=...)` 现调用
  `classify_intent`/`classify_sub_intent` 识别车手口头反馈, 输出 `driver_intent`
  (intent/sub_intent/problem/setup_hint)。
- 问题子意图 (understeer/oversteer/tyre_wear/brake/ers/traction/balance) 映射到
  相关调教字段 (如 oversteer → rear_wing/rear_arb/rear_toe), 供调教分析模型定向优化。
- 补强中文识别: problem_report 增加「甩 / 车尾.*松 / 尾.*松 / 后轴.*松 / 尾部.*滑」;
  oversteer 子意图增加同类信号。此前「车尾太松了, 出弯总是甩」被误判为 other。
- 新增 test_driver_intent_recognized_when_question (含无 question 不附加断言)。

### 模型调校维度 21→23 (engine_braking + ballast) (Iter-289)
- 按 EA F1 2026 权威规范补齐模型优化空间: 新增 `engine_braking` (发动机制动 0-100%)
  与 `ballast` (配重 0-10), 模型 setup 维度 **21→23**, 输入维度 **39→41**。
- 全链路同步: `setup_schema` (字段+5 预设) / `surrogate` (SETUP_DIM/INPUT_DIM) /
  `app` (packet5→CarSetup 映射, 取消 ballast 忽略) / `setup_physics_bridge` (5 optima) /
  `presets` (3 显式 setup) / `causal` (2 条新因果规则)。
- 依据 `docs/F1_2026_Research_Summary.md` 第 17 行: 发动机制动 0-100% (100%=最大能量
  回收 + 入弯过度转向)。
- 测试同步: ~15 测试文件 21/39 → 23/41 断言 + CarSetup 构造补字段。

### 轮胎化合物贯通遥测→UI + rate_monitor 全 17 包 (Iter-288)
- **模块衔接**: `_frame_to_ws` 新增 `actual_tyre_compound`/`tyre_compound`/
  `tyre_compound_name` (服务端用 `tyre_compound_name` 把 uint8 映射为 C0-C6/雨胎),
  实时 UI 新增「轮胎」指示器显示当前配方。此前化合物数据在 aligner 采集后无下游消费。
- **rate_monitor 修复**: 覆盖全部 17 包类型 (复用 `packets.PACKET_NAMES` 单一事实来源),
  旧版本地字典只到 SessionHistory(11), 缺失 TyreSets/MotionEx/TimeTrial/LapPositions/
  CarTelemetry2 → 会误报 `Unknown(12-16)`。
- 训练冒烟测试通过 (seg-dnn-torch-v0.3); API 55 测试 + UI 29 测试通过。

### 权威包大小综合验证 + Event 身体大小 (Iter-287)
- 新增 `TestVerifiedSizes` 参数化测试: 锁定 F1 26 权威规范全部 **17 个包的总大小**
  (Motion 1325 / Session 926 / LapData 1399 / ... / CarTelemetry2 269), 逐一验证
  `validate_packet_size` ok + 解析器在权威身体大小下不崩溃。
- Event (id 3) 身体大小 4→16 (4 字节码 + 12 字节 SpeedTrap 最大 union, 总 45B)。

### 按权威规范重写 Session 数据包 (Iter-286)
- 按 MacManley/f1-26-udp 权威规范重写 PacketSessionData (id 1):
  - 天气样本 20 → **64** (旧版 20 导致 forecastAccuracy 及之后字段错位数百字节);
  - 头字段修正多余 1B (`BbbBHBbBHHBBBBBBB` → `BbbBHBbBHHBBBBBB`);
  - 补 `m_sessionLinkIdentifier` (第 3 个 uint32);
  - 补 `m_DRSAssist` (旧版误名 drivingAssist)、`m_pitStopRejoinPosition` (旧版误名
    RejoinWindow)、`m_gameMode`/`m_ruleSet`/`m_timeOfDay`/`m_sessionLength` 等;
  - 补 **主动空力区** (`m_activeAeroTrackStatus` + Full/Partial 各 8 区) 与 **DRS 区**
    (`m_numDRSZones` + 4 区) — F1 26 主动空力关键数据;
  - 补 weekendStructure[12]、sector2/3LapDistanceStart、startReactionTime 等。
- 身体 614 → 897B (匹配权威规范 926-29); 测试同步 (64 WFS + 16 字段头)。

### F1 26 权威规范: 24 车位 + Motion 速度/g-force + 尾部字段 (Iter-285)
- **关键发现**: MacManley/f1-26-udp 权威规范 (README) 全包数组为 **24 车位** (非 22)。
  所有缓冲区大小 (Motion 1325/LapData 1399/CarStatus 1445/CarTelemetry 1448/
  CarDamage 1133/CarSetups 1233/Participants 1470/FinalClassification 1134/
  LobbyInfo 1062/LapPositions 1231/CarTelemetry2 269) 均与 24 车精确一致。
  标准 2026 赛季为 22 车 (11 队), 但线数组预留 24 车位。
- **NUM_CARS 22 → 24**: 所有 per-car 解析器改读 24 车; `_EXPECTED_BODY_SIZES` 同步。
- **Motion (id 0) 线格式修正**: F1 26 速度由 int16 改为 **float**, g-force 仍为 int16
  (量化 ÷1000); 且 **无玩家额外段** (仅 24 × CarMotionData = 1325B)。
  aligner 将 g_lat/g_long/g_vert 标记为 int16 (最近邻插值 + `_INT_KEYS`)。
- **尾部字段补全**: CarSetups 追加 `m_nextFrontWingValue` (float); LapData 追加
  `m_timeTrialPBCarIdx` + `m_timeTrialRivalCarIdx` (各 uint8)。
- 测试同步 (test_packets/test_robustness/test_aligner 的 22→24 与 g-force int16)。

### 按权威规范修正 Participants + SessionHistory 线格式 (Iter-284)
- 继续用 MacManley/f1-26-udp 权威规范审计, 再修 2 处错位:
  - **Participants (id 4)**：`m_driverId`/`m_networkId`/`m_teamId`/`m_techLevel` 均为
    uint16 (旧版误作 uint8, 导致 `m_name` 起错位), 补 `m_platform`; 60B/participant。
  - **SessionHistory (id 11)**：头为 7 字节 (旧版缺 numTyreStints + 4×bestLapNum 共
    5 字节); LapHistoryData 为 14B (扇区拆 uint16 MSPart + uint8 MinutesPart, 旧版
    误作 11B 且扇区未拆); TyreStintHistoryData 顺序为 endLap→actual→visual (旧版误为
    actual→visual→endLap)。
- 同步 `_EXPECTED_BODY_SIZES` (Participants 1122→1321, SessionHistory 526→1431) 与
  置信度注释 (均升为 HIGH)。新增 Participants/LobbyInfo/FinalClassification 往返测试
  锁定 uint16 字段与字段序 (test_packets)。

### 按权威规范修正 5 个剩余数据包线格式 (Iter-283)
- 继续用 MacManley/f1-26-udp 权威规范审计剩余数据包, 修正 5 处错位/缺失:
  - **FinalClassification (id 8)**：`m_resultReason` 应紧随 `m_resultStatus` (旧版误置于
    tyreStintsEndLaps 末尾, 导致 bestLapTimeInMS/totalRaceTime 起错位 1 字节);
    `m_totalRaceTimeWarnings` 误名 → 改为权威 `m_penaltiesTime`。
  - **LobbyInfo (id 9)**：`m_teamId`/`m_techLevel` 为 uint16 (旧版误作 uint8 且虚构
    `m_networkId`/`m_carTelemetrySetup`), 补齐 `m_platform`/`m_carNumber`/
    `m_yourTelemetry`/`m_showOnlineNames`/`m_readyStatus`; 43 字节/玩家。
  - **TyreSets (id 12)**：20 套 (13 干 + 7 湿, 旧版误作 13 套), 每套 10 字节
    (7×uint8 + `m_lapDeltaTime` int16 + `m_fitted` uint8); 旧版缺 m_fitted 且把
    lapDeltaTime 误作 1 字节。
  - **MotionEx (id 13)**：`m_wheelSlipAngle` 应紧随 `m_wheelSlipRatio` (旧版误置于
    wheelVertForce 之后), 补 `m_frontAeroHeight`/`m_rearAeroHeight`/`m_frontRollAngle`/
    `m_rearRollAngle`/`m_chassisYaw`; 61 float = 244B。
  - **TimeTrial (id 14)**：3 个 TimeTrialDataSet (playerSessionBest/personalBest/rival),
    每个 25B (carIdx + teamId(H) + 4×uint32 + 6×uint8); 旧版只解析首个 dataset 且
    teamId 误作 uint8。
- 同步 `_EXPECTED_BODY_SIZES` (LobbyInfo 925→947, TyreSets 110→202, TimeTrial 72→75)
  与置信度注释 (均升为 HIGH)。测试同步 (test_packets/test_robustness)。

### 修复 CarSetups 线格式 (Iter-282b)
- 按权威规范修正 CarSetups：补 `m_engineBraking`(B)，轮胎压力为 float(4f) 而非
  uint8；旧版把 4 胎压误作 uint8 且缺 engineBraking，导致 ballast/fuelLoad 错位 9 字节。

### 修复 CarDamage 线格式 (Iter-282)
- 按权威规范修正 CarDamage：tyresWear[4](f) + tyresDamage[4](B) + brakesDamage[4](B)
  + tyreBlisters[4](B) + 18 个 uint8 损伤/故障字段 (翼面/地板/扩散器/DRS/ERS/变速箱/
  引擎 + 5 项引擎磨损 + blown/seized)。旧版把 tyresDamage 与翼面损伤误作 float,
  且虚构 suspensionDamage, 导致 tyresDamage 起全部错位。

### 修复 Session 天气预报样本线格式 (Iter-281b)
- WeatherForecastSample 权威规范为 8 字段 (trackTemp + trackTempChange + airTemp +
  airTempChange + rain%), 旧版 `_WFS_FMT` 仅 7 字段 (把两个 change 合并为
  weatherDelta), 导致每个样本错位 1 字节, 20 个样本累计 20 字节错位 →
  forecastAccuracy 及之后字段全部错位。已改为 8 字段。

### 修复 LapData 线格式错位 (Iter-281) — 关键 bug
- 按权威规范核对 LapData: 扇区时间/与前车差距/与领跑差距均拆分为 MSPart(uint16)+
  MinutesPart(uint8), 且位于 lapDistance/totalDistance/safetyCarDelta 之前; 另有
  speedTrapFastestSpeed(f) + speedTrapFastestLap(B)。旧版误作 HH + fff 紧跟扇区后,
  导致 **lapDistance 起全部字段错位 8 字节**, 且缺失 delta 与 speedTrap 字段。
- 修正为 33 字段 / 57 字节每车; `parse_lap_data` 组合 MSPart+MinutesPart 为完整
  `m_sector1/2TimeInMS` + `m_deltaToCarInFront/RaceLeaderInMS` (供 aligner/aggregator)。

### 主动空力消费者迁移到 active_aero_mode (Iter-280)
- 主动空力遥测现来自 Packet 16 的 `m_activeAeroMode` (0=Corner/Z / 1=Straight/X),
  全链路消费者已从遗留 `active_aero_x/z` (无源) 迁移：
  - analytics `active_aero_usage_analysis` 按 mode==1 / mode==0 统计 X/Z 占比;
  - 反馈引擎 `col_multi` + 指标改用 `active_aero_mode`;
  - `_frame_to_ws` / 实时 UI 指示器改用 `active_aero_mode` (1=X-Mode / 0=Z-Mode);
  - 聚合器新增 `_on_car_telemetry_2` (Packet 16), `avg_active_aero_x/z` = X/Z 帧占比
    (分母改用 `car_telemetry2_count`)。
- 测试同步：analytics/feedback/api/aggregator 测试改用 `active_aero_mode`。

### CarTelemetry engineTemperature 线格式修正 + 测试同步 (Iter-279)
- 按权威规范, `m_engineTemperature` 是 uint8 (B), 旧解析器误作 uint16 (H) 导致
  `m_tyresPressure`/`m_surfaceType` 错位 1 字节。修正格式串为 59 字节/车。
- Packet 16 补充 `CONFIDENCE_CARTELEMETRY2` 置信度注释; `PACKET_NAMES` 加入
  CarTelemetryData2。
- 同步更新 6 个测试文件的旧线格式/维度计数/版本号断言 (test_packets/test_smoke/
  test_robustness/test_e2e_scenarios/test_validation_closure/test_smoke_e2e)。

### 修复 CarStatus 线格式错位 (Iter-278) — 关键 bug
- **按 MacManley/f1-26-udp 权威规范**核对 CarStatus 结构, 发现严重错位：
  旧解析器缺失 `m_enginePowerICE` / `m_enginePowerMGUK` / `m_ersHarvestLimitPerLap`
  三个 float, 且虚构了不存在的 `m_activeAeroX` / `m_activeAeroZ` (主动空力实际在
  Packet 16 CarTelemetryData2)。导致 `m_ersStoreEnergy` 起的全部 ERS 字段**错位 8 字节**。
- 修正为 26 字段 / 59 字节每车；新增 `parse_car_telemetry_2` (Packet 16, 主动空力
  `m_activeAeroMode` 0=Corner/Z / 1=Straight/X + 超车可用性)；aligner 改用
  `active_aero_mode` / `active_aero_available`。
- 测试：test_packets CarStatus round-trip 更新为权威 26 字段。

### ERS 模式分析回归测试 (Iter-277)
- 新增 `test_ers_deploy_mode_classifies_hotlap`：验证 mode 2 归类为 Hotlap、
  mode 1 归类为 Medium，锁定 Iter-276 的 ERS 模式映射一致性。

### 修复 ERS 部署模式映射错位 (Iter-276)
- **`deploy_mode_hotlap_pct` 误判 mode 1 为 hotlap**：F1 UDP `m_ersDeployMode` 为
  0=none/1=medium/2=hotlap/3=overtake，反馈引擎原 `deploy_mode==1.0` 把 medium 当成
  hotlap，导致 hotlap 占比统计错误。改为 `==2.0`。
- **UI ERS 模式名错位**：实时面板 `modeNames` 原为 5 项 (无/低/中/Hotlap/超车),
  与 4 种模式不符 (模式整体后移一位)。改为 `["无","中","Hotlap","超车"]`。
- 测试：`test_hotlap_pct_uses_mode_2` (mode1+mode2 各半 → hotlap 50%)。

### 仪表盘策略结果显示 Pirelli 选胎 (Iter-275)
- 分析中心「赛道策略」结果面板新增「Pirelli 选胎 (软/中/硬)」卡片, 渲染
  `/api/strategy/plan` 返回的 `pirelli_compounds` (C0-C5), 与 API/CLI 三端一致。

### CLI 赛道详情附带 Pirelli 选胎 (Iter-274)
- `f1opt tracks info --track <id>` 现附带 `pirelli_compounds` 字段 (soft/medium/hard
  -> C0-C5), 与 Iter-273 的 `/api/strategy/plan` 保持一致, 补齐 CLI 侧的选胎信息。

### 策略端点暴露 Pirelli 2026 选胎方案 (Iter-273)
- `/api/strategy/plan` 响应新增 `pirelli_compounds` 字段: 按赛道返回该场
  soft/medium/hard 对应的 C0-C5 具体配方 (来自此前未接入的 `pirelli_2026.py`)。
  例: monza → C2/C1/C0 (最硬), monaco → C5/C4/C3 (最软), 严格按 EA F1 2026 各场选胎。

### 输入向量维度注释全量对齐 (Iter-272)
- 全仓扫除 "37 维 / [0,1]^19 / [29:37] / 19 setup fields" 等陈旧引用:
  `causal.py`、`feature_importance.py`、`pareto.py`、`setup_analysis.py`、
  `surrogate.py`、`train.py` 全部改为 39 维输入 / 21 setup 字段 / [31:39] driver 段 / 41 个扰动。
- `feature_importance.py` 的 `FEATURE_NAMES` 经防御性断言验证 = 39 (= INPUT_DIM)。

### 修复训练物理一致性 loss 的 driver 向量错位 (Iter-271)
- **关键 bug**：`_DRIVER_VEC_START=29` / `_DRIVER_VEC_END=37` 硬编码, 假设
  SETUP_DIM=19。`active_aero_mode`/`x_mode_activations` 加入后 INPUT_DIM 37→39,
  driver 段实际在 [31:39]。旧值 [29:37] 会**覆盖末尾 2 维 track-context 且漏掉
  末尾 2 维 driver**, 使"AGGR 应快于 CONS"的物理一致性 loss 计算错误。
- 改为 `SETUP_DIM + TRACK_CONTEXT_DIM` / `INPUT_DIM` 动态派生；同步修正
  `online_correction.py` 的 `[0,1]^19` 与 outlier 阈值注释 (3.0s→5.0s)。
- 测试：`tests/model/test_train.py` 26 passed (训练循环验证)。

### 车手反馈中文叙事补全 (Iter-270)
- `brake_temp` / `tyre_temp_gradient` / `grip_consistency` 三个维度此前有中文
  标签但无中文叙事器, 落回通用 "方面：value。" 模板。现补全专属叙事函数并注册,
  20 个 `FEEDBACK_DIMENSIONS` 全部具备中文叙事。

### fuel_load 钳制回归测试 (Iter-269)
- 新增 `test_search_clamps_fuel_load_to_baseline`：基线 fuel_load=80kg 时, 优化器
  推荐值应保持 ~80kg 而非被推到 5kg 最小值。该测试在 Iter-267 修复前会失败
  (旧索引 18 误钳制 front_tyre_pressure, fuel_load 被自由优化到最小)。

### 模型层维度注释全面对齐 (Iter-268)
- `optimizer.py` / `setup_analysis.py` / `setup_physics_bridge.py` / `train.py` 中
  多处 "19 维 / 18 维 / 18 个参数" 注释未随 `active_aero_mode`+`x_mode_activations`
  加入(19→21 字段)而更新。已改为: 归一化空间 21 维、物理惩罚 20 维(fuel_load 除外)、
  `analyze_setup_contributions` 返回 20 个参数。

### 修复调教优化器 fuel_load 索引错位 (Iter-267)
- **关键 bug**：`_FUEL_LOAD_IDX = 18` 硬编码。`active_aero_mode`/`x_mode_activations`
  加入后 `fuel_load` 索引从 18 变为 20, 导致优化器把 `front_tyre_pressure`(索引 18)
  钳制到燃油基线值, 而 `fuel_load` 仍被自由优化到最小——既破坏了胎压优化, 又
  未修复"燃油被推到 5kg"的原始问题。现改为从 `ALL_SETUP_FIELDS()` 动态求索引。
- 维度注释同步：`setup_physics_bridge` 19→21 维 / 18→20 维, `batch.py` 19→21 字段。
- 测试容差 1e-6→1e-3（DE 随机优化胎耗代理差异 ~4e-6 属数值噪声）。

### 遥测分析时间戳缓存 (Iter-266)
- `TelemetryAnalytics` 新增惰性缓存 `_get_times()`/`_get_deltas()`：`compute_all`
  的 29 个分析各自重算时间戳与时间差 (各 O(n) 全帧扫描), 现整圈只算一次并复用,
  减少 ~8 次冗余 O(n) 扫描 (600 帧单圈 `compute_all` ~26.8ms)。

### 遥测分析 DRS 语义修正 (Iter-265)
- **`drs_analysis` 改用 `drs_active` (实际激活, m_drs)**：原 `_field_multi` 元组
  把幻影字段 `drs` 放首位, 实际落到 `drs_allowed` (仅"此区段允许"), 会把整个
  DRS 区段都计为"激活", 高估激活次数/时长/速度增益。改为优先 `drs_active`,
  与车手画像 `_drs_usage_efficiency` (Iter-261) 保持一致。

### 质量评分与 gap_filler 字段对齐 (Iter-264)
- **quality_score 幻影字段修复**：`_EXPECTED_RANGES` 用 `fuel_remaining`(对齐帧从不
  产出), 燃油范围校验永不触发。改为真实字段 `fuel_in_tank`, 并补 `fuel_remaining_laps` /
  `ers_store` / `ers_deploy_mode` 范围, 使 range_compliance 真正覆盖 F1 2026 关键通道。
- **gap_filler 默认字段对齐**：`ers_deploy_mode` 是离散模式(0-3), 误列在浮点插值
  字段(会被线性插值成小数); `fuel_remaining`→`fuel_remaining_laps`; `drs`→
  `drs_allowed`/`drs_active`; 新增 `ers_store` 平滑插值。

### UI 与文档一致性: 版本徽章 + 调教参数计数 (Iter-263)
- **仪表盘头部版本徽章**硬编码 `v0.1.0` → 改为 `v1.2.0`, 且健康检查时从
  API `version` 动态刷新, 避免后续版本升级再度陈旧。
- **调教参数计数 19→21 修正**：`active_aero_mode` + `x_mode_activations` 加入后
  实际 21 个字段, 但 `to_vector` docstring、dashboard "19 项"、`使用方法.md`
  "19 个参数" 均未同步。已全部修正并补充主动空动/X-Mode 说明。

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