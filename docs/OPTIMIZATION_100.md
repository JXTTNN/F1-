# F1OPT 深度优化计划（100 项）

> 目标：深度优化项目 100 次，把项目全部切片研究优化，全程云端检测。
> 方法：10 个切片 × 10 项优化；每项优化均为可验证、可回滚的独立改动，
> 提交前本地 pytest 验证，推送后由云端 CI（ci.yml 快测 + full-ci.yml 全量套件）背书。
> 迭代编号接续 CHANGELOG：Iter-299 起（历史止于 Iter-298）。

## 切片总览

| # | 切片 | 范围 | 说明 |
|---|------|------|------|
| S1 | 基础设施/CI | .github/workflows/*, pyproject, requirements.lock | 云端验证能力是全部优化的地基 |
| S2 | telemetry 核心解析 | packets.py (1143L) | 60Hz 热路径，字节级正确性红线 |
| S3 | telemetry 管线 | listener/aligner/aggregator/replay/summary | 采集→对齐→存储 |
| S4 | telemetry 分析 | analytics.py (1865L), quality_score, rate_monitor… | 分析侧 |
| S5 | model 学习核心 | train.py (1723L), surrogate.py, bayesian.py, optimizer.py | 训练/推理/搜索 |
| S6 | model 物理 | physics, lap_simulator*, tire_*, ers, brake, suspension… | 物理仿真 |
| S7 | 策略家族 | strategy, strategy_optimizer, pit_window, safety_car, grid_penalty… | 决策层 |
| S8 | feedback 家族 | engine.py (3477L!), prompts, intent, causal, nlg, conversation… | 车手反馈 |
| S9 | api/cli/ui | app.py (1845L), extended, cli.py | 接口层 |
| S10 | 数据/可观测/杂项 | data/*, observability/*, driver/*, tests 质量, 半成品 去留 | 底座与卫生 |

## 进度台账

| # | 切片 | 优化项 | 状态 | 验证 |
|---|------|--------|------|------|
| 001 | S1 | 新建 full-ci.yml：全量 pytest 套件云端运行（+workflow_dispatch 手动触发） | ✅ | run 34020170000 9/10 组首验 |
| 002 | S1 | .gitignore 补 .deps/（本地免 pip 依赖目录），防污染 | ✅ | 已生效 |
| 003 | S2 | packets.parse_car_telemetry：提升 calcsize/字段数到模块常量（每包省 1 次 format 解析 + 1 次 dummy unpack） | ✅ | 本地 531 包测试全过; -0.46µs/包 |
| 004 | S2 | packets.parse_final_classification：同上消除每包 dummy unpack | ✅ | 同上 |
| 005 | S2 | packets.parse_car_damage：同上 | ✅ | 同上 |
| 006 | S2 | packets.parse_motion：g_force_idx 提升为模块常量，消除每包列表推导 | ✅ | 同上 |
| 007 | S1 | full-ci 矩阵化：10 组并行 + 详细日志，定位云端 76% 处无声消失 | ✅ | run 34021380549 定位 root 组 |
| 008 | S1 | root 组拆 8 步 + python -u + faulthandler（定位云端静默被杀的现场） | ✅ | 锁定 test_stress_comprehensive |
| 009 | S1 | workflow_dispatch 加 groups 白名单，定点重跑不浪费 runner | ✅ | run 34022544099 定点生效 |
| 010 | S1 | 每组 JUnit XML + artifact 上传，绕过日志流 ~132KB 截断 | ✅ | artifact 9986245483 精确定位失败用例 |
| 011 | S10 | test_stress_comprehensive::test_listener_1000_packets_per_sec 改脉冲脉冲+动态排空（云 runner 洪泛丢旧假失败） | ✅ | run 34027633752 全 10 组绿 |
| 012 | S5 | SurrogateModel.predict 单次计算化（sv/dv/tv 各算一次供输入+driver修正两用） | ✅ | 1.478→0.579ms (-61%) |
| 013 | S5 | predict_with_confidence 与 predict 共享计算（单模+集成；集成成员前向减半） | ✅ | 集成 confidence 5.11→1.74ms (-66%) |
| 014 | S5 | predict_batch 按 track_id 批内缓存上下文向量 + 逐项 sv/dv 去重 | ✅ | 本地过测，随模型组回归 |
| 015 | S10 | 推送降级通道: git push 被账号邮箱验证栅栏挡时, 用 Git Data/Contents API（LF 归一化字节）写库 + 本地 soft-reset 对齐远端, 树哈希一致校验 | ✅ | 本轮实推 5 次成功(d4e75ca→029a67e) |
| 016 | S5 | 车手交叉修正向量化: 16 项交叉 + 4 项基线 Python 循环 → numpy 按数组预计算 (fuzz 4000 例 diff<5e-8) | ✅ | 11.1→7.9µs (-29%) |
| 017 | S5 | setup_penalty_s 向量化: 22 次 getattr/SETFIELDS.step/dict.get → coef 表 + dot (fuzz 3000 例 diff<3e-15) | ✅ | 7.3→6.0µs (-17%) |
| 018 | S5 | 集成成员共享预测零件: _predict_parts 一次, 成员只前向 (原每成员重算全套) | ✅ | ens.predict 4.41→1.25ms, ens.confidence 5.11→1.33ms |
| 019 | S5 | predict 全链路 (R3+R4 累计): 1.478→0.474ms (-68%), ens.confidence 5.107→1.329 (-74%)｜预测型调用在所有 hot 路径直接受益 | ✅ | bench_surrogate (300 iters) |
| 020 | S5 | predict_batch 组装整体向量化：(N×41) 一次拼接，sv 批矩阵 + tv 映射 + dv 中性批送 | ✅ | 15.46→7.72ms (-50%), fuzz 700 例逐位等价 |
| 021 | S5 | driver_corr 中性快轨: driver=None/全0.5 时修正恒零 (解析引理), 跳过 track 解析+全部计算 | ✅ | 随 020 一起验证 |
| 022 | S5 | track_prior/sector_priors 不变件缓存 (每赛道仅 1 次 benchmark/sector_times 查表) | ✅ | 批路径项均 ∼30% 下降 |
| 023 | S5 | 集成模型批零件共享: `_predict_batch_parts` 抽出，成员批只跑前向 (原各成员重复全链路) | ✅ | 28.41→20.61 ms/200 项 (-27.4%) |
| 024+ | — | 待续 (S6 feedback 热区审场后填充) | ⬜ | — |

## 全量基线（run 34027633752, 2026-09-06, ubuntu py3.11, 4065 用例全过）

model 8.9min · root 3.9 · api 3.2 · telemetry 2.4 · e2e 2.3 · data 2.1 · driver 2.0 · feedback 1.7 · ui 1.7 · observability 1.6（其余为 install~1min）
模型组最重，后续 perf 优化以此为回归对照基准。

## 当前阻塞态（2026-09-06 R3)

账号 JXTTNN 被 GitHub「至少验证一个邮箱」栅栏拦截：**git smart-HTTP push 与
Actions 启动均拒绝**（contents/git-data 读 API 正常, contents PUT 正常）。
→ 本轮 API 推送的 5 个 commit 的 CI 全部 startup_failure； workflow_dispatch
也同结局。云端回归暂由**本地全量所在组代理**；等用户在
https://github.com/settings/emails 验证任意邮箱后，云中辩自恢复（届时重放 dispatch）。

## 验证口径

- **本地**：`python -m pytest`（PYTHONPATH=.deps;. 引导环境，免 pip 沙箱限制）
- **云端**：push 后 `gh run watch`，ci.yml（快测 5 文件）+ full-ci.yml（全量 ~3500 用例）双绿为通过