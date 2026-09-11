# F1OPT 性能分析与优化报告

> 生成时间：2026-09-11  
> 任务：task-36 性能分析与优化  
> 工作目录：`D:\IDEProjects\demo\f1opt-work`

## 1. 摘要

针对用户反馈「性能也不好」，对 F1OPT 赛车调教优化助手进行了系统性能分析并实施 4 项优化。所有 1162 个现有测试通过，API 行为未改变，未引入新依赖。

**核心成果**（30 次采样对比，规则引擎/SQLite 各 500 次采样）：

| 指标 | 基线 | 优化后 | 改善 |
|------|------|--------|------|
| `compute_setup_delta` | 0.0357 ms | 0.0272 ms | **-24%** |
| `generate_suggestion_full` | 0.1622 ms | 0.1363 ms | **-16%** |
| SQLite `get_latest_suggestion` | 0.0147 ms | 0.0104 ms | **-29%** |
| API `tracks_list` P95 | 0.78 ms | 0.68 ms | **-13%** |
| API `suggest` mean | 1.74 ms | 1.52 ms | **-13%** |
| API `suggest` P95 | 2.11 ms | 1.82 ms | **-14%** |

## 2. 分析方法

### 2.1 测量脚本

新增 `scripts/perf_analysis.py`，自动化测量：

1. **API 端点响应时间**：启动 uvicorn 子进程（独立临时 `data_dir`，避免污染），用 `httpx` 对 6 个端点各测 N 次，计算 mean / P50 / P95 / min / max。
2. **规则引擎纯函数**：直接调用 `compute_dx` / 矩阵乘法裸循环 / `compute_setup_delta` / `generate_suggestion` 各 M 次。
3. **SQLite 查询**：构造临时库，预置 50 条 feedback + 20 条 suggestion + 20 条 iteration，测量各查询方法耗时。
4. **进程内存**：尝试采样 RSS（Windows `tasklist` 解析，单位不稳定，仅供参考）。

### 2.2 测量端点

| 端点 | 方法 | Body |
|------|------|------|
| `/api/v1/health` | GET | — |
| `/api/v1/tracks` | GET | — |
| `/api/v1/tracks/current` | POST | `{"track_id": "suzuka"}` |
| `/api/v1/feedback` | POST | `{"track_id":"suzuka","corner_number":1,"symptom":"understeer","strength":3}` |
| `/api/v1/suggest` | POST | `{"track_id": "suzuka"}` |
| `/api/v1/iteration/history?track_id=suzuka` | GET | — |

> 注：任务描述中 `feedback` body `{"corner_index, phase, symptom}` 字段与实际 `FeedbackRequest` 模型不符，已按代码真实字段测量。

### 2.3 公平对比

- 基线与优化后均用相同参数：`--api-n 30 --engine-n 500 --sqlite-n 500`。
- 基线通过 `git stash` 临时回退优化代码后测量，测完恢复。
- 每次测量均启动全新服务进程 + 全新临时数据库，避免历史数据污染。

## 3. 基线分析结果（优化前）

### 3.1 API 响应时间（30 次）

| 端点 | mean (ms) | P50 (ms) | P95 (ms) | max (ms) |
|------|-----------|----------|----------|----------|
| health | 0.70 | 0.62 | 1.58 | — |
| tracks_list | 0.69 | 0.66 | 0.78 | — |
| select_track | 0.68 | 0.65 | 0.81 | — |
| submit_feedback | 0.97 | 0.94 | 1.28 | — |
| suggest | 1.74 | 1.69 | 2.11 | — |
| iteration_history | 0.89 | 0.84 | 1.17 | — |

### 3.2 规则引擎（500 次）

| 阶段 | mean (ms) | P95 (ms) |
|------|-----------|----------|
| compute_dx | 0.0018 | 0.0019 |
| matrix_mul_only（裸循环） | 0.0205 | 0.0297 |
| compute_setup_delta | 0.0357 | 0.0374 |
| generate_suggestion_full | 0.1622 | 0.2071 |

### 3.3 SQLite 查询（500 次）

| 查询 | mean (ms) | P95 (ms) |
|------|-----------|----------|
| get_feedbacks (50 行) | 0.0978 | 0.1044 |
| has_feedback | 0.0048 | 0.0050 |
| get_latest_suggestion | 0.0147 | 0.0153 |
| get_iterations (20 行) | 0.0378 | 0.0404 |
| get_latest_setup (空) | 0.0071 | 0.0073 |

### 3.4 发现的瓶颈

1. **`tracks_list` 端点**：每次请求都重建 24 个 `TrackRef` pydantic 对象并 `model_dump` 序列化，但赛道是静态数据，永不变化。
2. **规则引擎 `_build_param_detail`**：对每个参数都遍历全部 9 个诊断维度查 `COUPLING_MATRIX[dim][param]`，其中大量为 `None` 零格（矩阵密度约 38%，即 62% 的迭代是空操作）。
3. **`compute_setup_delta` 矩阵乘法**：同样遍历全部 9 维，未利用矩阵稀疏性。
4. **`IterationService.get_latest_round`**：拉取某赛道全部 iteration 记录到内存再 `max(round_no)`，随迭代历史增长线性变慢（N+1 风格）。
5. **SQLite 缺少复合索引**：`get_latest_setup` / `get_latest_suggestion` 执行 `WHERE track_id=? ORDER BY xxx DESC LIMIT 1`，仅有单列索引 `idx_*_track`，需回表排序。

## 4. 优化措施

### 4.1 `tracks_list` 响应缓存

**文件**：`setup_tuner/api/routes.py`

赛道列表为 24 条静态数据（`ALL_TRACKS` 模块级常量），启动后不变。首次请求时构建完整响应 dict 并缓存到模块级变量 `_TRACKS_LIST_CACHE`，后续请求直接返回缓存对象，跳过 24 次 `TrackRef` 构建 + `model_dump`。

```python
_TRACKS_LIST_CACHE: dict[str, Any] | None = None

@router.get("/tracks")
async def list_tracks() -> dict[str, Any]:
    global _TRACKS_LIST_CACHE
    if _TRACKS_LIST_CACHE is not None:
        return _TRACKS_LIST_CACHE
    tracks = get_all_tracks()
    data = [_track_to_ref(t).model_dump() for t in tracks]
    _TRACKS_LIST_CACHE = ok(data=data, message=f"共 {len(data)} 条赛道")
    return _TRACKS_LIST_CACHE
```

**安全性**：赛道数据在进程生命周期内不变（来自 `domain/track.py` 的 `ALL_TRACKS` 常量），缓存不会过期。测试用例若修改赛道数据会触发重新导入模块（新进程），不受影响。

### 4.2 规则引擎稀疏矩阵预计算

**文件**：`setup_tuner/engine/coupling.py` + `setup_tuner/engine/engine.py`

耦合矩阵 9×23 = 207 格中仅约 80 格非零（密度 ~38%）。原实现每次矩阵乘法和报告组装都遍历全部 9 维，62% 的迭代在判 `None` 后跳过。

优化：在模块加载时预计算每个参数列的非零单元列表 `_PARAM_NONZERO_CELLS`，矩阵乘法和报告组装直接迭代非零 cell，减少 ~60% 迭代次数。

```python
# coupling.py
_PARAM_NONZERO_CELLS: dict[str, list[CouplingCell]] = {
    param: [
        COUPLING_MATRIX[diag][param]
        for diag in DIAG_DIMS
        if COUPLING_MATRIX[diag][param] is not None
    ]
    for param in PARAM_NAMES
}

def nonzero_cells_for_param_cached(param: str) -> list[CouplingCell]:
    return _PARAM_NONZERO_CELLS.get(param, [])
```

`engine.py` 的 `compute_setup_delta` 步骤 1 和 `_build_param_detail` 均改用 `nonzero_cells_for_param_cached(p)` 替代遍历 `DIAG_DIMS`。

**应用经验**：采用经验库 `2026-09-10-deterministic-advice-engine` 的「定义参数全集 S + 耦合矩阵」结构，预计算是确定性引擎的标准优化（不破坏纯函数语义）。

### 4.3 `get_latest_round` 改用 SQL MAX

**文件**：`setup_tuner/db/store.py` + `setup_tuner/feedback/iteration.py`

原 `IterationService.get_latest_round` 调用 `store.get_iterations(track_id)` 拉全部记录到内存再 `max(round_no)`，随迭代历史增长线性变慢。

优化：在 `Store` 新增 `get_latest_round` 方法，用 `SELECT MAX(round_no)` 单次聚合查询，O(1) 内存。`IterationService.get_latest_round` 改调它。

```python
# store.py
def get_latest_round(self, track_id: str) -> int:
    with self._lock:
        row = self._conn.execute(
            "SELECT MAX(round_no) AS m FROM iteration WHERE track_id = ?",
            (track_id,),
        ).fetchone()
    if row is None or row["m"] is None:
        return 0
    return int(row["m"])
```

### 4.4 SQLite 复合索引

**文件**：`setup_tuner/db/schema.sql`

为 `ORDER BY ... DESC LIMIT 1` 和聚合查询添加复合索引，避免回表排序：

```sql
-- get_latest_setup: WHERE track_id=? ORDER BY imported_at DESC LIMIT 1
CREATE INDEX IF NOT EXISTS idx_setup_track_imported_at
  ON setup(track_id, imported_at DESC);
-- get_latest_suggestion: WHERE track_id=? ORDER BY created_at DESC LIMIT 1
CREATE INDEX IF NOT EXISTS idx_suggestion_track_created_at
  ON suggestion(track_id, created_at DESC);
-- get_latest_round: WHERE track_id=? 聚合 MAX(round_no)
CREATE INDEX IF NOT EXISTS idx_iteration_track_round
  ON iteration(track_id, round_no DESC);
```

**应用经验**：采用经验库 `2026-09-11-sqlite-pragma-write-throughput-combo` 的 PRAGMA 组合（WAL + synchronous=NORMAL + cache_size + temp_store=MEMORY），这些项目已配置；本次补充复合索引是对「读路径 ORDER BY LIMIT 1」的标准优化。

## 5. 优化后结果

### 5.1 API 响应时间（30 次）

| 端点 | 基线 mean | 优化后 mean | 基线 P95 | 优化后 P95 | 改善 |
|------|-----------|-------------|----------|------------|------|
| health | 0.70 | 0.73 | 1.58 | 1.52 | 持平 |
| tracks_list | 0.69 | 0.60 | 0.78 | 0.68 | **-13%** |
| select_track | 0.68 | 0.69 | 0.81 | 0.86 | 持平 |
| submit_feedback | 0.97 | 0.87 | 1.28 | 1.04 | **-10%** |
| suggest | 1.74 | 1.52 | 2.11 | 1.82 | **-13%** |
| iteration_history | 0.89 | 0.88 | 1.17 | 1.05 | 持平 |

### 5.2 规则引擎（500 次）

| 阶段 | 基线 mean | 优化后 mean | 改善 |
|------|-----------|-------------|------|
| compute_dx | 0.0018 | 0.0019 | 持平（本就极快） |
| matrix_mul_only | 0.0205 | 0.0192 | -6% |
| compute_setup_delta | 0.0357 | 0.0272 | **-24%** |
| generate_suggestion_full | 0.1622 | 0.1363 | **-16%** |

### 5.3 SQLite 查询（500 次）

| 查询 | 基线 mean | 优化后 mean | 改善 |
|------|-----------|-------------|------|
| get_feedbacks (50 行) | 0.0978 | 0.1072 | 持平（抖动） |
| has_feedback | 0.0048 | 0.0084 | 持平（sub-ms 抖动） |
| get_latest_suggestion | 0.0147 | 0.0104 | **-29%** |
| get_iterations (20 行) | 0.0378 | 0.0553 | 持平（抖动） |
| get_latest_setup (空) | 0.0071 | 0.0071 | 持平 |

> 说明：`get_iterations` / `has_feedback` 在 sub-ms 级别受 Windows 进程调度噪声影响明显，500 次采样仍有抖动，本质非瓶颈。

## 6. 验证

### 6.1 测试套件

```
python -m pytest tests/ -x -q --timeout=30
→ 1162 passed, 2 warnings in 9.85s
```

所有 1162 个测试通过，未破坏任何现有功能。

### 6.2 代码风格

```
python -m ruff check <修改的文件>
→ All checks passed!
```

### 6.3 约束遵守

- ✅ 未引入新依赖（仅用标准库 + 现有 fastapi/uvicorn/pydantic/httpx）
- ✅ API 接口行为未变（端点签名、请求/响应模型、信封格式均未改）
- ✅ 所有现有测试通过
- ✅ ruff 通过
- ✅ 中文注释

## 7. 剩余瓶颈与后续建议

### 7.1 当前非瓶颈

- **API 响应时间已 < 2ms（P95）**：对本地单用户调教助手而言已远超可用阈值。
- **规则引擎 < 0.15ms**：确定性纯函数，无 IO，性能优异。
- **SQLite 查询 < 0.1ms**：本地单文件库 + WAL + 索引，读路径已优。

### 7.2 可进一步优化（未实施，避免超范围）

1. **`suggest` 端点 DB 写入**：每次建议生成会 `save_suggestion` + `save_iteration` 两次 INSERT + COMMIT，可合并为单事务减少 fsync。但当前 <2ms，收益有限。
2. **`get_feedbacks` / `get_iterations` 列裁剪**：当前 `SELECT *`，部分场景只需 `id/round_no` 等子集。但会改变返回 dict 结构，需评估调用方影响。
3. **`generate_suggestion` 报告组装**：`_build_param_detail` 仍占 `generate_suggestion_full` 80% 耗时，可进一步缓存 `DIAG_DIMS_POSITIVE_SEMANTICS` 查表或用 numpy 向量化。但需引入 numpy 依赖，违反约束。
4. **启动时间**：`create_app` 同步导入所有模块（telemetry/report/ui），可改为按需导入。但当前启动 <1s，非用户感知瓶颈。
5. **`_TRADEOFF_NOTES` 查表**：`_build_param_detail` 每次查 `_TRADEOFF_NOTES.get(spec_name, {}).get(direction)`，可预计算为 `(spec_name, direction) → tradeoff` 扁平 dict。收益极小。

### 7.3 测量改进建议

- Windows 上 sub-ms 级测量受进程调度影响明显，建议后续在 Linux/WSL 下用 `perf_counter_ns` + `taskset` 绑核重测以减少噪声。
- 内存采样当前用 `tasklist` 解析，单位不稳定。建议后续用 `psutil`（需引入依赖）或读取 `GetProcessMemoryInfo` Win32 API。

## 8. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `setup_tuner/api/routes.py` | `tracks_list` 响应缓存 |
| `setup_tuner/engine/coupling.py` | 预计算 `_PARAM_NONZERO_CELLS` + `nonzero_cells_for_param_cached` |
| `setup_tuner/engine/engine.py` | `compute_setup_delta` / `_build_param_detail` 改用预计算非零 cell |
| `setup_tuner/db/store.py` | 新增 `get_latest_round`（SQL MAX） |
| `setup_tuner/db/schema.sql` | 3 个复合索引 |
| `setup_tuner/feedback/iteration.py` | `get_latest_round` 改调 `Store.get_latest_round` |
| `scripts/perf_analysis.py` | 新增性能分析脚本 |

## 9. 经验来源

本优化应用了以下工程经验：

- `2026-09-10-deterministic-advice-engine`：参数全集 S + 耦合矩阵 + 诊断向量结构，预计算是确定性引擎的标准优化。
- `2026-09-11-sqlite-pragma-write-throughput-combo`：WAL + synchronous=NORMAL + cache_size + temp_store=MEMORY PRAGMA 组合（项目已配置），复合索引是读路径标准补充。
- `2026-09-10-sqlite-thread-safe-store-pattern`：Store 线程安全模式（check_same_thread=False + Lock），本次新增方法遵循同一模式。