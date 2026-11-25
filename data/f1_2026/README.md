# F1 2026 遥测 + 调教数据包

本目录是本项目用于 **F1 2026** 的外部数据落地层：一边是真实 2026 赛季遥测的
**去标识化聚合**，一边是 F1 26 游戏版的 **调教 → 圈速** 配对。两边共用
`track_id`（与 `setup_tuner.domain.track` 的 24 站一致），可以直接 join。

> ⚠️ 隐私口径：仓库 `.gitignore` 明确规定「车手真实遥测与训练数据（个人数据，
> 绝不入库）」。因此**原始逐点遥测不入库**，只留在本地 `.ref/tracing_2026/`；
> 入库的只有赛道级聚合统计（无车手、无车号、无圈号），逐弯明细留在本地。

## 文件

| 文件 | 行数 | 入库 | 内容 |
|------|------|------|------|
| `track_profile.jsonl` | 24 | ✅ | 每站一行：遥测画像 + 调教参数分布 + 最优/典型圈速 |
| `manifest.json` | — | ✅ | 来源、许可、抓取时间、覆盖度、行数、sha256 |
| `telemetry_corners.jsonl` | 100 | ❌ 本地 | 6 站 × 弯道的真实遥测聚合：`n_laps` + 每项指标的 `median / p10 / p90` |
| `setups_f1_26.jsonl` | 899 | ❌ 本地 | F1 26 调教（20 项）+ 圈速 + 作者/车队/工况/来源链接 |

逐弯明细与逐条调教被 `data/f1_2026/.gitignore` 排除：它们体积较大、可由
`scripts/build_f1_2026_bundle.py` 重建，且源头是个人录制/公开页面，按仓库
「个人数据不入库」的既有口径只保留在本地。入库的 `track_profile.jsonl`
只含**赛道级**统计（不含车手、圈号、逐弯明细）。

> 想入库完整明细时，把 `data/f1_2026/.gitignore` 里对应两行删掉并重跑
> `python scripts/build_f1_2026_bundle.py` 即可（脚本会刷新 manifest 的
> `committed` 标记与 sha256）。

`telemetry_corners.jsonl` 一行示例（数值已四舍五入）：

```json
{"track_id":"monza","corner":1,"corner_class":"slow","n_laps":63,
 "min_speed":{"median":86.4,"p10":78.1,"p90":95.2},
 "entry_speed":{"median":242.7,"p10":190.4,"p90":280.9},
 "corner_time_s":{"median":5.86,"p10":5.52,"p90":6.31},
 "throttle_full_pct":{"median":41.2,"p10":25.0,"p90":58.3},
 "brake_pct":{"median":28.9,"p10":16.7,"p90":45.1},
 "gear_min":{"median":2.0},"gear_max":{"median":4.0},
 "max_lateral_g":{"median":2.41},"max_long_g":{"median":-2.18}}
```

指标口径：`entry_speed / min_speed / exit_speed / avg_speed` 单位 km/h；
`corner_time_s` 为该弯通过时间（秒）；`throttle_full_pct` 为全油门采样占比；
`brake_pct` 为刹车采样占比；G 值为绝对值峰值（`max_long_g` 取负峰）。

`setups_f1_26.jsonl` 一行示例：

```json
{"track_id":"melbourne","author":"Uzhegov","lap_time_ms":76790,
 "setup":{"front_wing":23.0,"rear_wing":10.0,"on_throttle_diff":100.0,...},
 "setup_field_count":20,"team":"Racing Bulls","session":"Time Trial",
 "conditions":"dry","game_year":2026,
 "source_url":"https://www.f1laps.com/f1-26/setups/australia/<uuid>/"}
```

## 数据来源与许可

| 来源 | 用途 | 许可 / 合规 |
|------|------|-------------|
| [TracingInsights/2026](https://github.com/TracingInsights/2026) | 真实 2026 逐圈逐点遥测（FastF1 标准化） | Apache-2.0；GitHub raw 公开文件 |
| [F1Laps · F1 26 Setups](https://www.f1laps.com/f1-26/setups/) | 调教 + 圈速配对 | 公开 HTML 页面；robots.txt `Allow: /`（`/api/` 与登录区不触碰）；默认 1.2s/请求限速 |

两个上游都不是 F1 官方，商标归 Formula One Licensing B.V.；本目录仅作研究/训练用途。

## 覆盖度

- 遥测：**6 / 24 站**（melbourne、monaco、silverstone、monza、singapore、spa），
  每站 Race 会话、每位车手最多 3 圈，2026 年检通过 0 条非 2026 记录。
- 调教：**24 / 24 站**，899 条 2026 干/湿配对（干 547 / 湿 352）。

之所以只取 6 站：全量 24 站逐点遥测约数百 MB～GB 级，不适合入库；6 站已覆盖
街道（monaco/singapore）、高速低下压（monza）、高下压（silverstone）、
混合（spa/melbourne）四类工况，足以做弯速画像与工况分层。

## 刷新方式

```bash
# 1) 抓真实遥测（可续传；缓存落 .ref/，不入库）
python scripts/fetch_official_2026.py --race "Italian Grand Prix" --session Race --max-laps 3
# 或全量（体积大，慎用）
python scripts/fetch_official_2026.py --full --max-minutes 120

# 2) 切成逐弯样本（data/training/ 已被 gitignore，属本地训练数据）
python scripts/build_corner_dataset.py --races "Italian Grand Prix" ...

# 3) 抓调教 + 圈速（可续传）
python scripts/fetch_setup_laps.py --all-slices

# 4) 生成可入库的聚合包（本目录）
python scripts/build_f1_2026_bundle.py
```

## 已知口径问题（构建时已处理）

1. **1 号弯零速帧**：上游 `build_corner_dataset.py` 的 1 号弯窗口包含记圈点
   的零速帧（实测 142 / 6380 条，其中 135 条在 1 号弯），会把 `entry/min/avg
   speed` 的 p10 拉成 0。本包聚合时**剔除**这类样本，并在
   `manifest.counts.corner_rows_dropped_zero_speed` 里如实记数；上游未改，
   训练侧若直接用 `data/training/corner_dataset.jsonl` 需自行过滤。
2. **样本数与 `n_laps`**：`n_laps` 是该弯有效样本数（车手×圈），不是整圈数；
   同一弯不同车手的窗口点数不同，已按样本计。
3. **调教单位**：与项目 20 项参数完全一致（翼片为档位、胎压 bar、车高 mm 等），
   可直接喂 `CarSetup`。

## 与引擎的衔接建议

- `telemetry_corners.jsonl` 的 `median` 可作为 `TRACK_CORNER_ARCS` 之外的
  **弯速参考**，用于校验 `domain/track.py` 的 `speed_kmh` 是否偏离真实 2026；
- `track_profile.jsonl` 的 `setup_params` 分布可做**先验约束**（例如新手默认
  调教取各参数中位数），比全局缺省更贴近真实玩家分布；
- `setups_f1_26.jsonl` 已是 `train_real_lap_model.py` 的输入口径子集，
  可直接替换/补充 `data/setup_laps/f1laps_setups.jsonl` 中 2026 的部分。
