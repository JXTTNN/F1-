# F1 2026 遥测 + 调教数据包

本目录是项目用于 **F1 2026** 的外部数据落地层：真实 2026 赛季遥测（全部已举办赛站，
逐点原始文件 + 逐圈 + 逐弯）与 F1 26 游戏版调教。两边共用 `track_id`
（与 `setup_tuner.domain.track` 的 24 站一致），可直接 join。

> 入库口径（repo owner 要求）：**全部入库，不做裁剪/降采样**。
> 原始逐点遥测按赛道打成 `raw_telemetry/<track_id>.tar.gz`（上游原文件逐字节一致），
> 逐弯样本、逐圈汇总、聚合、调教也全部提交；`.ref/` 只是本机抓取缓存，不入库。

## 文件

| 文件 | 行数/数量 | 大小 | 内容 |
|------|-----------|------|------|
| `raw_telemetry/*.tar.gz` | 21 站 / 4 518 文件 | 259 MB | 逐点原始遥测**抽样**（每车手前 10 圈，每圈 600+ 采样点、14 通道），解压即还原上游目录结构 |
| `corner_samples.jsonl.gz` | 71 222 | 3.3 MB | 逐弯训练样本（每行一个 赛道×弯×圈 样本） |
| `telemetry_laps.jsonl` | 4 334 | 1.7 MB | 逐圈汇总：圈速/胎/天气/弯速极值/油门刹车占比 |
| `telemetry_corners.jsonl` | 344 | 254 KB | 21 站 × 弯道的遥测聚合：`n_laps` + 每项指标 `median/p10/p90` |
| `setups_f1_26.jsonl` | 899 | 906 KB | F1 26 调教（20 项）+ 圈速 + 作者/车队/工况/来源链接 |
| `track_profile.jsonl` | 24 | 37 KB | 每站一行：遥测画像 + 调教参数分布 + 最优/典型圈速 |
| `manifest.json` | — | 3 KB | 来源、许可、构建时间、覆盖度、行数、sha256 |

`raw_telemetry/manifest.json` 记录每站包的文件数、原始字节、包字节与 sha256。

`corner_samples.jsonl.gz` 解压后与 `data/training/corner_dataset.jsonl` 完全同构，
可直接喂 `train_telemetry_surrogate.py` / `train_real_lap_model.py`。

## 覆盖度

- 遥测（父目录，**原始抽样**）：**21 / 24 站**，Race 会话，每位车手最多 10 圈，共 **4 334 圈 / 71 222 条逐弯样本**。
  - **全量 Race 干净圈训练集见 `train/`：19 459 圈 / 314 891 条逐弯样本**（从 24 612 个可计时圈清洗而来）。
  - 缺 3 站：`jeddah`（2026 官方取消）、`lusail`（Qatar，赛历未到、上游无数据）、
    `yas_marina`（Abu Dhabi，赛历未到、上游无数据）。
  - 5 场比赛各有 1 圈在上游 404（Bahrain / Azerbaijan / Miami / Mexico City / Singapore），
    已记为 missing，不影响其他圈。
  - 21 场 Race 会话全部通过 2026 年检（`session_laptimes.lSD` 时间戳 100% 落在 2026）。
- 调教：**24 / 24 站**，899 条 2026 干/湿配对（干 547 / 湿 352）。

## 字段口径

`telemetry_corners.jsonl` 一行示例（数值四舍五入）：

```json
{"track_id":"monza","corner":1,"corner_class":"slow","n_laps":220,
 "min_speed":{"median":86.4,"p10":78.1,"p90":95.2},
 "entry_speed":{"median":242.7,"p10":190.4,"p90":280.9},
 "corner_time_s":{"median":5.86,"p10":5.52,"p90":6.31},
 "throttle_full_pct":{"median":41.2,"p10":25.0,"p90":58.3},
 "brake_pct":{"median":28.9,"p10":16.7,"p90":45.1},
 "gear_min":{"median":2.0},"gear_max":{"median":4.0},
 "max_lateral_g":{"median":2.41},"max_long_g":{"median":-2.18}}
```

速度单位 km/h；`corner_time_s` 为该弯通过时间（秒）；`throttle_full_pct` 为全油门采样占比；
`brake_pct` 为刹车采样占比；`n_laps` 是该弯的有效样本数（车手×圈），不是整圈数。

`telemetry_laps.jsonl` 一行示例：

```json
{"race":"Chinese Grand Prix","track_id":"shanghai","driver":"ALO","lap":12,
 "lap_time_s":97.418,"tyre_class":"medium","tyre_softness":0.5,"wet":0.0,
 "track_temp":24.1,"air_temp":16.2,"n_corners":16,
 "corner_time_sum_s":62.9,"min_speed_kmh":84.0,"max_speed_kmh":331.2,
 "throttle_full_pct":63.4,"brake_pct":14.2,
 "max_lateral_g":3.91,"max_long_g":-2.84}
```

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
| [F1Laps · F1 26 Setups](https://www.f1laps.com/f1-26/setups/) | 调教 + 圈速配对 | 公开 HTML 页面；robots.txt `Allow: /`（`/api/` 与登录区不触碰）；1.2s/请求限速 |

两个上游都不是 F1 官方，商标归 Formula One Licensing B.V.；本目录仅作研究/训练用途。

## 刷新方式

```bash
# 1) 抓真实遥测（可续传；缓存落 .ref/）
python - <<'PY'
from scripts.fetch_official_2026 import active_races, main
for race in active_races():
    main(["--race", race, "--session", "Race", "--max-laps", "10", "--workers", "12"])
PY

# 2) 切成逐弯样本（data/training/ 被 gitignore，属本地训练中间产物）
python scripts/build_corner_dataset.py --max-laps 400

# 3) 抓调教 + 圈速（可续传）
python scripts/fetch_setup_laps.py --all-slices

# 4) 生成数据包（聚合/逐条调教）+ 补齐逐圈/逐弯明细 + 打包原始遥测
python scripts/build_f1_2026_bundle.py
python scripts/f1_2026_extras.py
python scripts/pack_raw_telemetry.py
```

CI 也可以用 `.github/workflows/f1-2026-data.yml` 一键重建（手动 dispatch）。

## 已知口径问题（构建时已处理）

1. **1 号弯零速帧**：上游 `build_corner_dataset.py` 的 1 号弯窗口会包含记圈点零速帧，
   把 `entry/min/avg speed` 的 p10 拉成 0。本包聚合与逐圈汇总都**剔除**这类样本
   （实测 474 / 71 222，0.67%），并在
   `manifest.counts.corner_rows_dropped_zero_speed` 里如实记数；上游未改，
   训练侧若直接用 `data/training/corner_dataset.jsonl` 需自行过滤。
2. **G 值尖峰**：上游 `acc_x/y/z` 偶有尖峰（个别弯 `max_lateral_g` 可到数十 G），
   聚合用中位数/分位数抗噪；逐圈汇总的 `max_*_g` 是原始峰值，建模时建议先截尾。
3. **调教单位**：与项目 20 项参数完全一致（翼片为档位、胎压 bar、车高 mm 等），
   可直接喂 `CarSetup`。

## 与引擎的衔接建议

- `telemetry_corners.jsonl` 的 `median` 可作为 `TRACK_CORNER_ARCS` 之外的
  **弯速参考**，用于校验 `domain/track.py` 的 `speed_kmh` 是否偏离真实 2026；
- `track_profile.jsonl` 的 `setup_params` 分布可做**先验约束**（例如新手默认
  调教取各参数中位数），比全局缺省更贴近真实玩家分布；
- `setups_f1_26.jsonl` 已是 `train_real_lap_model.py` 的输入口径子集，
  可直接替换/补充 `data/setup_laps/f1laps_setups.jsonl` 中 2026 的部分。

## 训练集（`train/`，全量干净圈）

父目录的逐点原始包 `raw_telemetry/*.tar.gz` 是「每位车手前 10 圈」的抽样，
适合点级核查；**训练请用 `train/`**：

- `train/laps_clean.jsonl`：19,459 圈（21 站 Race，已剔除发车/进站/黄旗/删除/计时不准圈）
- `train/corner_samples_clean.jsonl.gz`：314,891 条逐弯样本
- `train/setups_f1_26.jsonl`：899 条调教（独立任务）
- `train/splits.json`：按分站 train/val/test = 15/3/3，防同场泄漏
- 清洗口径、字段与训练命令见 `train/DATASET_CARD.md`、`train/TRAINING.md`

> 公开真实遥测不含车队调教，所以“同一圈同时有遥测+调教”的数据不存在；
> 遥测代理与调教→圈速是两个独立任务，不能强行逐圈配对。
