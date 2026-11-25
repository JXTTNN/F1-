# 用这份训练集跑模型

## 0. 先看切分，避免泄漏

`data/f1_2026/train/splits.json` 已按分站切好：

- train：15 站
- val：3 站（madrid / spa / suzuka）
- test：3 站（melbourne / monza / singapore）

同一场比赛的圈只进一个 split。**不要**再随机按圈拆分，否则同场/同弯泄漏会把指标虚高。

## 1. 遥测代理模型（逐弯 → 弯时/弯速）

仓库现有训练脚本固定读 `data/training/corner_dataset.jsonl`，先把干净样本解压过去：

```bash
gzip -dc data/f1_2026/train/corner_samples_clean.jsonl.gz > data/training/corner_dataset.jsonl
python scripts/train_telemetry_surrogate.py \
  --corner-samples 4000 --ridge-samples 20000 --epochs 40
```

冒烟结果（本机，314,891 条干净逐弯样本）：ridge 在随机留出上 `val_MAE=0.0027`、`R²=0.9747`；MLP 在该子样本上为负 R²。这个数字只用于确认输入口径跑得通，**正式报告必须用分站留出**。

## 2. 调教 → 圈速模型

```bash
python scripts/train_real_lap_model.py \
  --source data/f1_2026/train/setups_f1_26.jsonl \
  --min-per-track 10 --folds 5
```

- 样本：899 条（24 站，干 547 / 湿 352）
- 输入：20 项调教 + 赛道/工况
- 说明：调教来自 F1 26 游戏，圈速也是游戏口径；这是**独立任务**，不要拿真实 F1 遥测圈速当标签。

## 3. 两条任务不要混

| 任务 | 数据 | 标签 | 用途 |
|---|---|---|---|
| 遥测代理 | `corner_samples_clean.jsonl.gz` | 弯时/弯速/油门刹车 | 真实 2026 赛道参考、敏感度 |
| 调教模型 | `setups_f1_26.jsonl` | 游戏圈速 | 调教推荐/排序 |

公开真实 F1 遥测没有车队调教；不存在“同一圈同时有真实遥测和真实调教”的数据集。若要做“调教+遥测同圈”监督，只能自己用 F1 26 游戏 UDP 采集（仓库里有采集器）。

## 4. 训练前处理建议

- `max_lateral_g` / `max_long_g` 有个别数十 G 尖峰，按 99 分位截尾。
- 湿圈只有 506 圈（2.6%），湿滑模型建议重采样或单独建模。
- `laps_clean.jsonl` 已剔除第 1 圈、进站/出站、非绿旗、赛会删除、计时不准、FastF1 补造圈。
