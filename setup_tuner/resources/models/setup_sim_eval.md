# 调教性能模型评测

- 模型：`setup_sim_nn.json`
- 评测时间：2026-11-02T08:24:54
- 赛道：19 条（跳过 0 条未覆盖）
- 标签来源：simulate_lap_delta_ms（遥测锚定准静态物理仿真）
- **真实跑圈样本：0**

## 决策相关指标

| 指标 | 值 | 门槛 | 结论 |
|---|---|---|---|
| `fresh_local_mae_ms` | 44.46 | <=120.0 | PASS |
| `pairwise_acc` | 0.7849 | >=0.75 | PASS |
| `direction_agreement` | 1.0 | >=0.8 | PASS |
| `zero_bias_ms` | 36.53 | <=40.0 | PASS |

## 全部指标

| 指标 | 值 |
|---|---|
| `fresh_local_mae_ms` | 44.46 |
| `fresh_wide_mae_ms` | 167.29 |
| `pairwise_acc` | 0.7849 |
| `pairwise_acc_30_100ms` | 0.8341 |
| `n_pairs` | 1181 |
| `n_pairs_30_100ms` | 916 |
| `direction_agreement` | 1.0 |
| `n_direction_checks` | 425 |
| `zero_bias_ms` | 36.53 |
| `train_val_mae_ms` | 92.689 |
| `train_val_mae_ms_local` | 54.421 |
| `model_tracks` | 19 |
| `model_samples` | 7600 |

## 说明

- `pairwise_acc` 是优化器每次「接受/拒绝候选」的实际上限：低于 0.5 等于掷硬币。
- `direction_agreement` 决定坐标上升能否朝正确方向走；只统计真实增益 ≥3 ms 的参数。
- 上表全部由**训练时未见过**的样本（不同随机种子）测得，不是训练集指标。
- 以下指标衡量对物理仿真器的保真度，不构成真实跑圈验证。
