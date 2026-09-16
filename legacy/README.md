# legacy/ —— 只读参考区，非可运行代码

本目录是 F1OPT 重构前的旧实现存档。**不属于 `setup_tuner` 运行时的一部分**，
不被任何生产代码 import，也不在 CI 的 lint / pytest 范围内。

## 保留目的（唯一）

作为**对照参考资料**，用于核对新实现是否与历史行为/官方规范一致。实际使用记录：

| 用途 | 举例 |
|------|------|
| 协议字段偏移核对 | `legacy/f1opt/telemetry/packets.py` 的 Packet 13 (MotionEx) 字段序，用于交叉验证 `setup_tuner/telemetry/packets.py::parse_motion_ex` 的 244 字节结构 |
| 赛道元数据核对 | `legacy/f1opt/data/tracks.py` / `corners.py` 的赛道长度、弯道数、手工弯名（见 `setup_tuner/domain/track.py` 头部注释的引用） |
| 赛道地图弯位参考 | `legacy/f1opt/data/track_maps/__init__.py` 中各赛道 `corners` 的真实位置 |

## 明确不要做的事

- **不要 `import legacy.*`** —— 旧实现含已知缺陷（例如把车轮数组按 `FL, FR, RL, RR`
  理解，而官方规范是 `RL, RR, FL, FR`），复用会重新引入 bug。
  新实现的正确做法始终是对照 EA F1 25 UDP 官方规范重写。
- **不要把本目录当作当前架构的文档** —— 目录结构、模块划分、API 形状均已过时。
- **不要整体删除** —— 上面三类核对场景仍有价值；如需删除请先确认
  `setup_tuner/` 中所有 `legacy/...` 引用注释都已重新指向官方规范出处。

## 目录内容

- `f1opt/` —— 旧主包（telemetry / model / feedback / api / ui）
- `tests/` —— 旧测试套件（含大体积样本数据 `data/real_f1_26_sample.jsonl`）
- `半成品/` —— 未完成的历史尝试，**无任何保留价值，仅供参考**
