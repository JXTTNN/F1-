# F1 2026 遥测调教优化系统：建设与验证

目标：**严格按 EA 游戏**（F1 25 + 2026 Season Pack，UDP `packetFormat=2026`）
建立"遥测 → 调教"的双向可算链路，并验证方向正确。

---

## 0. 本轮结论

| 项 | 结果 |
|---|---|
| 补齐的模块 | `setup_tuner/engine/setup_response.py`（调教 → 遥测响应） |
| 方向一致性 | **13/13 通过**（单参数拉极端 → 症状方向正确） |
| 闭环（坏调教 → 建议） | **11/11 通过**（建议指回缺省方向） |
| 单调性 | 下压力↑ → 直道阻力占比↑，通过 |
| 修复的真实缺陷 | 2 个（见 §4，均实证） |
| 静态检查 | `ruff` 0 告警 |
| 全量测试 | 2,253 passed / 3 skipped（改引擎映射后仍全绿） |

---

## 1. 严格按游戏：口径确认

游戏是 **EA SPORTS™ F1® 25 + 2026 Season Pack**，UDP `packetFormat = 2026`。
网上说的 "F1 26" 就是这个赛季包，不是另一款游戏。

### 1.1 调教参数（20 项，与游戏 Garage 一一对应）

| 类别 | 参数 | 范围 | 步长 | 缺省 | 单位 |
|---|---|---|---|---|---|
| 空气动力学 | front_wing / rear_wing | 0–50 | 1 | 25 | 级 |
| 变速箱 | on_throttle_diff / off_throttle_diff | 10–100 | 1 | 50 | % |
| 悬挂几何 | front_camber | -3.5 ~ -2.5 | 0.1 | -3.5 | ° |
| | rear_camber | -2.0 ~ -1.0 | 0.1 | -1.5 | ° |
| | front_toe | 0–0.20 | 0.01 | 0 | ° |
| | rear_toe | 0.10–0.35 | 0.01 | 0.20 | ° |
| 悬挂 | front_suspension / rear_suspension | 1–41 | 1 | 6 | 级 |
| | front_anti_roll_bar / rear_anti_roll_bar | 1–21 | 1 | 6 | 级 |
| | front_ride_height | 15–35 | 1 | 25 | 级 |
| | rear_ride_height | 40–60 | 1 | 50 | 级 |
| 刹车 | brake_pressure | 80–100 | 1 | 90 | % |
| | brake_bias | 50–70 | 1 | 58 | % |
| 轮胎 | front_left/right_tyre_pressure | 22.5–29.5 | 0.1 | 23.5 | psi |
| | rear_left/right_tyre_pressure | 20.5–26.5 | 0.1 | 22.0 | psi |

> `engine_braking` 在 UDP Packet 5 里占位但游戏里不可调，故不建模。

### 1.2 UDP 报文（14 类，解析层已对齐官方结构）

Motion 1325 / Session 926 / LapData 1399 / Event 45 / Participants 1470 /
CarSetups 1233 / CarTelemetry 1448 / CarStatus 1445 / FinalClassification 1134 /
CarDamage 1133 / SessionHistory 1460 / TyreSets 231 / MotionEx 273 /
LapPositions 1231 / CarTelemetry2 269 字节。
`LobbyInfo(9)`、`TimeTrial(14)` 与调教无关，未解析。

---

## 2. 系统链路

```
游戏 UDP (2026 格式)
   │  setup_tuner/telemetry/packets.py   ← 14 类包解析，只解玩家车位
   ▼
逐帧遥测 → 整圈摘要 lap_agg
   │  telemetry/lap_aggregator.py
   ▼
症状诊断（胎温/胎压/刹车/车高/路肩/转向输入/车损）
   │  engine/telemetry_diagnosis.py
   ▼
Dx 诊断向量（9 维）→ SetupDelta（20 参数，档位/上限约束）
   │  engine/diagnostic.py  →  engine/engine.py
   ▼
圈速代理（仿真 + 真实数据校准）→ 优化器
   │  engine/surrogate.py / optimizer.py
   ▼
【本轮新增】调教 → 遥测响应：反事实推演 + 方向验证
      engine/setup_response.py
```

---

## 3. 本轮新增：调教 → 遥测响应

### 3.1 为什么必须有

诊断引擎只会"读症状"，系统里原本**没有任何模块把调教变化翻译成遥测症状变化**。
缺这一层的后果：

1. 无法验证诊断方向对不对（不知道"前翼调低"该出现什么症状）；
2. 无法做反事实推演（改这个调教，症状会怎么变）；
3. 优化器只能依赖标量圈速代理，而该代理在真实数据上中位 ρ 仅 0.16。

### 3.2 设计原则：以缺省调教为原点

游戏缺省调教是一套均衡设置，跑标准工况应当"几乎没有症状"。因此所有症状都由
**相对缺省值的偏移量**驱动，而不是绝对物理量：

- 缺省 + 标准工况 → 症状数 ≈ 0（信噪比的前提）；
- 单项拉到极端 → 必然跨过诊断阈值，方向与游戏物理一致。

### 3.3 物理映射

| 调教 | 遥测效应 |
|---|---|
| 前翼 ↑ | 前轴下压力 ↑ → 推头减轻；直线阻力 ↑ → 直道变慢 |
| 后翼 ↑ | 后轴下压力 ↑ → 车更稳但推头 |
| 前防倾杆 ↑（硬） | 前轴侧倾 ↓ → 前轮载荷转移 ↑ → 推头 |
| 后防倾杆 ↑（硬） | 后轴侧倾 ↓ → 转向过度 |
| 前束 toe-out ↑ | 前轴响应更灵 → 转向过度倾向；直道需反复修正 |
| 车高 ↓ | 空力平台更好 → 但底板触地 |
| 刹车压力 ↑ | 锁死 + 刹车过热 |
| 刹车平衡偏前 | 前轮负担 ↑ → 前轮锁死、前刹车更热 |
| 开油门差速 ↑ | 出弯打滑 |
| 胎压偏离窗口 | 该轴温度漂移、推头/打滑 |

输出字段与 `diagnose_from_telemetry` 的判定阈值逐项对齐（胎温 95/105 °C、
前后温差 15 °C、胎压 22–26 psi、刹车温差 120 °C、触地占比 0.05 等）。

---

## 4. 验证与修复

### 4.1 方向一致性（13/13）

对 20 项参数逐一从缺省推向极端值，检查症状方向。用例覆盖翼片两端、防倾杆、
差速器、车高、刹车压力/平衡、胎压窗口。

```
[PASS] rear_wing=50.0   症状含 midcorner_understeer   # 后翼拉满 → 推头
[PASS] front_wing=50.0  症状含 exit_oversteer         # 前翼拉满 → 转向过度
[PASS] front_anti_roll_bar=21.0 症状含 midcorner_understeer
[PASS] brake_pressure=100.0     症状含 brake_fade + lockup
[PASS] front_ride_height=15.0   症状含 bottoming
```

**前提校验**：缺省调教在全部 **24 站** 上均无症状（0/24 出症状）。
这条不成立的话"有症状 = 调坏了"就无从谈起。

### 4.2 闭环（11/11）

坏调教 → 遥测 → 症状 → Dx → SetupDelta，检查建议是否指回缺省方向：

```
[PASS] brake_pressure=100.0  自身回正 brake_pressure -3.00
[PASS] front_ride_height=15.0 自身回正 front_ride_height +2.00
[PASS] rear_wing=50.0        联动回正 front_wing +5.00
```

"联动回正"表示该参数本身没动，但物理同源参数代偿（后翼过高 → 加大前翼）。

### 4.3 修复的两个真实缺陷

**缺陷 1：`brake_fade` 与 `lockup` 在 `brake_pressure` 上正好抵消**

原映射 `brake_fade → brake_power_req 0.70 / brake_stab_req 0.30`，
而 `brake_power_req` 对 `brake_pressure` 是 **+2.0**、`brake_stab_req` 是 **-0.8**。
压力 100% 时两症状同时触发，净 raw delta 恒为 `-2.2e-16` —— 刹车压力永远不动。

根因是语义错误：热衰减的根因是压力过高/散热不足，不是"制动力不够"。
已改为 `brake_stab_req 0.80 / brake_power_req 0.20`，raw 变为 **-2.8**，
建议 `brake_pressure -3.00`，方向正确。

**缺陷 2：`front_toe` 没有可观测的遥测效应**

toe-out 只影响 5 °C 胎温，跨不过任何阈值，诊断永远查不出问题。
已补：toe 纳入前后轴失衡度（+0.40 权重）并抬升平均转向角，
现在 `front_toe=0.20` 会报出转向过度类症状。

---

## 5. 数据现状（重要）

| 数据 | 规模 | 能不能用 |
|---|---|---|
| 真实游戏调教 + 圈速 | **2,666 条** / 24 站 / 1,045 作者（F1Laps） | 可标定排序模型 |
| 真实游戏逐帧遥测 | **0 条** | 必须自采（见 §6） |
| 真实 F1 2026 遥测 | 19,459 圈 / 314,891 弯样本 | **不能当游戏遥测用** |

真实圈速校准的现状：留一赛道配对准确率 **59.6%**（95% CI 58.0–61.2%，显著优于
随机 50%），站内中位 Spearman ρ = 0.156，预测量级比真实低约 3.1×。
结论：**能排序，不能定量**。

> 注意：之前入库的 TracingInsights 数据是**真实 F1 2026 赛事遥测**，
> 与游戏物理模型不同，不能直接喂给游戏调教诊断。它的价值是赛道画像与
> 逐弯基准参考，不是游戏遥测。

---

## 6. 采集真实游戏遥测

游戏内：设置 → 遥测 → UDP 遥测 **开**，IP `127.0.0.1`，端口 `20777`，
发送速率 `30–60 Hz`，格式选 **2026**。

```bash
pip install -e .
f1opt --host 127.0.0.1 --port 8000
```

打开浏览器后跑 5–10 圈，系统会按圈聚合并给出症状与调教建议。

---

## 7. 下一步

1. 用真实游戏遥测替换合成响应（`setup_response` 的系数需要实测校准）；
2. `brake_fade` 的强度分级目前只有 2/3 两档，可按温度细分；
3. 轮胎温度/压力规则可加"内外侧温差"维度（游戏 UDP 提供 `m_tyresInnerTemperature`）；
4. 把 `setup_response` 接进优化器做反事实搜索（当前只用于验证与解释）。
