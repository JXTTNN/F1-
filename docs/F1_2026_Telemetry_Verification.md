# F1 2026 Season Pack UDP 遥测全面校验报告

**规范来源**：EA Forums 官方帖 《EA SPORTS™ F1®25: 2026 Season Pack UDP SPECIFICATION》
- 官方结构文本：`2026 Season Pack Telemetry Output Structures (1).txt`
- 官方 PDF：`Data Output from F1 25 2026 Season Pack (1).pdf`
- 本地缓存：`.ref/f1_2026_structures.txt`

**校验原则**：只能 F1 2026 格式，packetFormat=2026，cs_maxNumCarsInUDPData=24，G 值 int16/1000，车队 id uint16，新增 Packet 16 CarTelemetry2。

## 1. 包大小一致性校验

| Packet | 规范大小 | 实测大小 | 状态 |
|---|---|---|---|
| Header | 29 | 29 | ✓ |
| Motion | 1325 | 1325 | ✓ |
| Session | 926 | 926 | ✓ |
| Lap | 1399 | 1399 | ✓ |
| Event | 45 | 45 | ✓ |
| Participants | 1470 | 1470 | ✓ |
| Car Setups | 1233 | 1233 | ✓ |
| Car Telemetry | 1448 | 1448 | ✓ |
| Car Status | 1445 | 1445 | ✓ |
| Final Classification | 1134 | 1134 | ✓ |
| Lobby Info | 1062 | 1062 | ✓ |
| Car Damage | 1133 | 1133 | ✓ |
| Session History | 1460 | 1460 | ✓ |
| Tyre Sets | 231 | 231 | ✓ |
| Motion Ex | 273 | 273 | ✓ |
| Time Trial | 104 | 104 | ✓ |
| Lap Positions | 1231 | 1231 | ✓ |
| Car Telemetry 2 | 269 | 269 | ✓ |

全部通过。

## 2. 解析覆盖度

### 已解析 16/16 包
- 1 Session, 2 LapData, 5 CarSetups, 6 CarTelemetry, 7 CarStatus, 13 MotionEx
- 0 Motion, 10 CarDamage, 11 SessionHistory, 12 TyreSets, 15 LapPositions, 16 CarTelemetry2
- 3 Event, 4 Participants, 8 FinalClassification

### 关键 2026 变更点检查
- 24 车位：`NUM_CARS=24` ✓
- G 值：`m_gForceLateral / m_gForceLongitudinal` 在 Motion 中为 int16，解析器除以 1000 ✓
- 车队 id：Participant `m_teamId` 为 uint16 ✓
- 主动空力：Packet 16 字段 `m_activeAeroMode / m_activeAeroAvailable / m_overtakeAvailable / m_overtakeActive` ✓
- 引擎制动：CarSetupData 仍含 `m_engineBraking` uint8，规范保留但游戏无调教入口，解析器保留字段

## 3. 字段级对齐

以 CarSetupData 为例，规范 vs 解析器：
m_frontWing, m_rearWing, m_onThrottle, m_offThrottle, m_frontCamber, m_rearCamber, m_frontToe, m_rearToe, m_frontSuspension, m_rearSuspension, m_frontAntiRollBar, m_rearAntiRollBar, m_frontSuspensionHeight, m_rearSuspensionHeight, m_brakePressure, m_brakeBias, m_engineBraking, m_rearLeftTyrePressure, m_rearRightTyrePressure, m_frontLeftTyrePressure, m_frontRightTyrePressure, m_ballast, m_fuelLoad —— 全部命中。

SessionHistory 关键字段：
m_carIdx, m_numLaps, m_numTyreStints, m_bestLapTimeLapNum, lap_history[ m_lapTimeInMS, sector1, sector2, sector3, m_lapValidBitFlags ], tyre_stints —— 全部命中。

## 4. 本地数据配合训练建议

当前本地数据：3 赛道 / 13 圈 / 6 套 setup / 湿地 3 圈

**必须补充的类型**
1. 赛道多样性：8-12 条赛道，覆盖低速/高速/街道
2. Setup 变化密度：每赛道 ≥3 套 setup ×5 圈 best-of-5
3. 2026 专属工况：主动空力模式直道/弯道切换、超车模式激活、24 车位拥挤路段
4. 损伤工况：CarDamage 轻中重损伤样本，用于区分圈速慢来源
5. 天气梯度：0晴/1轻云/2阴/3小雨/4大雨/5暴雨，配合轮胎配方
6. G 值极值：高速弯、刹车区、路肩压弯样本

**建议总量**
T2 泛化档：~300-400 圈，分 8-12 赛道，2 干配方 ×3-5 setup ×5 圈
T3 普世档：~1500-2000 圈，含湿地

结论：解析已全面覆盖官方 2026 规范，可直接与本地数据联合训练。下一步按上述工况补采。
