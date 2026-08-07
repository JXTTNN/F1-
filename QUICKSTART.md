# F1OPT 快速上手

## 第一步：下载

从 [Releases](https://github.com/JXTTNN/F1-/releases) 下载最新版本。无需 Python 或任何依赖，双击即可运行。

```bash
./f1opt --help
```

## 第二步：车手反馈

```bash
# 查看可用示例
./f1opt feedback --list-examples

# 弯道级
./f1opt feedback --track suzuka --question "为什么 T1 入弯总推头？"

# 扇区级
./f1opt feedback --track suzuka --question "S2 连续弯那段指向性太差了"

# 整体
./f1opt feedback --track bahrain --question "还能再快多少？"
```

**三级精度：**

| 级别 | 范围 | 示例 |
|---|---|---|
| **弯道** | 具体弯道 | "T1 推头" / "T130R 出弯慢" |
| **扇区** | 某一段/扇区 | "S2 指向性差" / "S3 高速段不稳" |
| **整体** | 整圈 | "还能快多少？" / "轮胎温度不均" |

## 第三步：调校搜索

```bash
# 差分进化（100 轮）
./f1opt search --track suzuka --iterations 100

# 贝叶斯优化
./f1opt bayesian --track monza --iterations 15 --acquisition ei

# 预测圈速（19 个调校参数）
./f1opt predict --track suzuka --setup-json '{"front_wing":30,"rear_wing":25,"on_throttle_diff":80,"off_throttle_diff":50,"front_suspension":5,"rear_suspension":5,"front_arb":5,"rear_arb":5,"front_tyre_pressure":23,"rear_tyre_pressure":21,"front_camber":-3.0,"rear_camber":-1.5,"front_toe":0.05,"rear_toe":0.15,"front_ride_height":5,"rear_ride_height":5,"front_brake_bias":50,"brake_pressure":90,"fuel_load":30}'
```

## 第四步：接遥测

1. 启动 F1 25/26 游戏，进入任意赛道
2. 在游戏设置中开启 UDP 遥测（端口 20777）
3. 运行 `./f1opt serve` 启动 API 服务
4. 跑完几圈后停止遥测，系统自动启动 LLM 分析

## 更多

| 内容 | 位置 |
|---|---|
| 完整文档 | [README.md](README.md) |
| 架构设计 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 技术研究 | [docs/F1_2026_Research_Summary.md](docs/F1_2026_Research_Summary.md) |
| 使用示例 | [examples/](examples/) |
| API 文档 | 启动 `./f1opt serve` 后访问 http://127.0.0.1:8000/docs |
| 源代码 | [f1opt/](f1opt/) |
| 测试 | [tests/](tests/) |
| 半成品 | [半成品/](半成品/) |