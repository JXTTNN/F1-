# 车手反馈使用示例

## 三级精确度 (corner / sector / overall)

系统自动识别你的问题精确度, 回答匹配对应级别.

### [corner] 精确到某个弯道

```bash
# 推头 (understeer)
f1opt feedback --track suzuka --question "为什么 T1 入弯总推头?"

# 过度转向 (oversteer)
f1opt feedback --track monza --question "出弯时车尾总往外甩"

# 弯心 traction 不足
f1opt feedback --track suzuka --question "T3 弯心的时候后轮总滑, 不敢加油"

# 出弯速度
f1opt feedback --track suzuka --question "T130R 出弯速度上不去, 总被甩开"

# 英文提问
f1opt feedback --track silverstone --question "How should I adjust the diff for T8?"
```

### [sector] 某一段/扇区

```bash
# 连续弯段指向性
f1opt feedback --track suzuka --question "S2 连续弯那一段车头太钝, 指向性差"

# 高速段稳定性
f1opt feedback --track monza --question "S3 高速段车身不稳, 像在飘"

# 入弯段锁死
f1opt feedback --track monaco --question "刹车点晚一点就锁死前轮"
```

### [overall] 整体感受

```bash
# 圈速潜力
f1opt feedback --track bahrain --question "圈速能再快多少?"

# 轮胎温度
f1opt feedback --track spa --question "轮胎温度左边比右边高很多"

# ERS 策略
f1opt feedback --track suzuka --question "ERS 怎么部署最快?"

# 微调建议
f1opt feedback --track suzuka --question "感觉车还行, 还能优化吗?"
```

## 完整示例输出

```bash
$ f1opt feedback --track suzuka --question "为什么 T1 入弯总推头?" --driver-style aggressive

{
  "granularity": "corner",
  "granularity_info": {
    "granularity": "corner",
    "confidence": 1.0,
    "corner_ref": "T1",
    "matched_pattern": "T1"
  },
  "summary": "T1 入弯推头主要因为前轮抓地不足...",
  "dimensions": [...],
  "engineer_message": "T1 入弯推头, 遥测显示..."
}
```

## 可用的车手风格

```bash
f1opt feedback --track suzuka --question "..." --driver-style aggressive   # 激进
f1opt feedback --track suzuka --question "..." --driver-style conservative # 保守
f1opt feedback --track suzuka --question "..." --driver-style default      # 默认
```

## 多轮对话

```bash
# 第一轮
f1opt feedback --track suzuka --question "为什么 T1 推头?" --session-id lap1

# 第二轮 (同一 session, 记住上文)
f1opt feedback --track suzuka --question "改了前翼还是推, 怎么办?" --session-id lap1
```

## 查看所有示例

```bash
f1opt feedback --list-examples
```