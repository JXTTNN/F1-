# F1OPT 调教优化助手 —— 实现任务清单（tasks.md）

> 文档版本：v1.0
> 上游：`spec.md`（58 条需求） + `design.md`（技术设计 2.x）
> 新代码包名：`setup_tuner/`（与 `legacy/f1opt` 完全隔离）
> 质量关卡：每个任务至少经过 `G0静态 → G1单元 → G2评审（查什么/怎么查见各任务）`；核心模块（T2/T6）额外 `G3官方溯源`。

---

## 一、任务总览与依赖

| 任务 | 内容 | blocked_by | 可并行组 |
|------|------|-----------|---------|
| T1 | 项目骨架 + 依赖收紧 | — | 批次① |
| T2 | 遥测解析模块（核心） | T1 | 批次① |
| T3 | 赛道/弯道数据 + SVG 资产 | T1 | 批次① |
| T4 | 调教 schema + 症状枚举 | T1 | 批次① |
| T5 | 数据层 + 反馈模块 | T3,T4 | 批次② |
| T6 | 分析模型（核心） | T4,T5 | 批次② |
| T7 | API 层 + 一键启动 | T2,T3,T4,T5,T6 | 批次③ |
| T8 | 前端 UI | T7 | 批次④ |
| T9 | 打包 + CI/CD | T1~T8 | 批次⑤ |

**关键路径**：T1 → T4 → T5 → T6 → T7 → T8 → T9
**关键路径**（备选）：T1 → T2 → T7 → T8 → T9

---

## 二、任务卡片

### T1 项目骨架 + 依赖收紧
- **目标文件**：`pyproject.toml`、`.env.example`、`setup_tuner/__init__.py`、`setup_tuner/config.py`
- **内容**：
  - `pyproject.toml` 依赖仅保留 `fastapi / uvicorn[standard] / pydantic>=2 / structlog`；**删除** `torch / scipy / pyarrow / opentelemetry-* / slowapi / httpx / websockets`(用 uvicorn 内建 ws)
  - `setup_tuner/config.py`：读 `.env`（UDP host/port、API host/port、data_dir、log level）
  - `.env.example` 仅含本地配置（无外部凭据）
- **验收**：`pip install -e .` 成功；`import setup_tuner` 不引入 numpy/torch；`config` 能解析 `.env`
- **测试要点**：G1 单测 `config` 解析（默认值 + 覆盖）；检查次数 1 轮，检查"依赖清单中无 ML 库"（怎么查：`pip freeze` 或 `pyproject` 依赖 grep 断言）

### T2 遥测解析模块（核心，需 G3 溯源）
- **目标文件**：`setup_tuner/telemetry/__init__.py`、`packets.py`、`listener.py`、`stream.py`
- **内容**：
  - `packets.py`：29B header（packetFormat/packetId/sessionUid/…playerCarIndex）+ 6 类包解析（Session 天气、LapData 圈速/扇区、CarSetups 调教、CarStatus 轮胎、CarTelemetry、赛道 Track 信息），**纯 `struct`，零 numpy**；对照 `legacy/f1opt/telemetry/packets.py` 核对字段偏移/类型/大小端
  - `listener.py`：UDP 监听 `127.0.0.1:20777`，容错（丢包/乱序/短包），只解析玩家车辆
  - `stream.py`：最新帧内存缓存，供 WS 推送
- **验收**：官方样本 fixture 能解析出 赛道/调教/天气/轮胎/圈速/扇区 六类数据；短包抛 `PacketTooShortError` 不崩溃
- **测试要点**：G1 用构造的官方样本逐字段断言（检查字节偏移/大小端/枚举）；G3 每条字段映射标注官方 UDP 规范出处；检查 3 次：①逐包字段断言 ②边界包(截断/超长/错包头) ③粘包/乱序容错

### T3 赛道/弯道数据 + SVG 资产
- **目标文件**：`setup_tuner/domain/track.py`、`setup_tuner/ui/tracks/*.svg`（24 条）、赛道数据导入脚本
- **内容**：
  - `domain/track.py`：`Track` / `Corner` / `CornerAnchor`（含 SVG 归一化坐标 anchor_x/y 0~1，用于可点击热区）
  - 从 `legacy/f1opt/ui/static/*.svg` **核对后复制** 24 条 SVG 到 `ui/tracks/`
  - 24 赛道元数据 + 弯道序列（对照 `legacy/f1opt/data/tracks.py`、`corners.py` 核对 track_id/名称/弯道数/udp_track_id）
- **验收**：24 赛道可加载；每赛道含弯道锚点；SVG 路径可定位；`udp_track_id` 与遥测 m_trackId 映射正确
- **测试要点**：G1 校验"24 赛道完整 + 每弯 anchor 在 0~1 区间 + udp_track_id 唯一"；检查 1 轮数据完整性校验

### T4 调教 schema + 症状枚举
- **目标文件**：`setup_tuner/domain/setup.py`、`setup_tuner/domain/symptoms.py`
- **内容**：
  - `domain/setup.py`：23 参数全集（前翼/后翼/主动空力Z/X、on/off差速、前/后外倾角、前/后束角、前/后弹簧、前/后防倾杆、前/后行驶高度、阻尼、刹车压力、刹车配比、前/后胎压、发动机制动、配重），每参数 range/default/step/单次上限
  - `domain/symptoms.py`：12 症状枚举（entry/apex/exit/global 四类）+ 强度 0–5（默认3）——**逐字对齐 spec FR-FBK-02**
  - 注意：阻尼、主动空力 Z/X 与 legacy 差异已在 design 2.6.2 记录，缺省取值标注"EA F1 2026 官方调教指南"
- **验收**：23 参数无遗漏、范围/步长齐；12 症状枚举与 spec 完全一致
- **测试要点**：G1 断言参数集 == 23 且含全部 7 大类；症状枚举 == 12 且 `tyre_wear` 中文为"胎耗偏高"（非胎温）；检查 2 次：①参数完整性 ②症状逐字一致性

### T5 数据层 + 反馈模块
- **目标文件**：`setup_tuner/db/schema.sql`、`db/store.py`、`feedback/service.py`、`feedback/iteration.py`
- **内容**：
  - `schema.sql`：6 张表（track/corner/setup/feedback/suggestion/iteration）字段级 DDL（对照 design 2.3.2）
  - `store.py`：标准库 `sqlite3` 连接 + CRUD
  - `service.py`：反馈录入/查询；**未点击弯道默认"正常"**；请求建议前校验"至少 1 条反馈或全局症状"
  - `iteration.py`：历史反馈/建议留存与对比
- **验收**：反馈可写入/查询；未点击弯道不生成记录；无反馈时请求建议提示正确
- **测试要点**：G1 断言反馈 CRUD + 未点击默认正常 + 空反馈拦截；检查 2 次：①CRUD 正确性 ②"没点=正常"语义

### T6 分析模型（核心，需 G3 溯源 + 双人交叉）
- **目标文件**：`setup_tuner/engine/__init__.py`、`diagnostic.py`、`coupling.py`、`rules.py`、`engine.py`、`confidence.py`
- **内容**：
  - `diagnostic.py`：诊断向量 Dx（8 维：前轴抓地/后轴抓地/入弯响应/高速稳定性/制动稳定性/制动力/出弯牵引/轮胎寿命 + 底盘离地），症状→Dx 求值
  - `coupling.py`：耦合矩阵 C（诊断维度×参数），每元素 符号/强度/出处
  - `rules.py`：规则库 `{症状, 全参数增量 delta_table, 依据:官方出处}` 加载与查询（为 Dx×C 的物化结果）
  - `engine.py`：`SetupDelta = clamp(Dx × C)`（矩阵乘法 → 遥测校准 → 单次上限 clip → 区间约束 → 档位对齐）
  - `confidence.py`：置信度（遥测证据充分度 + 反馈明确度）
- **验收**：任意单症状产出 `SetupDelta` 覆盖**全部**23 参数；相同输入输出完全一致（确定性）；每参数建议有官方出处；不越界
- **测试要点**：G1 断言 ①覆盖全部参数维度 ②方向符合官方 ③确定性（跑两次结果相等）④区间不越界；G3 每条规则 `source` 非空 + 抽样人工核对方向；**双人交叉验证**（独立 reviewer 复审矩阵方向与出处）；检查 3 次：①覆盖性 ②确定性 ③溯源性

### T7 API 层 + 一键启动
- **目标文件**：`setup_tuner/app.py`、`api/routes.py`、`api/ws.py`、`cli.py`
- **内容**：
  - `app.py`：FastAPI 工厂 + 静态资源挂载 + 生命周期（启停遥测监听）
  - `routes.py`：REST（health/tracks/corners/setup/feedback/suggest/iteration）
  - `ws.py`：WebSocket 事件（telemetry 流 / current_corner / suggestion）
  - `cli.py`：一键启动（启动服务 → 端口占用检测 → 自动开浏览器）
- **验收**：端点通，WS 遥测实时推送 + 当前弯高亮，`cli` 启动后浏览器自动打开，端口占用时明确提示
- **测试要点**：G1 用 TestClient 断言各端点 + WS 推送；检查 2 次：①REST 契约 ②WS 事件

### T8 前端 UI
- **目标文件**：`setup_tuner/ui/index.html`、`app.js`、`style.css`
- **内容**：
  - 中文界面：赛道选择、赛道图(SVG)渲染 + 弯道可点击热区、实时遥测 + 当前弯高亮、反馈面板（12 症状 + 强度）、建议报告展示
  - `app.js`：热区点击→反馈面板；未点击=正常；WS 订阅遥测/当前弯
- **验收**：点 SVG 赛道图弯道弹出反馈面板；遥测高亮当前弯；建议报告可读（setupDelta/联动/出处/置信度/tradeoff）
- **测试要点**：G1 前端逻辑单测（热区命中/反馈数据结构）；UI 走查（中文、无卡顿）；检查 2 次：①热区效果 ②报告渲染

### T9 打包 + CI/CD
- **目标文件**：`scripts/build_nuitka.py`、`scripts/smoke.py`、`.github/workflows/ci.yml`、`build-release.yml`、`smoke.yml`、`assets/一键启动.bat`
- **内容**：
  - `build_nuitka.py`：Nuitka 编译原生 exe（windows-latest runner）
  - `smoke.py`：云端冒烟（启动 exe → 探 HTTP 端口 → 断言 200）
  - `ci.yml`：ubuntu runner 跑 ruff+mypy+pytest
  - `build-release.yml`：windows-latest runner Nuitka 编译 + 打包 zip + 上传 Release
  - `smoke.yml`：云端冒烟
  - `assets/一键启动.bat`：启动 exe + 开浏览器
- **验收**：GitHub Actions 测试全绿；Nuitka 出 exe；冒烟 200；Release 有 zip 资产
- **测试要点**：G4 云端端到端（测试+打包+冒烟全绿）；检查 3 次：①ci 测试绿 ②build-release 打包成功 ③冒烟 200

---

## 三、里程碑验收

- **批次① 完成**：T1~T4 全绿 → 进入批次②
- **批次② 完成**：T5~T6 全绿（T6 通过双人交叉 + 溯源）→ 进入批次③
- **批次③④⑤ 完成**：T7~T9 全绿 + GitHub Actions 冒烟通过 → Phase D 结束，进入 Phase E（测试生成）/F（评审）/G（验证）