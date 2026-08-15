# 变更日志 (Changelog)

> 本文件记录 f1opt 系统的优化迭代历史。每次迭代都有可验证的提升。

## 2026-08 优化迭代

### Windows 兼容性
- **修复 CLI 入口点缺失**：添加 `[project.scripts]` 与 `cli.py` 的 `__main__` 块，
  修复 `f1opt` 命令不存在、以及 **EXE 打包后启动即退出** 的严重 bug。
- **移除已弃用的 ProactorEventLoop 显式设置**（Python 3.14+ 弃用）。
- **修复 `app.py` 的 `STARTUPINFO.dwFlags` 类属性错误**：该错误曾导致整个 API 层无法导入。
- **MemoryTracker 跨平台内存测量**：Windows 使用 `GetProcessMemoryInfo` (psapi)。
- **修复 `f1opt.spec` 的 `__spec_file__`**：改用 PyInstaller 正确的 `SPECPATH` 全局变量。

### F1 2026 阵容严格对齐 (10 队/20 车手 → 11 队/22 车手)
- 新增 **Cadillac** 车队 + Perez/Bottas 车手。
- Kick Sauber 正名 **Audi**。
- 修复 Racing Bulls 车手 ID 错位 (law/had)。
- 排位赛 Q1/Q2 淘汰数 5→6；赛季模拟器 22 车手 / 11 车队。

### 内置 LLM 车手反馈
- 同步反馈引擎维度计数 12→18（引擎已扩展但 docstring/测试停留在 12）。
- 修复 CLI `feedback` 命令调用不存在的 `engine.generate_feedback()` 方法。

### 遥测收集
- **架构重构**：recv 回调只解析 header（~1µs），body 解析移入异步 dispatch 循环，
  修复"recv 回调阻塞事件循环"的设计违规。
- 记录 Windows UDP 洪泛性能限制（25k pps 人工洪泛 ~60%，真实 60Hz 无影响）。

### 遥测分析
- 修复 `compute_all` 覆盖 bug：17 个分析方法（sector_timing/gear_usage/
  downforce_balance/brake_temp_balance/tyre_temp_gradient/fuel_per_sector 等）
  从未被全量入口调用，补齐为全部 28 个。

### 代码质量
- 修复 F821 未定义名（conversation/strategy 缺 Any、train 缺 Path、cli 缺 DriverProfile）。
- ruff 自动修复 32 处（未用 import/f-string/类型注解现代化）。
- 清理 F841 死代码（brake_model track_load、analytics fields/dt、corners med_frac）。

### EXE 构建
- `build.bat` 改为只装运行时依赖（不再为 EXE 安装 pytest/ruff/mypy）。

---

## 已知限制 (Known Limitations)

- **UDP 洪泛吞吐**：body 解析（Motion ≈ 75µs，22 车字典构造）仍在事件循环上，
  25k pps 人工洪泛约 60% 投递（阈值 75%）。真实 F1 60Hz 无影响。根治需线程卸载或
  numpy 向量化（见 `f1opt/telemetry/listener.py` 注释）。
