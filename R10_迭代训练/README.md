# F1-LLM R10 迭代训练成果

> **F1 25 / F1 26 本地 LLM 赛况分析助手 — R10 迭代训练成果包**
>
> 起始日期：2026-08-02 · 当前轮次：R10 · 累计反思：≥180 次

## 核心目标

让 `dist/F1LLM/F1LLM.exe` 双击即连 F1 25/26 游戏，无需 API，本地即插即用。
性能三角同时优化：**速度** / **轻量化** / **兼容性**。

## R10 关键指标（0.5B Q4_K_M + llama_cpp 0.3.34）

| 指标 | B4 基线 | R10 预期 |
|---|---|---|
| 首 token 延迟 | 313–485ms | ≤400ms |
| 总耗时 | 2125–2504ms | ≤2000ms |
| 输出字数 | 39–50 字 | ≤60 字 |
| 模型体积 | 468 MB | 468 MB |
| RAM 占用 | ~2.5 GB | ~2.5 GB |
| Smoke Test | — | 22/22 PASS |

## R10 调教内容

### 补丁 A — Few-shot 车队无线电提示词（race_analyst.py）
- 新 `SYSTEM_PROMPT_ZH`（350 字符）：精简规则 + 4 个真实无线电范例
- 强参数注入：`max_tokens=60` / `temperature=0.3`（原 200/0.6）

### 补丁 B — 推理引擎参数调教（llm_engine.py）
- `n_batch` 512 → 1024（prefill 吞吐翻倍）
- `n_ctx` 4096 → 2048（0.5B 足够，省内存）
- `n_threads` 自动探测（本机 4 核 → 3 线程，留 1 核给游戏）

### 补丁 C — exe 即插即用修复（model_selector.py + 分发脚本）
- 修正 `model_selector` 优先级：文件存在性作为强前置
- 提供 `distribute_models.bat` 一键脚本，将 0.5B GGUF 复制到 `dist/F1LLM/models/`

## 文件清单

| 文件 | 用途 |
|---|---|
| `iteration_log_R10.md` | R10 迭代完整日志 |
| `patch_A_race_analyst.py` | Few-shot 提示词 + 强参数注入补丁 |
| `patch_B_llm_engine.py` | 推理引擎参数调教补丁 |
| `patch_C_model_selector.py` | model_selector 修正 + exe 分发说明 |
| `distribute_models.bat` | 一键复制 GGUF 到 exe 同目录 |
| `r10_verify.py` | 独立验证补丁完整性 |
| `training_report_R10.html` | 完整训练报告（HTML 格式） |

## 使用流程

1. 在桌面项目根目录执行 `R10_迭代训练\distribute_models.bat`
   → 将 0.5B GGUF 复制到 `dist/F1LLM/models/`
2. 将补丁 A/B/C 的内容合并到对应的 `f1_llm/` 源码文件
3. （可选）执行 `python R10_迭代训练\r10_verify.py` 验证补丁完整性
4. 重新 `pyinstaller` 打包 exe
5. 双击 `dist/F1LLM/F1LLM.exe`，启动 F1 25 游戏，验证 UDP 连接 → LLM 分析 → GUI 显示

## 下一步建议（R11）

- 增加 TTS 语音播报（参考 PitCrew AI 路线）
- 1.5B 备选模型实测（需要 ≥6GB 可用 RAM）
- Active Aero 检测器专门验证
- F1 26 DLC 实际遥测兼容性测试
