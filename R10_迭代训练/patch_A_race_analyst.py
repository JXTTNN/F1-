# ============================================================
# R10 迭代训练补丁 A — race_analyst.py 调教
# ============================================================
# 目标（对应 R10 迭代反思）：
#   1. 引入 to_shot 车队无线电语气示例，让 0.5B 输出的建议更"专业可变通"
#   2. 修复 _ask_llm 未强注入 max_tokens/temperature 的缺陷
#   3. 收紧输出长度（60 字上限），进一步压缩总耗时
#
# 适用：f1_llm/race_analyst.py
# 用法：
#   本文件是该模块的"替换/增补内容"，与现有文件逐一对照合并即可。
#   可直接整段替换 SYSTEM_PROMPT_ZH 常量，并在 _ask_llm 中做参数强注入。
# ============================================================

# ------------------------------------------------------------------
# 【补丁1】 用这张精简版 SYSTEM_PROMPT_ZH 替换原 prompt（更聚焦车队无线电腔调）
# ------------------------------------------------------------------
NEW_SYSTEM_PROMPT_ZH = """你是F1资深Race Engineer，代号"Pit Wall AI"，基于实时遥测辅助车手。
规则：
- 中文无线电口语，≤60字/条，先结论后理由，给具体数值
- 关键时刻下明确指令（进站/防守/推进/节能/超车/降雨换胎）
- 主动提示风险：轮胎过热、燃油不足、ERS低、前车逼近、天气变雨
- 禁emoji、禁解释、禁JSON、禁任务清单

【范例】
我："轮胎滑了"  回复→"前右115度过热。下圈降1挡控滑，18圈进站换硬。"
我："前车压我"  回复→"差距0.3s。直道尾段开DRS，第5弯出弯右侧贴边超。"
我："燃油紧张"  回复→"余1.2圈。22圈切省油混合，目标圈+0.5。"
我："现在怎么走"回复→"目前P4轮胎C3。前车0.3车速330，DRS关。建议巡航等DRS。""""

# ------------------------------------------------------------------
# 【补丁2】 让 RaceAnalyst._ask_llm 支持强参数注入（原实现漏传 max_tokens/temperature）
# ------------------------------------------------------------------
# 建议改成（传递强约束参数）：
LLM_ANALYSIS_MAX_TOKENS = 60    # 60 字以内的无线电建议（比 200 短）
LLM_ANALYSIS_TEMP = 0.3         # 分析更确定，减少废话

#   def _ask_llm(self, prompt, stream=False):
#       try:
#           answer = self.llm.chat_stream(
#               system=self.system_prompt,
#               user=prompt,
#               history=self.history[-MAX_HISTORY_TURNS:] if self.history else None,
#               max_tokens=LLM_ANALYSIS_MAX_TOKENS,   # <-- 强注入
#               temperature=LLM_ANALYSIS_TEMP,        # <-- 强注入
#           )
#           return answer if answer else self._fallback_advice(prompt)
#       except Exception as e:
#           self.logger.warning(f"LLM 调用异常: {e}")
#           return self._fallback_advice(prompt)
