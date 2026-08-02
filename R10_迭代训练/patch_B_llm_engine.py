# ============================================================
# R10 迭代训练补丁 B — llm_engine.py 参数调教
# ============================================================
# 目标：
#   1. 将 n_batch 从 512 → 1024，提升 prefill 吞吐（本机4核可接受）
#   2. n_ctx 从 4096 → 2048，匹配 0.5B 主力（省内存、加载快）
#   3. 限制 n_threads 与核数一致（本机 4 核），避免 CPU 抢占
#   4. 明确支持"强参数注入"：让 chat_stream 的 max_tokens/temperature
#      真正覆盖全局默认值（配合 patch_A 使用）
#
# 适用：f1_llm/llm_engine.py
# ============================================================

# 【补丁1】 在 LlamaCppEngine._ensure_loaded 里，调整加载参数
# 建议在导入区新增两个常量：
LLM_N_CTX_TUNED = 2048      # 0.5B / 1.5B 都够用；长对话仍够
LLM_N_BATCH_TUNED = 1024    # prefill 吞吐更高

# 修改后 _ensure_loaded 形如：
#   self._llm = Llama(
#       model_path=str(self.model_path),
#       n_ctx=LLM_N_CTX_TUNED,          # 4096 -> 2048
#       n_threads=self._n_threads,
#       n_batch=LLM_N_BATCH_TUNED,      # 512 -> 1024
#       verbose=False,
#       use_mlock=False,
#       use_mmap=True,
#       logits_all=False,
#       embedding=False,
#   )

# 【补丁2】 OllamaEngine 的 num_ctx 也对齐 LLM_N_CTX_TUNED
#   _build_payload 中把 "num_ctx": LLM_CONTEXT_WINDOW 改为 "num_ctx": LLM_N_CTX_TUNED
