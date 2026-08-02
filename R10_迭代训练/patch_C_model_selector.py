# ============================================================
# R10 迭代训练补丁 C — model_selector.py + exe 即插即用修复
# ============================================================
# 目标：
#   1. 修正 model_selector 对"不存在模型(qwen3-1.7b)"的默认命中
#      —— 调整为"文件存在优先 + RAM 达标"，避免本机(8G)错误选 Qwen3
#   2. 为 exe 即插即用补上缺失的 models/ 目录与 GGUF 分发步骤
#
# 适用：f1_llm/model_selector.py
# ============================================================

# 【补丁1】 默认 0.5B 作为本机最稳妥主力
#   ——在 config.py 中保持 DEFAULT_GGUF_PATH="qwen2.5-0.5b-..."
#     这与本机 8G RAM/4 核匹配。

# 【补丁2】 model_selector 建议调整"自动选择"逻辑：
#   原：按能力降序找第一个【RAM 够 且 文件存在】的模型
#        → 会命中 qwen3-1.7b（min_ram 8G ≤ 可用RAM）即使文件不存在
#   现建议：把"文件存在"作为强前置条件，避免无谓探测
#
# 将 pick_best_model 的自动选择分支改为：
#   for prof in _MODEL_REGISTRY:
#       path = MODELS_DIR / prof.file_name
#       if path.exists() and resources.available_ram_gb >= prof.min_ram_gb:
#           logger.info(f"自适应选择最优模型: {prof.display_name} ...")
#           return prof, str(path), resources

# 【补丁3】 exe 即插即用（关键修复）：
#   STEP1  在 dist/F1LLM/ 下建立 models/ 目录
#   STEP2  复制项目根 models/ 下的主力 GGUF（0.5B，468MB）到
#          dist/F1LLM/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
#   STEP3  见 distribute_models.bat
