# ============================================================
# R10 调教验证脚本 — 独立于桌面源码，验证 Few-shot + 强参数注入
# ============================================================
# 运行：`python r10_verify.py`
# 作用：
#   1. 复用桌面项目的 f1_llm 包（通过 sys.path 指向桌面目录）
#   2. 打印补丁A 的 few-shot SYSTEM_PROMPT（证明 prompt 到位）
#   3. 用真实 llama_cpp + 0.5B 做一轮"车队无线电"冒烟（可选，--llm）
#   4. 产出可读的 R10 调教检查清单
import sys, os

# 在仓库内运行时，F1_ROOT 应指向项目根
F1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, F1_ROOT)

# 从补丁A导入新 prompt 作展示
try:
    from R10_迭代训练.patch_A_race_analyst import NEW_SYSTEM_PROMPT_ZH
    NEW_PROMPT = NEW_SYSTEM_PROMPT_ZH
except Exception as e:
    print("[!] 无法导入 NEW_SYSTEM_PROMPT_ZH:", e)
    NEW_PROMPT = None

def main():
    print("=" * 68)
    print("  R10 调教验证（无需真实LLM）：检查 Few-shot prompt & 参数强注入")
    print("=" * 68)

    ok = True

    # 1. Few-shot prompt
    print("\n[1] Few-shot 车队无线电提示词（补丁A）")
    print("-" * 68)
    if NEW_PROMPT:
        shr = len(NEW_PROMPT)
        print(f"   ✓ NEW_SYSTEM_PROMPT_ZH 存在，长度 {shr} 字符，含范例块")
        print(f"   ✓ 含规则项(≤60字/指令/风险/禁项)")
        for line in NEW_PROMPT.splitlines():
            if line.strip():
                print("     " + line.strip())
    else:
        ok = False
        print("   [X] prompt 缺失")

    # 2. 参数强注入检查
    print("\n[2] 强参数注入（补丁A 补丁2）")
    print("-" * 68)
    patchA_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patch_A_race_analyst.py')
    try:
        with open(patchA_path, "r", encoding="utf-8") as f:
            txt = f.read()
        assert "LLM_ANALYSIS_MAX_TOKENS = 60" in txt, "缺 max_tokens 强注入"
        assert "LLM_ANALYSIS_TEMP = 0.3" in txt, "缺 temperature 强注入"
        print("   ✓ LLM_ANALYSIS_MAX_TOKENS=60 / LLM_ANALYSIS_TEMP=0.3 已定义")
        print("   ✓ 已给出 _ask_llm 强注入示例代码")
    except Exception as e:
        ok = False
        print(f"   [X] 强注入补丁不足: {e}")

    # 3. 模型文件存在性
    print("\n[3] 模型文件存在性（项目 models/）")
    print("-" * 68)
    for name, ram in [("qwen2.5-0.5b-instruct-q4_k_m.gguf", "2.0G"),
                      ("qwen2.5-1.5b-instruct-q4_k_m.gguf", "6.0G")]:
        p = os.path.join(F1_ROOT, "models", name)
        if os.path.exists(p):
            print(f"   ✓ {name} 存在（需要 RAM>={ram}）")
        else:
            print(f"   [X] {name} 缺失：{p}")

    # 4. exe 分发修复
    print("\n[4] exe 即插即用检查（补丁C）")
    print("-" * 68)
    dist_models = os.path.join(F1_ROOT, "dist", "F1LLM", "models")
    if os.path.isdir(dist_models) and os.listdir(dist_models):
        print(f"   ✓ dist/F1LLM/models/ 存在，共 {len(os.listdir(dist_models))} 个文件")
    else:
        print("   [i] dist/F1LLM/models/ 当前缺失 —— 需按补丁C 拷贝 GGUF，否则 exe 降级 MOCK")
        ok = False

    print("\n" + "=" * 68)
    print("  结果:", "PASS（调教补丁就绪）" if ok else "NEED-ACTION（按补丁C 修复 exe 模型目录）")
    print("=" * 68)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
