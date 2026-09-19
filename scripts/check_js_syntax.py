"""检查 JS 文件的基本语法完整性。"""
from pathlib import Path

content = Path("setup_tuner/ui/app.js").read_text(encoding="utf-8")

# 括号匹配检查
opens = content.count("{")
closes = content.count("}")
print(f"Braces: {opens} open, {closes} close, diff={opens-closes}")

opens_p = content.count("(")
closes_p = content.count(")")
print(f"Parens: {opens_p} open, {closes_p} close, diff={opens_p-closes_p}")

opens_b = content.count("[")
closes_b = content.count("]")
print(f"Brackets: {opens_b} open, {closes_b} close, diff={opens_b-closes_b}")

# IIFE 结构检查
stripped = content.strip()
print(f"Starts with IIFE: {stripped.startswith('(function')}")
print(f"Ends with IIFE: {stripped.endswith(')();')}")

# 检查是否有明显的语法问题
lines = content.split("\n")
print(f"File: {len(content)} chars, {len(lines)} lines")

# 未闭合字符串检查（已移除，原因如下）
# 此处曾有一版"未闭合字符串"检查，靠统计行内引号奇偶判断。
# 该启发式对正则字面量（/[&<>"']/g）与对象字面量（{"&": "&amp;"}）必然误报，
# 在已知合法的 app.js 上就稳定报出 2 行假阳性。可靠版本需要真正的 JS 词法
# 分析（字符串/注释/正则状态机），不是本脚本的职责，故移除以免留下噪音，
# 括号配平与 IIFE 结构检查（上方）继续有效。
