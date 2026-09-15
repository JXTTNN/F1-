"""检查 JS 文件的基本语法完整性。"""
import sys
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

# 检查是否有未闭合的字符串
for i, line in enumerate(lines, 1):
    # 简单检查：行内引号数量为奇数可能有未闭合字符串
    single_q = line.count("'") - line.count("\\'")
    double_q = line.count('"') - line.count('\\"')
    # 注释行跳过
    stripped_line = line.strip()
    if stripped_line.startswith("//") or stripped_line.startswith("/*"):
        continue
    # 模板字符串中的引号不算
    if "`" in line:
        continue