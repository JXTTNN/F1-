#!/usr/bin/env python3
"""检查所有原始SVG中使用的path命令类型。"""
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw_svgs"

all_commands = set()
track_commands = {}

for svg_path in sorted(RAW_DIR.glob("*.svg")):
    content = svg_path.read_text(encoding="utf-8")
    # 提取所有path d属性
    d_matches = re.findall(r'\bd="([^"]+)"', content)
    cmds_in_file = set()
    for d in d_matches:
        # 提取所有命令字母
        cmds = re.findall(r'[a-zA-Z]', d)
        cmds_in_file.update(cmds)
    track_commands[svg_path.name] = sorted(cmds_in_file)
    all_commands.update(cmds_in_file)

print("所有原始SVG中使用的命令类型:")
print(f"  命令集合: {sorted(all_commands)}")
print()
print("各赛道使用的命令:")
for name, cmds in sorted(track_commands.items()):
    print(f"  {name}: {cmds}")

# 检查是否有H/h/V/v命令
hv_commands = all_commands & set('HhVv')
if hv_commands:
    print(f"\n⚠️ 发现H/h/V/v命令: {sorted(hv_commands)}")
    print("  旋转这些命令需要转换为L/l命令")
else:
    print("\n✅ 没有H/h/V/v命令，旋转实现可以简化")