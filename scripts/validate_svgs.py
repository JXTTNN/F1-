#!/usr/bin/env python3
"""验证所有24个SVG文件的结构完整性"""
import re
from pathlib import Path

# 仓库根目录（本脚本位于 <root>/scripts/），避免硬编码绝对路径
ROOT = Path(__file__).resolve().parents[1]
track_dir = ROOT / "setup_tuner" / "ui" / "tracks"
expected_corners = {
    'melbourne': 14, 'shanghai': 16, 'suzuka': 18, 'sakhir': 15,
    'jeddah': 27, 'miami': 19, 'montreal': 14, 'monaco': 19,
    'barcelona': 14, 'spielberg': 10, 'silverstone': 18, 'spa': 19,
    'hungaroring': 14, 'zandvoort': 14, 'monza': 11, 'madrid': 22,
    'baku': 20, 'singapore': 19, 'austin': 20, 'mexico_city': 17,
    'sao_paulo': 15, 'las_vegas': 17, 'lusail': 16, 'yas_marina': 16,
}

all_ok = True
for track_id, n_expected in expected_corners.items():
    svg_path = track_dir / f'{track_id}.svg'
    content = svg_path.read_text(encoding='utf-8')
    
    # 检查SVG结构
    has_bg = 'class="track-bg"' in content
    has_edge = 'class="track-edge"' in content
    has_line = 'class="track-line"' in content
    has_accent = 'class="track-line-accent"' in content
    has_start = '#FF1801' in content
    
    # 计算弯道圆点数
    circles = re.findall(r'<circle cx="[\d.]+" cy="[\d.]+" r="6"', content)
    n_corners = len(circles)
    
    # 检查赛道名称
    name_match = re.search(r'letter-spacing="2">([A-Z ]+)<', content)
    name = name_match.group(1) if name_match else 'MISSING'
    
    # 检查弧线flag是否正确（0或1）
    arc_flags_ok = True
    arc_matches = re.findall(r' a [\d.]+ [\d.]+ [\d.]+ ([\d.]+) ([\d.]+) ', content)
    for la, sw in arc_matches:
        if float(la) not in (0, 1) or float(sw) not in (0, 1):
            arc_flags_ok = False
            break
    
    status = 'OK' if (has_bg and has_edge and has_line and has_accent and has_start 
                      and n_corners == n_expected and arc_flags_ok) else 'FAIL'
    if status == 'FAIL':
        all_ok = False
    
    print(f'{track_id:15s} corners={n_corners:2d}/{n_expected:2d} bg={has_bg} edge={has_edge} line={has_line} accent={has_accent} start={has_start} name={name} arc_flags={arc_flags_ok} [{status}]')

print()
if all_ok:
    print('全部OK!')
else:
    print('有问题需要修复!')
