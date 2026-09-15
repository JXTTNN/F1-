"""检查所有赛道SVG中的弯道标号与track.py定义是否一致"""
import re
from pathlib import Path
from setup_tuner.domain.track import ALL_TRACKS

def extract_svg_corner_numbers(svg_path: Path) -> list[int]:
    """从SVG文件中提取弯道编号"""
    if not svg_path.exists():
        return []
    text = svg_path.read_text(encoding="utf-8")
    # 匹配 <text ...>数字</text> 模式（弯道标号）
    numbers = re.findall(r'<text[^>]*text-anchor="middle"[^>]*>(\d+)</text>', text)
    # 过滤掉赛道名称文本（如 MELBOURNE）
    result = []
    for n in numbers:
        try:
            v = int(n)
            if 1 <= v <= 30:  # 弯道编号范围
                result.append(v)
        except ValueError:
            pass
    return result

def check_all_tracks():
    """检查所有赛道的弯道标号一致性"""
    tracks_dir = Path("setup_tuner/ui/tracks")
    issues = []

    for track in ALL_TRACKS:
        svg_path = tracks_dir / f"{track.track_id}.svg"
        svg_corners = extract_svg_corner_numbers(svg_path)
        track_corners = [c.number for c in track.corners]

        # 检查数量
        if len(svg_corners) != len(track_corners):
            issues.append({
                "track": track.track_id,
                "issue": f"弯道数量不一致: track.py={len(track_corners)}, SVG={len(svg_corners)}",
                "track_corners": track_corners,
                "svg_corners": svg_corners,
            })
            continue

        # 检查编号是否匹配
        if svg_corners != track_corners:
            issues.append({
                "track": track.track_id,
                "issue": f"弯道编号不匹配: track.py={track_corners}, SVG={svg_corners}",
                "track_corners": track_corners,
                "svg_corners": svg_corners,
            })
            continue

        # 检查编号是否连续从1开始
        expected = list(range(1, len(track_corners) + 1))
        if track_corners != expected:
            issues.append({
                "track": track.track_id,
                "issue": f"弯道编号不连续: track.py={track_corners}, 期望={expected}",
            })

    print(f"检查了 {len(ALL_TRACKS)} 条赛道")
    print(f"发现问题: {len(issues)} 条")

    if issues:
        print("\n=== 问题详情 ===")
        for issue in issues:
            print(f"\n赛道: {issue['track']}")
            print(f"  问题: {issue['issue']}")
            if 'track_corners' in issue:
                print(f"  track.py弯道: {issue['track_corners']}")
            if 'svg_corners' in issue:
                print(f"  SVG弯道标号: {issue['svg_corners']}")
    else:
        print("\n✅ 所有赛道弯道标号一致")

    return issues

if __name__ == "__main__":
    check_all_tracks()