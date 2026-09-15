#!/usr/bin/env python3
"""将 julesr0y/f1-circuits-svg 项目的真实赛道SVG转换为F1OPT格式。

流程：
1. 下载24条赛道的最新layout SVG（white-outline样式）
2. 解析SVG path数据（M/L/C/c/Z命令，含隐式重复命令）
3. 缩放到800x600画布（保持比例，居中）
4. 通过曲率分析检测弯道位置（匹配预期弯道数）
5. 生成F1OPT格式SVG（深色背景、赛道线、起点标记、弯道编号）
6. 输出更新的 _track_anchors.py

参考项目: https://github.com/julesr0y/f1-circuits-svg (CC-BY-4.0)
"""

from __future__ import annotations

import math
import re
import sys
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

# 导入legacy track_maps数据（含真实弯道像素坐标）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "legacy"))
try:
    from f1opt.data.track_maps import TRACK_MAPS as _LEGACY_TRACK_MAPS
except ImportError:
    _LEGACY_TRACK_MAPS = {}

# =========================================================================== #
# 配置
# =========================================================================== #

BASE_URL = (
    "https://raw.githubusercontent.com/julesr0y/f1-circuits-svg/main/"
    "circuits/minimal/white-outline/"
)
OUTPUT_DIR = Path("D:/F1OPT-Test/setup_tuner/ui/tracks")
ANCHORS_FILE = Path("D:/F1OPT-Test/setup_tuner/domain/_track_anchors.py")

TRACK_LAYOUT_MAP = {
    "melbourne": "melbourne-2", "shanghai": "shanghai-1", "suzuka": "suzuka-2",
    "sakhir": "bahrain-1", "jeddah": "jeddah-1", "miami": "miami-1",
    "montreal": "montreal-6", "monaco": "monaco-6", "barcelona": "catalunya-6",
    "spielberg": "spielberg-3", "silverstone": "silverstone-8",
    "spa": "spa-francorchamps-4", "hungaroring": "hungaroring-3",
    "zandvoort": "zandvoort-5", "monza": "monza-7", "madrid": "madring-1",
    "baku": "baku-1", "singapore": "marina-bay-4", "austin": "austin-1",
    "mexico_city": "mexico-city-3", "sao_paulo": "interlagos-2",
    "las_vegas": "las-vegas-1", "lusail": "lusail-1", "yas_marina": "yas-marina-2",
}

TRACK_CORNERS_COUNT = {
    "melbourne": 14, "shanghai": 16, "suzuka": 18, "sakhir": 15,
    "jeddah": 27, "miami": 19, "montreal": 14, "monaco": 19,
    "barcelona": 14, "spielberg": 10, "silverstone": 18, "spa": 19,
    "hungaroring": 14, "zandvoort": 14, "monza": 11, "madrid": 22,
    "baku": 20, "singapore": 19, "austin": 20, "mexico_city": 17,
    "sao_paulo": 15, "las_vegas": 17, "lusail": 16, "yas_marina": 16,
}

# F1官方赛道方向旋转角度（来自julesr0y/f1-circuits-svg项目的f1-orientation）
# 用于将SVG赛道图旋转到与F1官方方向一致
TRACK_F1_ORIENTATION = {
    "melbourne": -46, "shanghai": 24, "sakhir": 0, "jeddah": 0,
    "miami": -18, "montreal": -55, "monaco": 45, "barcelona": 32,
    "spielberg": -31, "silverstone": 90, "spa": -100,
    "hungaroring": 52, "zandvoort": 0, "monza": -95, "madrid": 0,
    "baku": 73, "singapore": 0, "austin": 30, "mexico_city": -8,
    "sao_paulo": 90, "las_vegas": 0, "lusail": 0, "yas_marina": 99,
    "suzuka": 0,
}

TRACK_DISPLAY_NAMES = {
    "melbourne": "MELBOURNE", "shanghai": "SHANGHAI", "suzuka": "SUZUKA",
    "sakhir": "BAHRAIN", "jeddah": "JEDDAH", "miami": "MIAMI",
    "montreal": "MONTREAL", "monaco": "MONACO", "barcelona": "BARCELONA",
    "spielberg": "SPIELBERG", "silverstone": "SILVERSTONE", "spa": "SPA",
    "hungaroring": "HUNGARORING", "zandvoort": "ZANDVOORT", "monza": "MONZA",
    "madrid": "MADRID", "baku": "BAKU", "singapore": "SINGAPORE",
    "austin": "AUSTIN", "mexico_city": "MEXICO CITY", "sao_paulo": "SAO PAULO",
    "las_vegas": "LAS VEGAS", "lusail": "LUSAIL", "yas_marina": "YAS MARINA",
}

CANVAS_W = 800
CANVAS_H = 600
MARGIN = 50

SAMPLES_PER_SEGMENT = 20
CURVATURE_WINDOW = 7

# =========================================================================== #
# SVG Path 解析（支持隐式重复命令）
# =========================================================================== #

def tokenize_path(d: str) -> list[tuple[str, list[float]]]:
    """将SVG path的d属性解析为(命令, 参数列表)列表。

    关键：支持隐式重复命令——当一个命令后面跟多组数值时，
    每组数值代表一个隐式的相同命令。例如：
      C 1,2,3,4,5,6,7,8,9,10,11,12
    等价于：
      C 1,2,3,4,5,6 C 7,8,9,10,11,12
    """
    # 正则：匹配命令字母 或 数值（含负号、小数、科学计数法、省略前导0如.5）
    # SVG path允许 .5 表示 0.5，所以数值格式为：-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?
    token_re = re.compile(r'([MLCSQHVZAmlcsqhvza])|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)')
    tokens = token_re.findall(d)

    # 先拆成命令+数值序列
    raw_seq: list[tuple[str, list[float]]] = []
    current_cmd = None
    current_nums: list[float] = []

    for cmd_char, num_str in tokens:
        if cmd_char:
            # 遇到新命令，先保存前一个命令
            if current_cmd is not None:
                raw_seq.append((current_cmd, current_nums))
            current_cmd = cmd_char
            current_nums = []
        else:
            current_nums.append(float(num_str))

    # 保存最后一个命令
    if current_cmd is not None:
        raw_seq.append((current_cmd, current_nums))

    # 展开隐式重复命令
    # 每个命令的参数数量：M/m=2, L/l=2, C/c=6, S/s=4, Q/q=4, T/t=2, H/h=1, V/v=1, Z/z=0, A/a=7
    PARAM_COUNT = {
        'M': 2, 'm': 2, 'L': 2, 'l': 2,
        'C': 6, 'c': 6, 'S': 4, 's': 4,
        'Q': 4, 'q': 4, 'T': 2, 't': 2,
        'H': 1, 'h': 1, 'V': 1, 'v': 1,
        'Z': 0, 'z': 0,
        'A': 7, 'a': 7,
    }

    expanded: list[tuple[str, list[float]]] = []
    for cmd, nums in raw_seq:
        pc = PARAM_COUNT.get(cmd, 0)
        if pc == 0:
            expanded.append((cmd, []))
            continue
        # 特殊处理：M命令后续隐式重复的是L命令（不是M）
        if cmd == 'M' and len(nums) > 2:
            expanded.append(('M', nums[:2]))
            for i in range(2, len(nums), 2):
                expanded.append(('L', nums[i:i+2]))
        elif cmd == 'm' and len(nums) > 2:
            expanded.append(('m', nums[:2]))
            for i in range(2, len(nums), 2):
                expanded.append(('l', nums[i:i+2]))
        else:
            # 按参数数量分组
            for i in range(0, len(nums), pc):
                group = nums[i:i+pc]
                if len(group) == pc:
                    expanded.append((cmd, group))

    return expanded


def path_to_points(d: str) -> list[tuple[float, float]]:
    """将SVG path的d属性转换为采样点列表。

    支持M/L/C/c/S/s/Q/q/T/t/H/h/V/v/Z/z命令。
    对贝塞尔曲线进行细分采样。
    """
    commands = tokenize_path(d)
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    last_ctrl = None  # 上一个C/c/S/s命令的控制点（用于S/s平滑曲线）

    for cmd, nums in commands:
        if cmd in ('M', 'm'):
            if cmd == 'M':
                current = (nums[0], nums[1])
            else:
                current = (current[0] + nums[0], current[1] + nums[1])
            start = current
            points.append(current)
            last_ctrl = None

        elif cmd in ('L', 'l'):
            if cmd == 'L':
                current = (nums[0], nums[1])
            else:
                current = (current[0] + nums[0], current[1] + nums[1])
            points.append(current)
            last_ctrl = None

        elif cmd in ('H', 'h'):
            if cmd == 'H':
                current = (nums[0], current[1])
            else:
                current = (current[0] + nums[0], current[1])
            points.append(current)
            last_ctrl = None

        elif cmd in ('V', 'v'):
            if cmd == 'V':
                current = (current[0], nums[0])
            else:
                current = (current[0], current[1] + nums[0])
            points.append(current)
            last_ctrl = None

        elif cmd in ('C', 'c'):
            p0 = current
            if cmd == 'C':
                p1 = (nums[0], nums[1])
                p2 = (nums[2], nums[3])
                p3 = (nums[4], nums[5])
            else:
                p1 = (current[0] + nums[0], current[1] + nums[1])
                p2 = (current[0] + nums[2], current[1] + nums[3])
                p3 = (current[0] + nums[4], current[1] + nums[5])
            for t_step in range(1, SAMPLES_PER_SEGMENT + 1):
                t = t_step / SAMPLES_PER_SEGMENT
                x = ((1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] +
                     3*(1-t)*t**2 * p2[0] + t**3 * p3[0])
                y = ((1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] +
                     3*(1-t)*t**2 * p2[1] + t**3 * p3[1])
                points.append((x, y))
            current = p3
            last_ctrl = p2

        elif cmd in ('S', 's'):
            # 平滑三次贝塞尔：控制点1 = current关于last_ctrl的镜像
            p0 = current
            if last_ctrl is not None:
                p1 = (2 * current[0] - last_ctrl[0], 2 * current[1] - last_ctrl[1])
            else:
                p1 = current
            if cmd == 'S':
                p2 = (nums[0], nums[1])
                p3 = (nums[2], nums[3])
            else:
                p2 = (current[0] + nums[0], current[1] + nums[1])
                p3 = (current[0] + nums[2], current[1] + nums[3])
            for t_step in range(1, SAMPLES_PER_SEGMENT + 1):
                t = t_step / SAMPLES_PER_SEGMENT
                x = ((1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] +
                     3*(1-t)*t**2 * p2[0] + t**3 * p3[0])
                y = ((1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] +
                     3*(1-t)*t**2 * p2[1] + t**3 * p3[1])
                points.append((x, y))
            current = p3
            last_ctrl = p2

        elif cmd in ('Q', 'q'):
            p0 = current
            if cmd == 'Q':
                p1 = (nums[0], nums[1])
                p2 = (nums[2], nums[3])
            else:
                p1 = (current[0] + nums[0], current[1] + nums[1])
                p2 = (current[0] + nums[2], current[1] + nums[3])
            for t_step in range(1, SAMPLES_PER_SEGMENT + 1):
                t = t_step / SAMPLES_PER_SEGMENT
                x = ((1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0])
                y = ((1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1])
                points.append((x, y))
            current = p2
            last_ctrl = p1

        elif cmd in ('T', 't'):
            p0 = current
            if last_ctrl is not None:
                p1 = (2 * current[0] - last_ctrl[0], 2 * current[1] - last_ctrl[1])
            else:
                p1 = current
            if cmd == 'T':
                p2 = (nums[0], nums[1])
            else:
                p2 = (current[0] + nums[0], current[1] + nums[1])
            for t_step in range(1, SAMPLES_PER_SEGMENT + 1):
                t = t_step / SAMPLES_PER_SEGMENT
                x = ((1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0])
                y = ((1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1])
                points.append((x, y))
            current = p2
            last_ctrl = p1

        elif cmd in ('A', 'a'):
            # 弧线命令: rx ry x-axis-rotation large-arc-flag sweep-flag x y
            rx, ry, x_rot, large_arc, sweep, ex, ey = nums
            p0 = current
            if cmd == 'A':
                p1 = (ex, ey)
            else:
                p1 = (current[0] + ex, current[1] + ey)

            # 使用SVG弧线参数化公式将弧线转换为采样点
            # 参考: https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
            if rx == 0 or ry == 0:
                # 退化为直线
                dist = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
                n_steps = max(2, int(dist / 5))
                for t_step in range(1, n_steps + 1):
                    t = t_step / n_steps
                    x = p0[0] + t * (p1[0] - p0[0])
                    y = p0[1] + t * (p1[1] - p0[1])
                    points.append((x, y))
            else:
                # 确保 rx, ry 为正
                rx = abs(rx)
                ry = abs(ry)
                # 角度转弧度
                phi = x_rot * math.pi / 180.0

                # 步骤1: 计算中间点
                dx = (p0[0] - p1[0]) / 2.0
                dy = (p0[1] - p1[1]) / 2.0
                # 旋转
                x1p = math.cos(phi) * dx + math.sin(phi) * dy
                y1p = -math.sin(phi) * dx + math.cos(phi) * dy

                # 步骤2: 修正半径
                rx2 = rx * rx
                ry2 = ry * ry
                x1p2 = x1p * x1p
                y1p2 = y1p * y1p
                # 确保 rx, ry 足够大
                lam = x1p2 / rx2 + y1p2 / ry2
                if lam > 1:
                    s = math.sqrt(lam)
                    rx *= s
                    ry *= s
                    rx2 = rx * rx
                    ry2 = ry * ry

                # 步骤3: 计算中心点
                sign = -1 if large_arc == sweep else 1
                num = rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2
                den = rx2 * y1p2 + ry2 * x1p2
                if den == 0:
                    cxp = 0.0
                    cyp = 0.0
                else:
                    factor = sign * math.sqrt(max(0, num / den))
                    cxp = factor * (rx * y1p / ry)
                    cyp = factor * (-ry * x1p / rx)

                # 步骤4: 旋转回原坐标系
                ccx = math.cos(phi) * cxp - math.sin(phi) * cyp + (p0[0] + p1[0]) / 2.0
                ccy = math.sin(phi) * cxp + math.cos(phi) * cyp + (p0[1] + p1[1]) / 2.0

                # 步骤5: 计算起始角和扫角
                def angle_between(ux, uy, vx, vy):
                    dot = ux * vx + uy * vy
                    len_u = math.sqrt(ux * ux + uy * uy)
                    len_v = math.sqrt(vx * vx + vy * vy)
                    if len_u == 0 or len_v == 0:
                        return 0.0
                    cos_a = max(-1.0, min(1.0, dot / (len_u * len_v)))
                    a = math.acos(cos_a)
                    if ux * vy - uy * vx < 0:
                        a = -a
                    return a

                theta1 = angle_between(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
                dtheta = angle_between((x1p - cxp) / rx, (y1p - cyp) / ry,
                                       (-x1p - cxp) / rx, (-y1p - cyp) / ry)

                if sweep == 0 and dtheta > 0:
                    dtheta -= 2 * math.pi
                elif sweep == 1 and dtheta < 0:
                    dtheta += 2 * math.pi

                # 步骤6: 采样弧线
                dist = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
                n_steps = max(4, int(dist / 3))  # 每3单位一个采样点
                for t_step in range(1, n_steps + 1):
                    t = t_step / n_steps
                    a = theta1 + t * dtheta
                    # 椭圆上的点
                    ex_p = rx * math.cos(a)
                    ey_p = ry * math.sin(a)
                    # 旋转
                    x = math.cos(phi) * ex_p - math.sin(phi) * ey_p + ccx
                    y = math.sin(phi) * ex_p + math.cos(phi) * ey_p + ccy
                    points.append((x, y))

            current = p1
            last_ctrl = None

        elif cmd in ('Z', 'z'):
            if current != start:
                points.append(start)
            current = start
            last_ctrl = None

    return points


def compute_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def scale_points(
    points: list[tuple[float, float]],
    target_w: float, target_h: float, margin: float,
) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = compute_bbox(points)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    if bbox_w <= 0 or bbox_h <= 0:
        return points
    avail_w = target_w - 2 * margin
    avail_h = target_h - 2 * margin
    scale = min(avail_w / bbox_w, avail_h / bbox_h)
    scaled_w = bbox_w * scale
    scaled_h = bbox_h * scale
    offset_x = (target_w - scaled_w) / 2 - min_x * scale
    offset_y = (target_h - scaled_h) / 2 - min_y * scale
    return [(p[0] * scale + offset_x, p[1] * scale + offset_y) for p in points]


# =========================================================================== #
# 曲率分析与弯道检测
# =========================================================================== #

def compute_curvatures(points: list[tuple[float, float]]) -> list[float]:
    """计算每个点的曲率（转角的绝对值）。"""
    n = len(points)
    if n < 3:
        return [0.0] * n

    curvatures = [0.0] * n
    for i in range(n):
        i_prev = (i - CURVATURE_WINDOW) % n
        i_next = (i + CURVATURE_WINDOW) % n
        p_prev = points[i_prev]
        p_curr = points[i]
        p_next = points[i_next]

        dx1 = p_curr[0] - p_prev[0]
        dy1 = p_curr[1] - p_prev[1]
        dx2 = p_next[0] - p_curr[0]
        dy2 = p_next[1] - p_curr[1]

        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)

        if len1 < 1e-6 or len2 < 1e-6:
            curvatures[i] = 0.0
            continue

        cross = dx1 * dy2 - dy1 * dx2
        sin_angle = cross / (len1 * len2)
        sin_angle = max(-1.0, min(1.0, sin_angle))
        curvatures[i] = abs(sin_angle)

    return curvatures


def detect_corners(
    points: list[tuple[float, float]],
    n_corners: int,
) -> list[tuple[float, float]]:
    """通过曲率分析检测弯道位置，确保检测到恰好n_corners个弯道。

    策略：
    1. 计算所有点的曲率
    2. 用非极大值抑制找局部最大值候选
    3. 按曲率值降序排序候选
    4. 贪心选择：从最高曲率开始，确保间距，选n_corners个
    5. 如果候选不够，降低阈值重新找
    """
    curvatures = compute_curvatures(points)
    n = len(points)

    # 计算路径总长和累积弧长
    total_len = 0.0
    arc_lens = [0.0]
    for i in range(n - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        sl = math.sqrt(dx**2 + dy**2)
        total_len += sl
        arc_lens.append(arc_lens[-1] + sl)
    # 闭合路径的最后一段
    if n > 1:
        dx = points[0][0] - points[-1][0]
        dy = points[0][1] - points[-1][1]
        total_len += math.sqrt(dx**2 + dy**2)

    # 非极大值抑制窗口（以弧长计）
    nms_window = max(3, int(n * 0.005))  # 约0.5%的路径长度

    # 找局部最大值
    candidates: list[tuple[float, int]] = []
    for i in range(n):
        is_local_max = True
        for j in range(max(0, i - nms_window), min(n, i + nms_window + 1)):
            if j != i and curvatures[j] > curvatures[i]:
                is_local_max = False
                break
        if is_local_max and curvatures[i] > 0.001:
            candidates.append((curvatures[i], i))

    # 按曲率值降序排序
    candidates.sort(key=lambda x: -x[0])

    # 最小间距（弧长）：路径总长 / (n_corners * 1.5)
    # 确保弯道之间有足够间距，但不会太严格
    min_spacing = total_len / (n_corners * 2.0) if n_corners > 0 else 0

    selected: list[int] = []

    # 第一轮：严格间距
    for curv, idx in candidates:
        if len(selected) >= n_corners:
            break
        too_close = False
        for s in selected:
            arc_diff = abs(arc_lens[idx] - arc_lens[s])
            arc_diff = min(arc_diff, total_len - arc_diff)
            if arc_diff < min_spacing:
                too_close = True
                break
        if not too_close:
            selected.append(idx)

    # 第二轮：放宽间距到一半
    if len(selected) < n_corners:
        min_spacing *= 0.5
        for curv, idx in candidates:
            if len(selected) >= n_corners:
                break
            if idx in selected:
                continue
            too_close = False
            for s in selected:
                arc_diff = abs(arc_lens[idx] - arc_lens[s])
                arc_diff = min(arc_diff, total_len - arc_diff)
                if arc_diff < min_spacing:
                    too_close = True
                    break
            if not too_close:
                selected.append(idx)

    # 第三轮：放宽间距到1/4
    if len(selected) < n_corners:
        min_spacing *= 0.5
        for curv, idx in candidates:
            if len(selected) >= n_corners:
                break
            if idx in selected:
                continue
            too_close = False
            for s in selected:
                arc_diff = abs(arc_lens[idx] - arc_lens[s])
                arc_diff = min(arc_diff, total_len - arc_diff)
                if arc_diff < min_spacing:
                    too_close = True
                    break
            if not too_close:
                selected.append(idx)

    # 第四轮：无间距限制，直接按曲率排序取
    if len(selected) < n_corners:
        for curv, idx in candidates:
            if len(selected) >= n_corners:
                break
            if idx not in selected:
                selected.append(idx)

    # 第五轮：如果候选不够，在整个路径上均匀补充
    if len(selected) < n_corners:
        # 在曲率最高的区域附近均匀采样
        remaining = n_corners - len(selected)
        # 按弧长均匀分布补充点
        for k in range(remaining):
            target_arc = total_len * (k + 0.5) / remaining
            # 找最接近target_arc的点
            best_idx = 0
            best_diff = abs(arc_lens[0] - target_arc)
            for i in range(1, n):
                diff = abs(arc_lens[i] - target_arc)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            if best_idx not in selected:
                selected.append(best_idx)

    # 按路径顺序排列
    selected.sort()
    return [points[i] for i in selected[:n_corners]]


# =========================================================================== #
# SVG 生成
# =========================================================================== #

def generate_f1opt_svg(
    path_d_scaled: str,
    corner_points: list[tuple[float, float]],
    start_point: tuple[float, float],
    display_name: str,
) -> str:
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
        f'width="{CANVAS_W}" height="{CANVAS_H}">'
    )
    lines.append("  <defs>")
    lines.append("    <style>")
    lines.append("      .track-bg { fill: #0d0d12; }")
    lines.append(
        "      .track-line { fill: none; stroke: #3a3e48; stroke-width: 3; "
        "stroke-linecap: round; stroke-linejoin: round; }"
    )
    lines.append(
        "      .track-line-accent { fill: none; stroke: rgba(59,158,255,0.20); "
        "stroke-width: 1.5; stroke-linecap: round; stroke-dasharray: 6,4; }"
    )
    lines.append(
        "      .track-edge { fill: none; stroke: #3a3e48; stroke-width: 1; "
        "stroke-linecap: round; stroke-linejoin: round; opacity: 0.6; }"
    )
    lines.append("    </style>")
    lines.append("  </defs>")
    lines.append(f'  <rect class="track-bg" width="{CANVAS_W}" height="{CANVAS_H}" rx="8"/>')
    lines.append(f'  <path class="track-edge" d="{path_d_scaled}"/>')
    lines.append(f'  <path class="track-line" d="{path_d_scaled}"/>')
    lines.append(f'  <path class="track-line-accent" d="{path_d_scaled}"/>')

    sx, sy = start_point
    lines.append(
        f'  <rect x="{sx - 5:.1f}" y="{sy - 5:.1f}" width="10" height="10" '
        f'fill="#FF1801" rx="1"/>'
    )

    # 标注文字防重叠：检查所有弯道对的像素间距，聚集的弯道交替放在圆点上方/下方
    LABEL_MIN_SPACING = 18.0  # 标注文字最小间距阈值
    n_corners_total = len(corner_points)
    label_offsets = [-1] * n_corners_total  # 默认全部上方

    # 对每对弯道检查像素间距，如果接近则交替标注方向
    for i in range(n_corners_total):
        cx_i, cy_i = corner_points[i]
        for j in range(i + 1, n_corners_total):
            cx_j, cy_j = corner_points[j]
            dist = math.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
            if dist < LABEL_MIN_SPACING:
                # 两个弯道标注会重叠，把j放下方（如果i在上方）
                if label_offsets[i] == -1:
                    label_offsets[j] = 1
                # 如果i已经在下方，j保持上方（默认-1）

    for i, (cx, cy) in enumerate(corner_points, 1):
        lines.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="none" '
            f'stroke="#3B9EFF" stroke-width="1.5" opacity="0.85"/>'
        )
        offset_y = label_offsets[i - 1] * 9
        lines.append(
            f'  <text x="{cx:.1f}" y="{cy + offset_y:.1f}" text-anchor="middle" '
            f'fill="#9494a8" font-size="9" font-family="monospace">{i}</text>'
        )

    lines.append(
        f'  <text x="{CANVAS_W // 2}" y="{CANVAS_H - 12}" text-anchor="middle" '
        f'fill="#5c5c70" font-size="11" font-family="monospace" '
        f'letter-spacing="2">{display_name}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def rotate_path_d(d: str, angle_deg: float) -> str:
    """旋转path d中的所有坐标，围绕原点(0,0)旋转angle_deg度。

    旋转公式：x'=x*cos-y*sin, y'=x*sin+y*cos
    旋转是线性变换，相对命令和绝对命令用相同公式旋转位移/坐标。

    特殊处理：
    - H/h命令（水平线，只有x）：旋转后变为L/l命令（需要跟踪当前y）
    - V/v命令（垂直线，只有y）：旋转后变为L/l命令（需要跟踪当前x）
    - A/a命令（弧线）：rx,ry不变；x-axis-rotation加上angle_deg；large-arc,sweep不变；x,y旋转
    - SVG规范规定path第一个命令如果是m，应当作M处理（绝对）
    - 需要跟踪当前点(cur_x,cur_y)和子路径起点(start_x,start_y)用于Z/z命令
    """
    if angle_deg == 0:
        return d

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    def rot(x: float, y: float) -> tuple[float, float]:
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    commands = tokenize_path(d)
    result_parts: list[str] = []
    is_first_cmd = True
    cur_x, cur_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0

    for cmd, nums in commands:
        if cmd in ('Z', 'z'):
            result_parts.append(cmd)
            cur_x, cur_y = start_x, start_y
            is_first_cmd = False
            continue

        # SVG规范规定path第一个命令如果是m，应当作M处理（绝对）
        is_absolute = cmd.isupper() or (is_first_cmd and cmd == 'm')

        if cmd in ('M', 'm'):
            # M/m: x y
            if is_absolute:
                nx, ny = rot(nums[0], nums[1])
                result_parts.append(f"M {nx:.2f} {ny:.2f}")
                cur_x, cur_y = nums[0], nums[1]
            else:
                dx, dy = rot(nums[0], nums[1])
                result_parts.append(f"m {dx:.2f} {dy:.2f}")
                cur_x += nums[0]
                cur_y += nums[1]
            start_x, start_y = cur_x, cur_y

        elif cmd in ('L', 'l'):
            if is_absolute:
                nx, ny = rot(nums[0], nums[1])
                result_parts.append(f"L {nx:.2f} {ny:.2f}")
                cur_x, cur_y = nums[0], nums[1]
            else:
                dx, dy = rot(nums[0], nums[1])
                result_parts.append(f"l {dx:.2f} {dy:.2f}")
                cur_x += nums[0]
                cur_y += nums[1]

        elif cmd in ('C', 'c'):
            # C/c: x1 y1 x2 y2 x y
            if is_absolute:
                nx1, ny1 = rot(nums[0], nums[1])
                nx2, ny2 = rot(nums[2], nums[3])
                nx, ny = rot(nums[4], nums[5])
                result_parts.append(
                    f"C {nx1:.2f} {ny1:.2f} {nx2:.2f} {ny2:.2f} {nx:.2f} {ny:.2f}"
                )
                cur_x, cur_y = nums[4], nums[5]
            else:
                dx1, dy1 = rot(nums[0], nums[1])
                dx2, dy2 = rot(nums[2], nums[3])
                dx, dy = rot(nums[4], nums[5])
                result_parts.append(
                    f"c {dx1:.2f} {dy1:.2f} {dx2:.2f} {dy2:.2f} {dx:.2f} {dy:.2f}"
                )
                cur_x += nums[4]
                cur_y += nums[5]

        elif cmd in ('S', 's'):
            # S/s: x2 y2 x y
            if is_absolute:
                nx2, ny2 = rot(nums[0], nums[1])
                nx, ny = rot(nums[2], nums[3])
                result_parts.append(f"S {nx2:.2f} {ny2:.2f} {nx:.2f} {ny:.2f}")
                cur_x, cur_y = nums[2], nums[3]
            else:
                dx2, dy2 = rot(nums[0], nums[1])
                dx, dy = rot(nums[2], nums[3])
                result_parts.append(f"s {dx2:.2f} {dy2:.2f} {dx:.2f} {dy:.2f}")
                cur_x += nums[2]
                cur_y += nums[3]

        elif cmd in ('Q', 'q'):
            # Q/q: x1 y1 x y
            if is_absolute:
                nx1, ny1 = rot(nums[0], nums[1])
                nx, ny = rot(nums[2], nums[3])
                result_parts.append(f"Q {nx1:.2f} {ny1:.2f} {nx:.2f} {ny:.2f}")
                cur_x, cur_y = nums[2], nums[3]
            else:
                dx1, dy1 = rot(nums[0], nums[1])
                dx, dy = rot(nums[2], nums[3])
                result_parts.append(f"q {dx1:.2f} {dy1:.2f} {dx:.2f} {dy:.2f}")
                cur_x += nums[2]
                cur_y += nums[3]

        elif cmd in ('T', 't'):
            # T/t: x y
            if is_absolute:
                nx, ny = rot(nums[0], nums[1])
                result_parts.append(f"T {nx:.2f} {ny:.2f}")
                cur_x, cur_y = nums[0], nums[1]
            else:
                dx, dy = rot(nums[0], nums[1])
                result_parts.append(f"t {dx:.2f} {dy:.2f}")
                cur_x += nums[0]
                cur_y += nums[1]

        elif cmd in ('H', 'h'):
            # H/h: x -> 旋转后变为L/l（旋转后不再是水平的）
            if cmd == 'H':
                # 绝对：目标点为(nums[0], cur_y)，旋转后
                nx, ny = rot(nums[0], cur_y)
                result_parts.append(f"L {nx:.2f} {ny:.2f}")
                cur_x = nums[0]
            else:
                # 相对：位移为(nums[0], 0)，旋转后
                dx, dy = rot(nums[0], 0.0)
                result_parts.append(f"l {dx:.2f} {dy:.2f}")
                cur_x += nums[0]

        elif cmd in ('V', 'v'):
            # V/v: y -> 旋转后变为L/l（旋转后不再是垂直的）
            if cmd == 'V':
                # 绝对：目标点为(cur_x, nums[0])，旋转后
                nx, ny = rot(cur_x, nums[0])
                result_parts.append(f"L {nx:.2f} {ny:.2f}")
                cur_y = nums[0]
            else:
                # 相对：位移为(0, nums[0])，旋转后
                dx, dy = rot(0.0, nums[0])
                result_parts.append(f"l {dx:.2f} {dy:.2f}")
                cur_y += nums[0]

        elif cmd in ('A', 'a'):
            # A/a: rx ry x-axis-rotation large-arc-flag sweep-flag x y
            # rx,ry不变；x-axis-rotation加上angle_deg；large-arc,sweep不变；x,y旋转
            rx = nums[0]
            ry = nums[1]
            x_rot = nums[2] + angle_deg
            large_arc = nums[3]
            sweep = nums[4]
            if is_absolute:
                nx, ny = rot(nums[5], nums[6])
                result_parts.append(
                    f"A {rx:.2f} {ry:.2f} {x_rot:.2f} {large_arc:.0f} {sweep:.0f} {nx:.2f} {ny:.2f}"
                )
                cur_x, cur_y = nums[5], nums[6]
            else:
                dx, dy = rot(nums[5], nums[6])
                result_parts.append(
                    f"a {rx:.2f} {ry:.2f} {x_rot:.2f} {large_arc:.0f} {sweep:.0f} {dx:.2f} {dy:.2f}"
                )
                cur_x += nums[5]
                cur_y += nums[6]

        is_first_cmd = False

    return " ".join(result_parts)


def rotate_points(points: list[tuple[float, float]], angle_deg: float) -> list[tuple[float, float]]:
    """旋转采样点列表，围绕原点(0,0)旋转angle_deg度。"""
    if angle_deg == 0:
        return points
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]


def scale_path_d(d: str, scale: float, offset_x: float, offset_y: float) -> str:
    """将path的d属性中的所有坐标按scale缩放并偏移。

    注意：A/a命令的参数为 rx ry x-rot large-arc sweep x y，
    其中 x-rot/large-arc/sweep 不是坐标，不能缩放或偏移。
    """
    commands = tokenize_path(d)
    result_parts = []
    is_first_cmd = True
    for cmd, nums in commands:
        if cmd in ('Z', 'z'):
            result_parts.append(cmd)
            is_first_cmd = False
            continue
        if cmd in ('A', 'a'):
            # A/a: rx ry x-rot large-arc sweep x y
            rx = nums[0] * scale
            ry = nums[1] * scale
            x_rot = nums[2]  # 不变
            large_arc = nums[3]  # 不变（0或1）
            sweep = nums[4]  # 不变（0或1）
            if cmd == 'A':
                x = nums[5] * scale + offset_x
                y = nums[6] * scale + offset_y
            else:
                x = nums[5] * scale  # 相对坐标只需缩放
                y = nums[6] * scale
            nums_str = f"{rx:.2f} {ry:.2f} {x_rot:.2f} {large_arc:.0f} {sweep:.0f} {x:.2f} {y:.2f}"
            result_parts.append(f"{cmd} {nums_str}")
            is_first_cmd = False
            continue
        # 其他命令：偶数索引=x坐标，奇数索引=y坐标
        # 绝对命令（大写）：坐标 = 原坐标 * scale + offset
        # 相对命令（小写）：坐标 = 原坐标 * scale（只缩放，不加offset）
        # 例外：SVG规范规定path第一个命令如果是m，应当作M处理
        is_absolute = cmd.isupper() or (is_first_cmd and cmd == 'm')
        scaled_nums = []
        for i, num in enumerate(nums):
            if i % 2 == 0:
                val = num * scale + (offset_x if is_absolute else 0.0)
            else:
                val = num * scale + (offset_y if is_absolute else 0.0)
            scaled_nums.append(val)
        nums_str = " ".join(f"{v:.2f}" for v in scaled_nums)
        result_parts.append(f"{cmd} {nums_str}")
        is_first_cmd = False
    return " ".join(result_parts)


# =========================================================================== #
# 主流程
# =========================================================================== #

def download_svg(layout_id: str, max_retries: int = 3) -> str:
    # 优先使用本地缓存
    cache_path = Path(__file__).resolve().parent / "raw_svgs" / f"{layout_id}.svg"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    # 本地缓存不存在时从网上下载
    url = BASE_URL + layout_id + ".svg"
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            last_err = e
            print(f"    下载重试 {attempt + 1}/{max_retries}...", end=" ")
    raise last_err


def get_legacy_corner_distances(track_id: str) -> list[tuple[float, float]] | None:
    """从legacy track_maps获取弯道的距离信息（distance_start, distance_end）。

    弯道的distance_start/distance_end表示弯道在赛道上的距离（米），
    用于在SVG path采样点上通过距离比例定位弯道位置。

    Args:
        track_id: 赛道ID

    Returns:
        [(distance_start, distance_end), ...] 列表；如果track_id不在legacy中返回None。
    """
    tm = _LEGACY_TRACK_MAPS.get(track_id)
    if not tm:
        return None
    corners_sorted = sorted(tm.corners, key=lambda c: c.corner_id)
    return [(c.distance_start, c.distance_end) for c in corners_sorted]


def map_corners_by_distance(
    corner_distances: list[tuple[float, float]],
    scaled_points: list[tuple[float, float]],
    track_total_m: float,
) -> list[tuple[float, float]]:
    """用距离比例在SVG path采样点上定位弯道。

    弯道的distance_start/distance_end表示弯道在赛道上的距离（米），
    用弯道中点的距离比例在SVG path的采样点上找到对应位置。
    这样弯道标注会准确落在赛道线上。

    Args:
        corner_distances: [(distance_start, distance_end), ...]
        scaled_points: 缩放后的赛道采样点
        track_total_m: legacy赛道的总长度（米）

    Returns:
        弯道坐标列表[(x, y), ...]
    """
    # 计算采样点的累积弧长
    arc_lens = [0.0]
    for i in range(len(scaled_points) - 1):
        dx = scaled_points[i+1][0] - scaled_points[i][0]
        dy = scaled_points[i+1][1] - scaled_points[i][1]
        arc_lens.append(arc_lens[-1] + math.sqrt(dx*dx + dy*dy))
    # 闭合路径的最后一段
    dx = scaled_points[0][0] - scaled_points[-1][0]
    dy = scaled_points[0][1] - scaled_points[-1][1]
    total_px = arc_lens[-1] + math.sqrt(dx*dx + dy*dy)

    result = []
    for dist_start, dist_end in corner_distances:
        corner_mid = (dist_start + dist_end) / 2
        ratio = corner_mid / track_total_m
        # 在采样点上找到对应比例的位置（插值）
        target = ratio * total_px
        lo, hi = 0, len(arc_lens) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if arc_lens[mid] < target:
                lo = mid
            else:
                hi = mid
        seg_len = arc_lens[hi] - arc_lens[lo]
        if seg_len < 1e-9:
            result.append(scaled_points[lo])
        else:
            t = (target - arc_lens[lo]) / seg_len
            x = scaled_points[lo][0] + t * (scaled_points[hi][0] - scaled_points[lo][0])
            y = scaled_points[lo][1] + t * (scaled_points[hi][1] - scaled_points[lo][1])
            result.append((x, y))

    return result


def _decluster_corners(
    pts: list[tuple[float, float]],
    scaled_points: list[tuple[float, float]],
    min_spacing: float = 15.0,
) -> list[tuple[float, float]]:
    """将聚集的弯道坐标沿赛道方向分散，确保视觉上可区分。

    策略：
    1. 对每个弯道，检查与所有之前弯道的距离（不只是前一个）
    2. 如果距离<min_spacing，沿赛道前进方向搜索满足间距的采样点
    3. 如果前进方向找不到，沿后退方向搜索
    4. 如果都找不到，沿赛道法线方向偏移
    """
    n = len(pts)
    if n < 2:
        return pts

    result = list(pts)

    # 预计算：每个弯道在赛道上最近的采样点索引
    track_indices = []
    for px, py in pts:
        best_idx = 0
        best_d = float('inf')
        for j, (tx, ty) in enumerate(scaled_points):
            d = (px - tx)**2 + (py - ty)**2
            if d < best_d:
                best_d = d
                best_idx = j
        track_indices.append(best_idx)

    def dist_to_all_prev(idx, x, y):
        """检查点(x,y)与result[0..idx-1]中所有点的最小距离。"""
        min_d = float('inf')
        for k in range(idx):
            dx = x - result[k][0]
            dy = y - result[k][1]
            d = (dx*dx + dy*dy)**0.5
            if d < min_d:
                min_d = d
        return min_d

    for i in range(1, n):
        # 检查与所有之前弯道的距离
        if dist_to_all_prev(i, result[i][0], result[i][1]) >= min_spacing:
            continue

        # 需要偏移：沿赛道方向搜索满足min_spacing的位置
        base_idx = track_indices[i]
        best_pos = None
        n_sp = len(scaled_points)

        # 沿赛道前进方向搜索
        for direction in [1, -1]:
            search_idx = base_idx
            accumulated = 0.0
            steps = 0
            max_steps = n_sp  # 最多搜索一圈

            while steps < max_steps:
                next_idx = (search_idx + direction) % n_sp
                seg_len = (
                    (scaled_points[next_idx][0] - scaled_points[search_idx][0])**2 +
                    (scaled_points[next_idx][1] - scaled_points[search_idx][1])**2
                )**0.5
                accumulated += seg_len
                search_idx = next_idx
                steps += 1

                if accumulated >= min_spacing:
                    candidate = scaled_points[search_idx]
                    if dist_to_all_prev(i, candidate[0], candidate[1]) >= min_spacing:
                        best_pos = candidate
                        break

            if best_pos is not None:
                break

        if best_pos is not None:
            result[i] = best_pos
        else:
            # 沿赛道方向找不到合适位置，沿法线方向偏移
            base_ti = track_indices[i]
            next_ti = (base_ti + 1) % n_sp
            prev_ti = (base_ti - 1) % n_sp
            tx = scaled_points[next_ti][0] - scaled_points[prev_ti][0]
            ty = scaled_points[next_ti][1] - scaled_points[prev_ti][1]
            tl = (tx*tx + ty*ty)**0.5
            if tl > 0:
                tx /= tl
                ty /= tl
            else:
                tx, ty = 1.0, 0.0
            # 法线方向（垂直于切线）
            nx, ny = -ty, tx
            result[i] = (
                result[i][0] + nx * min_spacing,
                result[i][1] + ny * min_spacing,
            )

    # 确保所有坐标在画布范围内
    result = [
        (max(10.0, min(790.0, x)), max(10.0, min(590.0, y)))
        for x, y in result
    ]
    return result


def extract_path_d(svg_content: str) -> str:
    root = ET.fromstring(svg_content)
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    if not paths:
        paths = root.findall(".//path")
    if not paths:
        raise ValueError("SVG中没有找到path元素")
    d = paths[0].get("d")
    if not d:
        raise ValueError("path元素没有d属性")
    return d


def process_track(track_id: str, layout_id: str, n_corners: int) -> dict:
    # 1. 下载SVG
    svg_content = download_svg(layout_id)

    # 2. 提取path d属性
    raw_d = extract_path_d(svg_content)

    # 3. 解析为采样点
    raw_points = path_to_points(raw_d)

    # 3.5 应用f1-orientation旋转
    orientation = TRACK_F1_ORIENTATION.get(track_id, 0)
    if orientation != 0:
        raw_d = rotate_path_d(raw_d, orientation)
        raw_points = rotate_points(raw_points, orientation)

    if len(raw_points) < 10:
        raise ValueError(f"采样点太少: {len(raw_points)}")

    # 4. 计算边界框和缩放参数
    min_x, min_y, max_x, max_y = compute_bbox(raw_points)
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    avail_w = CANVAS_W - 2 * MARGIN
    avail_h = CANVAS_H - 2 * MARGIN
    scale = min(avail_w / bbox_w, avail_h / bbox_h)

    scaled_w = bbox_w * scale
    scaled_h = bbox_h * scale
    offset_x = (CANVAS_W - scaled_w) / 2 - min_x * scale
    offset_y = (CANVAS_H - scaled_h) / 2 - min_y * scale

    # 5. 缩放path d属性
    scaled_d = scale_path_d(raw_d, scale, offset_x, offset_y)

    # 6. 缩放采样点
    scaled_points = scale_points(raw_points, CANVAS_W, CANVAS_H, MARGIN)

    # 7. 弯道位置：直接在SVG path采样点上做曲率分析检测弯道
    #    这样弯道标注一定落在赛道线上（因为是从赛道线上检测出来的）
    corner_points = detect_corners(scaled_points, n_corners)


    # 8. 起点位置
    commands = tokenize_path(raw_d)
    start_raw = (0.0, 0.0)
    for cmd, nums in commands:
        if cmd in ('M', 'm') and len(nums) >= 2:
            if cmd == 'M':
                start_raw = (nums[0], nums[1])
            else:
                start_raw = (nums[0], nums[1])  # m的第一个点是绝对的
            break
    start_point = (
        start_raw[0] * scale + offset_x,
        start_raw[1] * scale + offset_y,
    )

    # 9. 生成SVG
    display_name = TRACK_DISPLAY_NAMES.get(track_id, track_id.upper())
    svg_content_out = generate_f1opt_svg(scaled_d, corner_points, start_point, display_name)

    # 10. 写入文件
    output_path = OUTPUT_DIR / f"{track_id}.svg"
    output_path.write_text(svg_content_out, encoding="utf-8")

    return {
        "track_id": track_id,
        "layout_id": layout_id,
        "corner_points": corner_points,
        "start_point": start_point,
        "display_name": display_name,
        "n_points": len(raw_points),
    }


def generate_anchors_file(results: list[dict]) -> str:
    lines = []
    lines.append('"""从 julesr0y/f1-circuits-svg 真实赛道SVG提取的弯道像素坐标。')
    lines.append("")
    lines.append("由 scripts/convert_track_svgs.py 自动生成，请勿手工编辑。")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("# (canvas_width, canvas_height)")
    lines.append("TRACK_CANVAS: dict[str, tuple[int, int]] = {")
    for r in results:
        lines.append(f'    "{r["track_id"]}": ({CANVAS_W}, {CANVAS_H}),')
    lines.append("}")
    lines.append("")
    lines.append("# {track_id: {corner_number: (x_px, y_px)}}")
    lines.append("TRACK_ANCHORS: dict[str, dict[int, tuple[float, float]]] = {")
    for r in results:
        lines.append(f'    "{r["track_id"]}": {{')
        for i, (cx, cy) in enumerate(r["corner_points"], 1):
            lines.append(f'        {i}: ({cx:.1f}, {cy:.1f}),')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("F1OPT 赛道图SVG转换脚本 v2")
    print("参考项目: julesr0y/f1-circuits-svg (CC-BY-4.0)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failed = []

    for track_id, layout_id in TRACK_LAYOUT_MAP.items():
        n_corners = TRACK_CORNERS_COUNT[track_id]
        print(f"\n处理: {track_id} ({layout_id}) — {n_corners}弯道...", end=" ")

        try:
            result = process_track(track_id, layout_id, n_corners)
            results.append(result)
            n_detected = len(result["corner_points"])
            n_pts = result["n_points"]
            print(f"OK ({n_detected}/{n_corners}弯道, {n_pts}采样点)")
            # 每条赛道完成后增量保存anchors文件
            if results:
                anchors_content = generate_anchors_file(results)
                ANCHORS_FILE.write_text(anchors_content, encoding="utf-8")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append((track_id, str(e)))

    # 最终生成 _track_anchors.py
    if results:
        anchors_content = generate_anchors_file(results)
        ANCHORS_FILE.write_text(anchors_content, encoding="utf-8")
        print(f"\n已更新: {ANCHORS_FILE}")

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"完成: {len(results)}/{len(TRACK_LAYOUT_MAP)} 条赛道")
    if failed:
        print(f"失败: {len(failed)} 条")
        for tid, err in failed:
            print(f"  {tid}: {err}")
    else:
        print("全部成功！")

    # 验证弯道检测数量
    print(f"\n弯道检测验证:")
    all_match = True
    for r in results:
        expected = TRACK_CORNERS_COUNT[r["track_id"]]
        detected = len(r["corner_points"])
        status = "OK" if detected == expected else f"MISMATCH ({detected}/{expected})"
        if detected != expected:
            all_match = False
        print(f"  {r['track_id']}: {status}")

    if all_match:
        print("\n所有弯道检测数量匹配！")
    else:
        print("\n部分弯道检测数量不匹配，需要进一步调优。")


if __name__ == "__main__":
    main()
