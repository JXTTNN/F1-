"""车手教练与学习模块 — 自适应驾驶训练、结构化学习路径、技能评估.

本模块**不修改**已有画像/反馈代码, 而是基于 :class:`DriverProfile` 与单圈指标
(``list[dict]``) 生成个性化的训练计划、追踪进度、构建阶段性学习路径并评估
车手技能等级. 所有自然语言输出均为中文, 全程确定性 (无随机性), 与
:mod:`f1opt.feedback.nlg` 的离线叙事风格一致.

公开 API:

- :class:`CoachingPlan` — 训练计划数据容器 (聚焦领域/练习/目标/时长/难度).
- :class:`DriverCoach` — 个性化教练: 评估短板、生成计划、追踪进度、迭代计划.
- :class:`LearningPath` — 结构化学习路径 (5-7 阶段, 含前置依赖).
- :class:`SkillAssessment` — 技能评估 (5 维技能分解 + 等级判定).
- :data:`LEVEL_THRESHOLDS` — 技能等级阈值 (模块级常量).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from f1opt.driver.profile import DriverProfile

__all__ = [
    "LEVEL_THRESHOLDS",
    "CoachingPlan",
    "DriverCoach",
    "LearningPath",
    "SkillAssessment",
]


#: 技能等级阈值 (模块级). overall_skill < 阈值 → 对应等级; EXPERT 为兜底.
LEVEL_THRESHOLDS: dict[str, float] = {
    "ROOKIE": 0.2,
    "AMATEUR": 0.4,
    "INTERMEDIATE": 0.6,
    "ADVANCED": 0.8,
    "EXPERT": 1.0,
}

#: 难度顺序 (用于 next_plan 的难度递进).
_DIFFICULTY_ORDER: tuple[str, ...] = ("easy", "medium", "hard")

#: 难度对应的训练时长 (圈).
_DIFFICULTY_LAPS: dict[str, int] = {"easy": 20, "medium": 15, "hard": 10}

#: 原型 → 默认难度映射 (DEVELOPMENT=easy, RACE_CRAFT=medium, AGGRESSIVE=hard).
_ARCHETYPE_DIFFICULTY: dict[str, str] = {
    "DEVELOPMENT": "easy",
    "RACE_CRAFT": "medium",
    "AGGRESSIVE": "hard",
    "AGGRESSIVE_OVERTAKER": "hard",
    "SMOOTH_OPERATOR": "medium",
    "QUALIFIER": "hard",
    "TIRE_WHISPERER": "medium",
    "WET_SPECIALIST": "medium",
}

#: "数值越低越好" 的指标集合 (用于进度/达标比较). 其余视为 "越高越好".
_LOWER_IS_BETTER: set[str] = {
    "lap_time",
    "lap_time_cv",
    "brake_aggression_cv",
    "sector_1_time",
    "sector_2_time",
    "sector_3_time",
    "sector_1_cv",
    "sector_2_cv",
    "sector_3_cv",
    "tire_wear_score",
    "lockup_proxy",
}

#: 技能维度的中文标签.
_SKILL_LABELS_ZH: dict[str, str] = {
    "braking": "制动控制",
    "cornering": "弯道技术",
    "consistency": "圈速一致性",
    "racecraft": "比赛技巧",
    "tire_mgmt": "轮胎管理",
}

#: 原型对各技能维度的期望水平 (用于 compare_to_archetype).
_ARCHETYPE_EXPECTATIONS: dict[str, dict[str, float]] = {
    "DEVELOPMENT": {"braking": 0.35, "cornering": 0.35, "consistency": 0.30,
                    "racecraft": 0.30, "tire_mgmt": 0.40},
    "RACE_CRAFT": {"braking": 0.60, "cornering": 0.60, "consistency": 0.70,
                   "racecraft": 0.70, "tire_mgmt": 0.60},
    "AGGRESSIVE": {"braking": 0.70, "cornering": 0.65, "consistency": 0.50,
                   "racecraft": 0.70, "tire_mgmt": 0.45},
    "AGGRESSIVE_OVERTAKER": {"braking": 0.70, "cornering": 0.65, "consistency": 0.50,
                             "racecraft": 0.75, "tire_mgmt": 0.45},
    "SMOOTH_OPERATOR": {"braking": 0.65, "cornering": 0.70, "consistency": 0.75,
                        "racecraft": 0.55, "tire_mgmt": 0.70},
    "QUALIFIER": {"braking": 0.70, "cornering": 0.65, "consistency": 0.50,
                  "racecraft": 0.60, "tire_mgmt": 0.50},
    "TIRE_WHISPERER": {"braking": 0.60, "cornering": 0.65, "consistency": 0.70,
                       "racecraft": 0.55, "tire_mgmt": 0.80},
    "WET_SPECIALIST": {"braking": 0.60, "cornering": 0.65, "consistency": 0.65,
                       "racecraft": 0.60, "tire_mgmt": 0.65},
}

#: 短板 → 目标指标默认值 (越低越好的为上限, 越高越好的为下限).
_WEAKNESS_TARGETS: dict[str, tuple[str, float]] = {
    "braking_consistency": ("brake_aggression_cv", 0.05),
    "corner_exit_speed": ("throttle_smoothness", 0.75),
    "lap_time_consistency": ("lap_time_cv", 0.02),
    "sector_1_pace": ("sector_1_time", 30.0),
    "sector_2_pace": ("sector_2_time", 28.5),
    "sector_3_pace": ("sector_3_time", 29.0),
    "tire_management": ("tire_wear_score", 0.4),
}

#: 短板练习库 (name + 中文 description); target_metric/target_value 由计划填充.
_EXERCISE_LIBRARY: dict[str, dict[str, str]] = {
    "braking_consistency": {
        "name": "制动一致性训练",
        "description": "在同一弯道重复练习制动点与制动力, 将制动输入波动控制在 5% 以内, "
                       "建立稳定的肌肉记忆。",
    },
    "corner_exit_speed": {
        "name": "出弯油门控制",
        "description": "练习渐进式油门施加, 在弯心后线性增加油门开度, 提升出弯牵引力与尾速。",
    },
    "lap_time_consistency": {
        "name": "圈速稳定性训练",
        "description": "以稳定节奏连续完成多圈, 目标将圈速变异系数控制在 2% 以内, 减少单圈波动。",
    },
    "sector_1_pace": {
        "name": "第一分段提速",
        "description": "重点优化第一分段的入弯走线与制动点, 通过更晚制动与更早加速提升该段圈速。",
    },
    "sector_2_pace": {
        "name": "第二分段提速",
        "description": "重点优化第二分段的中高速弯走线, 提升弯中最小速度与出弯衔接效率。",
    },
    "sector_3_pace": {
        "name": "第三分段提速",
        "description": "重点优化第三分段, 关注最后一个弯道的出弯速度以最大化直道尾速。",
    },
    "tire_management": {
        "name": "轮胎管理训练",
        "description": "练习柔和的输入与提前制动, 降低横向滑移与胎面损耗, 延长轮胎工作窗口。",
    },
}

#: 默认短板 (无数据或不足时使用, 顺序固定以保证确定性).
_DEFAULT_WEAKNESSES: tuple[str, ...] = (
    "braking_consistency",
    "corner_exit_speed",
    "sector_2_pace",
    "tire_management",
    "lap_time_consistency",
)

#: 学习路径阶段定义 (6 阶段, 含中文名/聚焦/练习/前置依赖/目标).
_LEARNING_STAGES: list[dict[str, Any]] = [
    {
        "stage": 1,
        "name": "基础制动",
        "focus": "braking",
        "exercises": ["制动点重复训练", "渐进刹车入弯"],
        "prerequisites": [],
        "targets": {"brake_aggression_cv": 0.10},
    },
    {
        "stage": 2,
        "name": "弯道节奏",
        "focus": "cornering",
        "exercises": ["弯心走线训练", "trail-braking 进阶"],
        "prerequisites": [1],
        "targets": {"throttle_smoothness": 0.65},
    },
    {
        "stage": 3,
        "name": "出弯加速",
        "focus": "throttle",
        "exercises": ["渐进油门施加", "出弯牵引力控制"],
        "prerequisites": [1, 2],
        "targets": {"throttle_smoothness": 0.75},
    },
    {
        "stage": 4,
        "name": "圈速一致性",
        "focus": "consistency",
        "exercises": ["连续稳定圈速", "分段节奏固化"],
        "prerequisites": [2, 3],
        "targets": {"lap_time_cv": 0.03},
    },
    {
        "stage": 5,
        "name": "轮胎管理",
        "focus": "tire_management",
        "exercises": ["柔和输入训练", "胎温窗口控制"],
        "prerequisites": [3],
        "targets": {"tire_wear_score": 0.5},
    },
    {
        "stage": 6,
        "name": "综合实战",
        "focus": "racecraft",
        "exercises": ["超车走线", "防守节奏", " stint 管理"],
        "prerequisites": [4, 5],
        "targets": {"lap_time_cv": 0.02, "throttle_smoothness": 0.75},
    },
]


# --- 通用工具 --------------------------------------------------------------
def _clamp01(x: float) -> float:
    """将 ``x`` 钳位到 [0, 1]。"""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _f_opt(d: dict[str, Any], key: str) -> float | None:
    """安全取字段并转 float; 缺失或不可转换返回 None。"""
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cv(values: list[float]) -> float:
    """变异系数 std/|mean|; 样本不足或均值为 0 返回 0.0。"""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.mean(arr))
    if m == 0.0:
        return 0.0
    return float(np.std(arr) / abs(m))


def _collect_floats(lap_metrics: list[dict[str, Any]], key: str) -> list[float]:
    """从单圈指标列表收集指定字段的所有非 None 浮点值 (保持原序)。"""
    out: list[float] = []
    for lm in lap_metrics:
        v = _f_opt(lm, key)
        if v is not None:
            out.append(v)
    return out


def _compute_current_metrics(lap_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """从单圈指标列表聚合当前指标 (均值 + CV + 分段)."""
    result: dict[str, float] = {}
    if not lap_metrics:
        return result

    lap_times = _collect_floats(lap_metrics, "lap_time")
    if lap_times:
        result["lap_time"] = float(np.mean(lap_times))
        result["lap_time_cv"] = _cv(lap_times)

    brake_vals = _collect_floats(lap_metrics, "brake_aggression")
    if brake_vals:
        result["brake_aggression"] = float(np.mean(brake_vals))
        result["brake_aggression_cv"] = _cv(brake_vals)

    thr_vals = _collect_floats(lap_metrics, "throttle_smoothness")
    if thr_vals:
        result["throttle_smoothness"] = float(np.mean(thr_vals))

    wear_vals = _collect_floats(lap_metrics, "tire_wear_score")
    if wear_vals:
        result["tire_wear_score"] = float(np.mean(wear_vals))

    # 分段: 取每圈 sector_times 列表对应位置.
    max_sectors = 0
    for lm in lap_metrics:
        st = lm.get("sector_times")
        if isinstance(st, (list, tuple)):
            max_sectors = max(max_sectors, len(st))
    for si in range(max_sectors):
        vals: list[float] = []
        for lm in lap_metrics:
            st = lm.get("sector_times")
            if isinstance(st, (list, tuple)) and si < len(st) and st[si] is not None:
                try:
                    vals.append(float(st[si]))
                except (TypeError, ValueError):
                    pass
        if vals:
            result[f"sector_{si + 1}_time"] = float(np.mean(vals))
            result[f"sector_{si + 1}_cv"] = _cv(vals)
    return result


def _meets_target(metric: str, current: float, target: float) -> bool:
    """判断当前值是否达标 (依据 _LOWER_IS_BETTER 决定比较方向)."""
    if metric in _LOWER_IS_BETTER:
        return current <= target
    return current >= target


# --- CoachingPlan ----------------------------------------------------------
@dataclass
class CoachingPlan:
    """训练计划容器.

    Attributes
    ----------
    focus_areas
        聚焦的短板领域列表 (如 ``["braking_consistency", "corner_exit_speed"]``).
    exercises
        练习列表, 每项为 ``{name, description, target_metric, target_value,
        duration_laps}``.
    targets
        目标指标 → 目标值, 如 ``{"lap_time_cv": 0.02, "sector_2_time": 28.5}``.
    duration_laps
        计划总训练圈数.
    difficulty
        难度: ``easy`` / ``medium`` / ``hard``.
    """

    focus_areas: list[str]
    exercises: list[dict[str, Any]]
    targets: dict[str, float]
    duration_laps: int
    difficulty: str
    # 保留字段用于未来扩展 (不影响相等性与序列化语义).
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典 (便于 JSON 导出)."""
        return {
            "focus_areas": list(self.focus_areas),
            "exercises": [dict(ex) for ex in self.exercises],
            "targets": dict(self.targets),
            "duration_laps": int(self.duration_laps),
            "difficulty": str(self.difficulty),
            "metadata": dict(self.metadata),
        }


# --- DriverCoach -----------------------------------------------------------
class DriverCoach:
    """个性化车手教练: 评估短板、生成计划、追踪进度并迭代.

    Parameters
    ----------
    profile
        车手画像 (8 维风格向量), 用于结合风格给出训练侧重.
    archetype
        驾驶风格原型名 (如 ``"RACE_CRAFT"`` / ``"DEVELOPMENT"`` /
        ``"AGGRESSIVE"``), 决定默认难度.
    track_id
        赛道标识 (如 ``"melbourne"``), 供未来按赛道调整练习.
    """

    def __init__(
        self,
        profile: DriverProfile,
        archetype: str = "RACE_CRAFT",
        track_id: str = "melbourne",
    ) -> None:
        self.profile = profile
        self.archetype = archetype
        self.track_id = track_id

    # -- 难度 ---------------------------------------------------------------
    def _difficulty_for_archetype(self, archetype: str | None = None) -> str:
        """由原型映射默认难度; 未知原型回退 medium。"""
        arch = archetype if archetype is not None else self.archetype
        return _ARCHETYPE_DIFFICULTY.get(arch, "medium")

    # -- 短板评估 -----------------------------------------------------------
    def assess_weaknesses(self, lap_metrics: list[dict[str, Any]]) -> list[str]:
        """从单圈数据识别至多 3 项短板.

        评估维度: 圈速一致性 (lap_time_cv)、制动一致性 (brake_aggression_cv)、
        出弯平顺度 (throttle_smoothness 偏低)、分段节奏 (最差分段 CV)、
        轮胎管理 (brake_aggression 偏高). 按严重度降序取前 3, 不足时以默认
        短板补齐. 空数据返回固定默认短板 (不崩溃).
        """
        if not lap_metrics:
            return list(_DEFAULT_WEAKNESSES[:3])

        current = _compute_current_metrics(lap_metrics)
        candidates: list[tuple[str, float]] = []

        lap_cv = current.get("lap_time_cv")
        if lap_cv is not None and lap_cv > 0.02:
            candidates.append(("lap_time_consistency", lap_cv))

        brake_cv = current.get("brake_aggression_cv")
        if brake_cv is not None and brake_cv > 0.05:
            candidates.append(("braking_consistency", brake_cv))

        thr_mean = current.get("throttle_smoothness")
        if thr_mean is not None and thr_mean < 0.7:
            candidates.append(("corner_exit_speed", 1.0 - thr_mean))

        brake_mean = current.get("brake_aggression")
        if brake_mean is not None and brake_mean > 0.6:
            candidates.append(("tire_management", brake_mean))

        # 最差分段 (CV 最高), 仅当 CV 超阈值.
        sector_scores: list[tuple[str, float]] = []
        for si in (1, 2, 3):
            scv = current.get(f"sector_{si}_cv")
            if scv is not None and scv > 0.02:
                sector_scores.append((f"sector_{si}_pace", scv))
        if sector_scores:
            worst = max(sector_scores, key=lambda p: p[1])
            candidates.append(worst)

        # 按严重度降序; 同分时保持插入序 (确定性).
        candidates.sort(key=lambda p: p[1], reverse=True)
        result: list[str] = [c[0] for c in candidates[:3]]

        # 不足 3 项时以默认短板补齐 (去重).
        for w in _DEFAULT_WEAKNESSES:
            if len(result) >= 3:
                break
            if w not in result:
                result.append(w)
        return result[:3]

    # -- 目标构建 -----------------------------------------------------------
    def _build_targets(
        self, weaknesses: list[str], lap_metrics: list[dict[str, Any]]
    ) -> dict[str, float]:
        """由短板 + 当前数据构建目标指标字典。"""
        current = _compute_current_metrics(lap_metrics)
        targets: dict[str, float] = {}
        for w in weaknesses:
            spec = _WEAKNESS_TARGETS.get(w)
            if spec is None:
                continue
            metric, default_val = spec
            # 分段时间目标: 在当前最佳 (均值) 基础上提速 2%, 否则用默认值.
            if metric.startswith("sector_") and metric.endswith("_time"):
                cur = current.get(metric)
                if cur is not None and cur > 0.0:
                    targets[metric] = round(cur * 0.98, 3)
                else:
                    targets[metric] = default_val
            else:
                targets[metric] = default_val
        return targets

    # -- 计划生成 -----------------------------------------------------------
    def generate_plan(self, lap_metrics: list[dict[str, Any]]) -> CoachingPlan:
        """生成个性化训练计划 (确定性)."""
        weaknesses = self.assess_weaknesses(lap_metrics)
        difficulty = self._difficulty_for_archetype()
        duration_laps = _DIFFICULTY_LAPS.get(difficulty, 15)
        targets = self._build_targets(weaknesses, lap_metrics)

        exercises: list[dict[str, Any]] = []
        for w in weaknesses:
            lib = _EXERCISE_LIBRARY.get(w)
            spec = _WEAKNESS_TARGETS.get(w)
            if lib is None or spec is None:
                continue
            metric, _default = spec
            target_value = targets.get(metric, _default)
            exercises.append({
                "name": lib["name"],
                "description": lib["description"],
                "target_metric": metric,
                "target_value": float(target_value),
                "duration_laps": int(duration_laps),
            })

        return CoachingPlan(
            focus_areas=list(weaknesses),
            exercises=exercises,
            targets=targets,
            duration_laps=int(duration_laps),
            difficulty=difficulty,
            metadata={"archetype": self.archetype, "track_id": self.track_id},
        )

    # -- 进度追踪 -----------------------------------------------------------
    def track_progress(
        self, lap_metrics: list[dict[str, Any]], plan: CoachingPlan
    ) -> dict[str, Any]:
        """对比当前指标与计划目标, 返回进度结构。"""
        current = _compute_current_metrics(lap_metrics)
        targets = plan.targets
        targets_total = len(targets)
        targets_met = 0
        areas_improved: list[str] = []
        areas_regressed: list[str] = []

        for metric, target in targets.items():
            cur = current.get(metric)
            if cur is None:
                # 缺失指标视为未达标, 计入回退区.
                areas_regressed.append(metric)
                continue
            if _meets_target(metric, cur, target):
                targets_met += 1
                areas_improved.append(metric)
            else:
                areas_regressed.append(metric)

        progress_pct = (targets_met / targets_total * 100.0) if targets_total > 0 else 0.0

        if targets_total == 0:
            recommendation = "当前计划无量化目标, 建议补充目标后再评估进度。"
        elif progress_pct >= 80.0:
            recommendation = "目标达成良好, 建议进入下一阶段训练并适当提升难度。"
        elif progress_pct >= 50.0:
            recommendation = "已达成部分目标, 继续当前训练, 重点巩固未达标项。"
        elif progress_pct >= 20.0:
            recommendation = "进展有限, 建议复盘薄弱环节, 调整训练方法后继续。"
        else:
            recommendation = "暂未达成目标, 建议降低难度回到基础练习, 逐步建立信心。"

        return {
            "targets_met": int(targets_met),
            "targets_total": int(targets_total),
            "progress_pct": float(progress_pct),
            "areas_improved": areas_improved,
            "areas_regressed": areas_regressed,
            "recommendation": recommendation,
        }

    # -- 迭代下一计划 -------------------------------------------------------
    def next_plan(
        self, current_plan: CoachingPlan, lap_metrics: list[dict[str, Any]]
    ) -> CoachingPlan:
        """基于进度生成下一计划.

        目标全部达成 → 难度递进 (上限 hard); 否则保持或下调难度并基于当前
        短板重新生成练习 (允许练习/目标随短板变化而调整).
        """
        progress = self.track_progress(lap_metrics, current_plan)
        total = progress["targets_total"]
        met = progress["targets_met"]

        try:
            idx = _DIFFICULTY_ORDER.index(current_plan.difficulty)
        except ValueError:
            idx = 1

        if total > 0 and met >= total:
            # 全部达标 → 难度递进.
            new_idx = min(idx + 1, len(_DIFFICULTY_ORDER) - 1)
            new_difficulty = _DIFFICULTY_ORDER[new_idx]
        elif total > 0 and met == 0:
            # 完全未达标 → 下调一档 (下限 easy) 以巩固基础.
            new_idx = max(idx - 1, 0)
            new_difficulty = _DIFFICULTY_ORDER[new_idx]
        else:
            # 部分达标 → 维持当前难度.
            new_difficulty = current_plan.difficulty

        # 基于当前短板重新生成练习/目标, 再覆盖难度与时长.
        refreshed = self.generate_plan(lap_metrics)
        duration_laps = _DIFFICULTY_LAPS.get(new_difficulty, refreshed.duration_laps)
        # 同步练习的 duration_laps 字段.
        exercises = [
            {**ex, "duration_laps": int(duration_laps)}
            for ex in refreshed.exercises
        ]
        return CoachingPlan(
            focus_areas=refreshed.focus_areas,
            exercises=exercises,
            targets=refreshed.targets,
            duration_laps=int(duration_laps),
            difficulty=new_difficulty,
            metadata={
                "archetype": self.archetype,
                "track_id": self.track_id,
                "previous_progress_pct": progress["progress_pct"],
                "advanced": new_difficulty != current_plan.difficulty,
            },
        )

    # -- 激励话术 -----------------------------------------------------------
    def motivational_message(self, progress: dict[str, Any]) -> str:
        """根据进度生成中文激励话术 (非空)。"""
        pct = float(progress.get("progress_pct", 0.0))
        if pct >= 80.0:
            return "出色！目标达成度很高, 继续保持当前训练节奏, 向更高难度迈进。"
        if pct >= 50.0:
            return "进步明显！已达成过半目标, 保持专注, 下一个训练周期继续巩固。"
        if pct >= 20.0:
            return "正在进步中, 不要气馁。建议复盘薄弱环节, 调整训练重点后再试。"
        return "暂未达成目标, 这很正常。回到基础练习, 逐步建立信心与节奏, 稳扎稳打。"


# --- LearningPath ----------------------------------------------------------
class LearningPath:
    """结构化学习路径: 阶段化技能进阶 (含前置依赖与达标目标).

    Parameters
    ----------
    archetype
        驾驶风格原型名, 用于在路径摘要中体现个性化侧重.
    """

    def __init__(self, archetype: str) -> None:
        self.archetype = archetype

    def stages(self) -> list[dict[str, Any]]:
        """返回学习阶段列表 (深拷贝以保证调用方不可变内部状态)。"""
        return [dict(s) for s in _LEARNING_STAGES]

    def current_stage(self, completed_stages: list[int]) -> dict[str, Any]:
        """返回下一个未完成阶段; 全部完成时返回末阶段。"""
        done = set(int(s) for s in completed_stages)
        for s in _LEARNING_STAGES:
            if s["stage"] not in done:
                return dict(s)
        return dict(_LEARNING_STAGES[-1])

    def assess_stage_completion(
        self, stage: dict[str, Any], lap_metrics: list[dict[str, Any]]
    ) -> bool:
        """检查阶段目标是否全部达成 (无目标或无数据返回 False)。"""
        targets = stage.get("targets") or {}
        if not targets:
            return False
        current = _compute_current_metrics(lap_metrics)
        if not current:
            return False
        for metric, target in targets.items():
            cur = current.get(metric)
            if cur is None:
                return False
            if not _meets_target(metric, cur, float(target)):
                return False
        return True

    def path_summary(self) -> str:
        """返回完整学习路径的中文描述 (非空)。"""
        parts: list[str] = [f"针对「{self.archetype}」原型, 学习路径共 "
                            f"{len(_LEARNING_STAGES)} 个阶段:"]
        for s in _LEARNING_STAGES:
            prereq = "无" if not s["prerequisites"] else "需先完成第 " + ",".join(
                str(p) for p in s["prerequisites"]
            ) + " 阶段"
            parts.append(
                f"第{s['stage']}阶段「{s['name']}」: 聚焦 {s['focus']}, "
                f"练习 {len(s['exercises'])} 项 ({prereq})."
            )
        parts.append("建议按阶段顺序逐步推进, 每阶段达标后再进入下一阶段。")
        return "".join(parts)


# --- SkillAssessment -------------------------------------------------------
class SkillAssessment:
    """车手技能评估: 5 维技能分解 + 等级判定 + 原型对比."""

    def __init__(self) -> None:
        pass

    # -- 单维技能计算 -------------------------------------------------------
    @staticmethod
    def _skill_braking(
        lap_metrics: list[dict[str, Any]], profile: DriverProfile
    ) -> float:
        brake_vals = _collect_floats(lap_metrics, "brake_aggression")
        if brake_vals:
            cv = _cv(brake_vals)
            return _clamp01(1.0 - cv / (cv + 0.1))
        # 回退: 进攻性接近 0.5 (适度) 视为制动控制较好.
        return _clamp01(1.0 - abs(profile.aggression_score - 0.5))

    @staticmethod
    def _skill_cornering(
        lap_metrics: list[dict[str, Any]], profile: DriverProfile
    ) -> float:
        thr_vals = _collect_floats(lap_metrics, "throttle_smoothness")
        if thr_vals:
            return _clamp01(float(np.mean(thr_vals)))
        return _clamp01((profile.throttle_smoothness + profile.steer_smoothness) / 2.0)

    @staticmethod
    def _skill_consistency(
        lap_metrics: list[dict[str, Any]], profile: DriverProfile
    ) -> float:
        lap_times = _collect_floats(lap_metrics, "lap_time")
        if lap_times:
            cv = _cv(lap_times)
            return _clamp01(1.0 - cv / (cv + 0.02))
        return _clamp01(profile.consistency_score)

    @staticmethod
    def _skill_racecraft(profile: DriverProfile) -> float:
        # 进攻性接近 0.6 (适度进攻) 为佳; DRS 效率贡献次之.
        aggr = profile.aggression_score
        aggr_score = _clamp01(1.0 - abs(aggr - 0.6) / 0.6)
        drs = profile.drs_usage_efficiency
        return _clamp01(0.6 * aggr_score + 0.4 * drs)

    @staticmethod
    def _skill_tire_mgmt(
        lap_metrics: list[dict[str, Any]], profile: DriverProfile
    ) -> float:
        aggr = profile.aggression_score
        smooth = (profile.throttle_smoothness + profile.steer_smoothness) / 2.0
        base = _clamp01(0.6 * (1.0 - aggr) + 0.4 * smooth)
        wear_vals = _collect_floats(lap_metrics, "tire_wear_score")
        if wear_vals:
            wear = float(np.mean(wear_vals))
            # wear 可能是百分比 (>1) 或 [0,1] 比率; 统一归一化.
            wear_norm = wear / 100.0 if wear > 1.0 else wear
            wear_score = _clamp01(1.0 - wear_norm)
            base = _clamp01(0.5 * base + 0.5 * wear_score)
        return base

    @staticmethod
    def _skill_level(overall: float) -> str:
        """overall_skill → 等级标签 (基于 LEVEL_THRESHOLDS)。"""
        if overall < LEVEL_THRESHOLDS["ROOKIE"]:
            return "ROOKIE"
        if overall < LEVEL_THRESHOLDS["AMATEUR"]:
            return "AMATEUR"
        if overall < LEVEL_THRESHOLDS["INTERMEDIATE"]:
            return "INTERMEDIATE"
        if overall < LEVEL_THRESHOLDS["ADVANCED"]:
            return "ADVANCED"
        return "EXPERT"

    # -- 综合评估 -----------------------------------------------------------
    def assess(
        self, lap_metrics: list[dict[str, Any]], profile: DriverProfile
    ) -> dict[str, Any]:
        """评估车手技能, 返回综合分/分解/等级/中文优劣势。"""
        breakdown = {
            "braking": self._skill_braking(lap_metrics, profile),
            "cornering": self._skill_cornering(lap_metrics, profile),
            "consistency": self._skill_consistency(lap_metrics, profile),
            "racecraft": self._skill_racecraft(profile),
            "tire_mgmt": self._skill_tire_mgmt(lap_metrics, profile),
        }
        overall = _clamp01(float(np.mean(list(breakdown.values()))))
        level = self._skill_level(overall)

        strengths: list[str] = []
        weaknesses: list[str] = []
        for area, score in breakdown.items():
            label = _SKILL_LABELS_ZH[area]
            if score >= 0.7:
                strengths.append(f"{label}出色 ({score:.2f})")
            elif score < 0.4:
                weaknesses.append(f"{label}需加强 ({score:.2f})")
        if not strengths:
            strengths.append("风格均衡, 各项基础扎实")
        if not weaknesses:
            weaknesses.append("暂无明显短板, 可挑战更高难度")

        return {
            "overall_skill": overall,
            "skill_breakdown": breakdown,
            "skill_level": level,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }

    # -- 原型对比 -----------------------------------------------------------
    def compare_to_archetype(self, skill: dict[str, Any], archetype: str) -> dict[str, Any]:
        """将技能分解与原型期望对比, 返回差距与评估。"""
        expectations = _ARCHETYPE_EXPECTATIONS.get(
            archetype, _ARCHETYPE_EXPECTATIONS["RACE_CRAFT"]
        )
        breakdown = skill.get("skill_breakdown", {}) or {}
        gaps: dict[str, float] = {}
        for area, exp in expectations.items():
            actual = float(breakdown.get(area, 0.0))
            gaps[area] = round(actual - exp, 4)
        met = [a for a, g in gaps.items() if g >= 0.0]
        below = [a for a, g in gaps.items() if g < 0.0]
        overall_gap = float(np.mean(list(gaps.values()))) if gaps else 0.0
        if len(met) >= len(expectations) * 0.6:
            assessment = "符合或超过原型预期"
        elif len(met) >= len(expectations) * 0.3:
            assessment = "部分达到原型预期"
        else:
            assessment = "低于原型预期, 需加强训练"
        return {
            "archetype": archetype,
            "expectations": dict(expectations),
            "gaps": gaps,
            "areas_meeting_expectation": met,
            "areas_below_expectation": below,
            "overall_gap": overall_gap,
            "assessment": assessment,
        }
