"""Command-line interface for the F1 setup optimizer.

Provides subcommands for training, prediction, setup search, validation,
serving the API, and inspecting tracks/setups. Uses argparse (no external
CLI framework). Each subcommand parses args, calls the relevant function,
and prints results as formatted output (table or JSON with ``--json`` flag).
Exits 0 on success, 1 on error (error message to stderr).

Iter-177: add ``--style`` preset option to ``feedback`` subcommand for
driver driving-style presets (aggressive, conservative, balanced, smooth,
late_braker). Each preset maps to a :class:`~f1opt.driver.profile.DriverProfile`
via :func:`_resolve_style_profile`.

Public API:
    - :func:`build_parser` — construct the argparse parser.
    - :func:`main` — entry point, returns process exit code.
    - :func:`format_output` — format data as table or JSON.
    - ``cmd_*`` functions — one handler per subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from f1opt.driver.profile import DriverProfile

from f1opt.data.setup_schema import DEFAULT_SETUP, SETUP_FIELDS, CarSetup
from f1opt.data.tracks import ALL_TRACKS, TRACKS_BY_ID

__all__ = ["build_parser", "main", "format_output"]


# --------------------------------------------------------------------------- #
# Parser construction
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the :class:`argparse.ArgumentParser` for the f1opt CLI."""
    parser = argparse.ArgumentParser(
        prog="f1opt",
        description="F1 2026 setup optimizer CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- train -------------------------------------------------------------
    p_train = subparsers.add_parser("train", help="train the surrogate model")
    p_train.add_argument("--iterations", type=int, default=500)
    p_train.add_argument("--save", dest="save", action="store_true", default=True,
                         help="保存模型权重并刷新缓存 (默认)")
    p_train.add_argument("--no-save", dest="save", action="store_false",
                         help="不保存 (测试隔离)")
    p_train.add_argument("--log", action="store_true")
    p_train.add_argument("--json", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # --- predict -----------------------------------------------------------
    p_predict = subparsers.add_parser("predict", help="predict lap time for a setup")
    p_predict.add_argument("--track", required=True)
    p_predict.add_argument("--setup-json", required=True)
    p_predict.add_argument("--json", action="store_true")
    p_predict.set_defaults(func=cmd_predict)

    # --- search ------------------------------------------------------------
    p_search = subparsers.add_parser("search", help="search for optimal setup")
    p_search.add_argument("--track", required=True)
    p_search.add_argument("--tire-wear-weight", type=float, default=0.0)
    p_search.add_argument(
        "--method",
        choices=["differential", "bayesian"],
        default="differential",
    )
    p_search.add_argument("--iterations", type=int, default=100)
    p_search.add_argument("--seed", type=int, default=42,
                         help="随机种子 (固定可复现)")
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    # --- bayesian ----------------------------------------------------------
    p_bayes = subparsers.add_parser("bayesian", help="Bayesian search")
    p_bayes.add_argument("--track", required=True)
    p_bayes.add_argument("--iterations", type=int, default=15)
    p_bayes.add_argument("--seed", type=int, default=42,
                         help="随机种子 (固定可复现)")
    p_bayes.add_argument(
        "--acquisition",
        choices=["ei", "ucb", "pi"],
        default="ei",
    )
    p_bayes.add_argument("--json", action="store_true")
    p_bayes.set_defaults(func=cmd_bayesian)

    # --- validate ----------------------------------------------------------
    p_validate = subparsers.add_parser("validate", help="run model validation")
    p_validate.add_argument("--track", required=True)
    p_validate.add_argument("--json", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    # --- serve -------------------------------------------------------------
    p_serve = subparsers.add_parser("serve", help="start the API server (uvicorn)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--extended", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    # --- tracks (subcommand group) -----------------------------------------
    p_tracks = subparsers.add_parser("tracks", help="track information")
    tracks_sub = p_tracks.add_subparsers(dest="tracks_command")
    p_tracks_list = tracks_sub.add_parser("list", help="list all 24 tracks")
    p_tracks_list.add_argument("--json", action="store_true")
    p_tracks_list.set_defaults(func=cmd_tracks_list)
    p_tracks_info = tracks_sub.add_parser("info", help="show track details")
    p_tracks_info.add_argument("--track", required=True)
    p_tracks_info.add_argument("--json", action="store_true")
    p_tracks_info.set_defaults(func=cmd_tracks_info)

    # --- teams (subcommand group) ------------------------------------------
    p_teams = subparsers.add_parser("teams", help="team & driver information")
    teams_sub = p_teams.add_subparsers(dest="teams_command")
    p_teams_list = teams_sub.add_parser("list", help="list all 11 teams / 22 drivers")
    p_teams_list.add_argument("--json", action="store_true")
    p_teams_list.set_defaults(func=cmd_teams_list)

    # --- setup (subcommand group) ------------------------------------------
    p_setup = subparsers.add_parser("setup", help="setup information")
    setup_sub = p_setup.add_subparsers(dest="setup_command")
    p_setup_default = setup_sub.add_parser("default", help="show default setup")
    p_setup_default.add_argument("--json", action="store_true")
    p_setup_default.set_defaults(func=cmd_setup_default)
    p_setup_validate = setup_sub.add_parser("validate", help="validate a setup")
    p_setup_validate.add_argument("--setup-json", required=True)
    p_setup_validate.add_argument("--json", action="store_true")
    p_setup_validate.set_defaults(func=cmd_setup_validate)

    # --- feedback (Iter-177: + --style preset) -----------------------------
    p_feedback = subparsers.add_parser(
        "feedback",
        help="generate driver feedback from telemetry frames",
        description=(
            "Generate rule-based / LLM-enhanced feedback for a driver.\n\n"
            "Driver feedback with three precision levels:\n"
            "  [corner]  精确到某个弯道 (T1、T130R、发卡弯):\n"
            "    f1opt feedback --track suzuka --question '为什么 T1 入弯总推头?'\n"
            "  [sector]  某一段/扇区 (S2、直道段、连续弯段):\n"
            "    f1opt feedback --track suzuka --question 'S2 连续弯那一段车头太钝'\n"
            "  [overall] 整体感受 (全圈、整车平衡、总体策略):\n"
            "    f1opt feedback --track bahrain --question '圈速能再快多少?'\n"
            "\n"
            "Driving style presets (--style):\n"
            "  aggressive  — 高激进度, 晚刹车, 强力ERS部署\n"
            "  conservative — 平稳驾驶, 保胎优先, 保守ERS\n"
            "  balanced    — 中性平衡, 综合考量\n"
            "  smooth      — 高平顺性, 最小化输入扰动\n"
            "  late_braker — 超晚刹车点, 超高入弯激进\n"
            "\n"
            "Use --list-examples to see all supported question styles grouped by granularity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_feedback.add_argument("--track", required=False,
                            help="track id (e.g. suzuka, monza, monaco)")
    p_feedback.add_argument("--question", default=None,
                            help="driver's follow-up question (see examples above)")
    p_feedback.add_argument("--session-id", default=None,
                            help="conversation session id for multi-turn memory")
    p_feedback.add_argument("--driver-style",
                            choices=["default", "aggressive", "conservative"],
                            default="default")
    p_feedback.add_argument("--style",
                            choices=["aggressive", "conservative", "balanced", "smooth", "late_braker"],
                            default=None,
                            help="preset driving style profile (overrides --driver-style)")
    p_feedback.add_argument("--frames-json", default=None,
                            help="JSON file with telemetry frames (default: synthetic)")
    p_feedback.add_argument("--list-examples", action="store_true",
                            help="print driver feedback question examples and exit")
    p_feedback.add_argument("--json", action="store_true")
    p_feedback.set_defaults(func=cmd_feedback)

    # --- template (Iter-183: 车手反馈模板) ---------------------------------
    p_template = subparsers.add_parser(
        "template",
        help="generate driver feedback template",
        description=(
            "Generate structured feedback templates for drivers to fill in "
            "after a session. Templates are grouped by granularity:\n"
            "  corner  — corner-specific feedback (understeer, oversteer, braking, traction)\n"
            "  sector  — sector-level feedback (balance, tyres)\n"
            "  overall — overall lap feedback (general, setup, ERS, comparison)\n"
            "  all     — all 10 templates\n"
            "\n"
            "Examples:\n"
            "  f1opt template --group overall --lang zh\n"
            "  f1opt template --id corner_understeer --lang en\n"
            "  f1opt template --group all --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_template.add_argument("--group", choices=["corner", "sector", "overall", "all"],
                            default="all", help="template group")
    p_template.add_argument("--id", default=None,
                            help="specific template id (overrides --group)")
    p_template.add_argument("--lang", choices=["zh", "en"], default="zh",
                            help="language (default: zh)")
    p_template.add_argument("--json", action="store_true")
    p_template.set_defaults(func=cmd_template)

    return parser


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Fallback serializer for objects not natively JSON-serializable."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def format_output(data: Any, json_mode: bool) -> str:
    """Format ``data`` as a JSON string (``json_mode=True``) or a table.

    - JSON mode: ``json.dumps`` with indentation; CarSetup/pydantic models are
      serialized via :func:`_json_default`.
    - Table mode: dicts render as ``key: value`` lines; lists of dicts render
      as a tab-separated table with a header row; other lists render one item
      per line; scalars render via ``str``.
    """
    if json_mode:
        return json.dumps(data, indent=2, default=_json_default, sort_keys=True)
    if isinstance(data, dict):
        lines = [f"{k}: {v}" for k, v in data.items()]
        return "\n".join(lines) if lines else "(empty)"
    if isinstance(data, list):
        if not data:
            return "(empty)"
        if isinstance(data[0], dict):
            headers = list(data[0].keys())
            rows = ["\t".join(str(h) for h in headers)]
            for row in data:
                rows.append("\t".join(str(row.get(h, "")) for h in headers))
            return "\n".join(rows)
        return "\n".join(str(item) for item in data)
    return str(data)


def _print(data: Any, json_mode: bool) -> None:
    """Format ``data`` and print to stdout."""
    print(format_output(data, json_mode))


def _err(msg: str) -> None:
    """Print an error message to stderr."""
    # Windows: 确保控制台能正确输出中文错误信息
    if sys.platform == "win32":
        try:
            print(f"错误: {msg}", file=sys.stderr, flush=True)
        except UnicodeEncodeError:
            print(f"error: {msg}", file=sys.stderr)
    else:
        print(f"error: {msg}", file=sys.stderr)


def _parse_setup_json(setup_json: str) -> CarSetup:
    """Parse a setup JSON string into a :class:`CarSetup`.

    Missing fields fall back to :data:`DEFAULT_SETUP` (partial setups allowed).
    Raises :class:`ValueError` on invalid JSON or schema validation failure.
    """
    try:
        setup_dict = json.loads(setup_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid setup JSON: {exc}") from exc
    if not isinstance(setup_dict, dict):
        raise ValueError("setup JSON must be a JSON object")
    # 部分 setup 允许：缺失字段回退 DEFAULT_SETUP (否则 21 字段全必填)。
    merged = {**DEFAULT_SETUP.model_dump(), **setup_dict}
    return CarSetup(**merged)


def _resolve_style_profile(style: str) -> DriverProfile:
    """Iter-177: map a --style preset name to a DriverProfile.

    Supported presets:
        aggressive   — AGGRESSIVE_PROFILE (高激进, 晚刹, 强ERS)
        conservative — CONSERVATIVE_PROFILE (平稳, 保胎, 低激进)
        balanced     — neutral (all ~0.5)
        smooth       — high smoothness (~0.90), low aggression
        late_braker  — ultra-late braking (brake_point_norm=0.10), max aggression
    """
    from f1opt.driver.profile import (
        AGGRESSIVE_PROFILE,
        CONSERVATIVE_PROFILE,
        DriverProfile,
    )
    if style == "aggressive":
        return AGGRESSIVE_PROFILE
    if style == "conservative":
        return CONSERVATIVE_PROFILE
    if style == "balanced":
        return DriverProfile(
            brake_point_norm=0.50, throttle_smoothness=0.50,
            steer_smoothness=0.50, corner_balance_pref=0.50,
            aggression_score=0.50, consistency_score=0.50,
            ers_usage_intensity=0.50, drs_usage_efficiency=0.50,
        )
    if style == "smooth":
        return DriverProfile(
            brake_point_norm=0.60, throttle_smoothness=0.90,
            steer_smoothness=0.90, corner_balance_pref=0.55,
            aggression_score=0.25, consistency_score=0.80,
            ers_usage_intensity=0.40, drs_usage_efficiency=0.60,
        )
    if style == "late_braker":
        return DriverProfile(
            brake_point_norm=0.10, throttle_smoothness=0.35,
            steer_smoothness=0.40, corner_balance_pref=0.30,
            aggression_score=0.95, consistency_score=0.30,
            ers_usage_intensity=0.85, drs_usage_efficiency=0.70,
        )
    return AGGRESSIVE_PROFILE


def _known_track_id(track_id: str) -> bool:
    """Return True if ``track_id`` resolves to a known F1 2026 track.

    Aliases are resolved via :func:`canonical_track_id` (e.g. ``bahrain`` →
    ``sakhir``) so both the circuit name and the city-name key are accepted.
    Unknown ids return False so CLI/API callers can reject them gracefully
    instead of silently falling back to a generic prior lap time.
    """
    from f1opt.data.ea_f1_2026_benchmark import canonical_track_id

    return TRACKS_BY_ID.get(canonical_track_id(track_id)) is not None


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Train the surrogate model."""
    from f1opt.model.train import train

    try:
        model = train(iterations=args.iterations, log=args.log, save=args.save)
        summary = {
            "status": "trained",
            "iterations": args.iterations,
            "model_version": model.model_version,
            "saved": args.save,
        }
        _print(summary, args.json)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict lap time for a setup."""
    if not _known_track_id(args.track):
        _err(f"unknown track: {args.track}")
        return 1
    try:
        setup = _parse_setup_json(args.setup_json)
    except ValueError as exc:
        _err(str(exc))
        return 1
    try:
        from f1opt.model.surrogate import predict_lap_time

        lap_time = float(predict_lap_time(setup, args.track))
        result = {
            "track": args.track,
            "lap_time": lap_time,
            "setup": setup.model_dump(),
        }
        _print(result, args.json)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for optimal setup (differential or bayesian method)."""
    if not _known_track_id(args.track):
        _err(f"unknown track: {args.track}")
        return 1
    try:
        if args.method == "bayesian":
            from f1opt.model.bayesian import bayesian_search_setup

            bayesian_result = bayesian_search_setup(
                args.track,
                DEFAULT_SETUP,
                n_iterations=args.iterations,
                acquisition="ei",
                seed=args.seed,
            )
            output = {
                "method": "bayesian",
                "track": args.track,
                "recommended_setup": bayesian_result["recommended_setup"].model_dump(),
                "recommended_lap_time": bayesian_result["recommended_lap_time"],
                "baseline_lap_time": bayesian_result["baseline_lap_time"],
                "predicted_gain_s": bayesian_result["predicted_gain_s"],
                "iterations": bayesian_result["iterations"],
            }
        else:
            from f1opt.model.optimizer import search_setup

            search_result = search_setup(
                args.track,
                iterations=args.iterations,
                tire_wear_weight=args.tire_wear_weight,
                seed=args.seed,
            )
            output = {
                "method": "differential",
                "track": args.track,
                "recommended": search_result.recommended,
                "baseline": search_result.baseline,
                "predicted_gain_s": search_result.predicted_gain_s,
                "baseline_lap_time": search_result.baseline_lap_time,
                "recommended_lap_time": search_result.recommended_lap_time,
                "algorithm": search_result.algorithm,
                "iterations": search_result.iterations,
            }
        _print(output, args.json)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_bayesian(args: argparse.Namespace) -> int:
    """Bayesian search."""
    if not _known_track_id(args.track):
        _err(f"unknown track: {args.track}")
        return 1
    try:
        from f1opt.model.bayesian import bayesian_search_setup

        result = bayesian_search_setup(
            args.track,
            DEFAULT_SETUP,
            n_iterations=args.iterations,
            acquisition=args.acquisition,
            seed=args.seed,
        )
        output = {
            "track": args.track,
            "recommended_setup": result["recommended_setup"].model_dump(),
            "recommended_lap_time": result["recommended_lap_time"],
            "baseline_lap_time": result["baseline_lap_time"],
            "predicted_gain_s": result["predicted_gain_s"],
            "iterations": result["iterations"],
            "acquisition": result["acquisition"],
            "gp_final_std": result["gp_final_std"],
        }
        _print(output, args.json)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Run model validation."""
    if not _known_track_id(args.track):
        _err(f"unknown track: {args.track}")
        return 1
    try:
        from f1opt.model.validation import SurrogateValidator

        validator = SurrogateValidator()
        report = validator.full_report()
        output = {
            "track": args.track,
            "passed": report["passed"],
            "summary": report["summary"],
            "checks": report["checks"],
        }
        _print(output, args.json)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API server (uvicorn).

    Iter-252: 默认即完整 App (核心 + 扩展路由), 保证两个 UI 视图 (实时面板
    ``/`` 与智能分析中心 ``/dashboard.html``) 点开即用 —— 智能分析中心调用的
    ``/api/bayesian-search`` / ``/api/pareto-search`` / ``/api/compare/*`` /
    ``/api/weather/impact`` / ``/api/health/extended`` 均在扩展路由内, 仅
    ``--extended`` 才挂载导致默认 serve 下这些 tab 全部 404。
    ``--extended`` 保留为兼容别名 (行为一致)。
    """
    try:
        import uvicorn

        from f1opt.api.extended_app import create_extended_app

        app = create_extended_app(start_listener=True)
        # Windows: 使用 windows_events 兼容的事件循环
        if sys.platform == "win32":
            uvicorn.run(app, host=args.host, port=args.port, loop="asyncio")
        else:
            uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_tracks_list(args: argparse.Namespace) -> int:
    """List all 24 tracks."""
    tracks = [
        {
            "track_id": t.track_id,
            "official_name": t.official_name,
            "circuit_name": t.circuit_name,
            "country": t.country,
            "round_number": t.round_number,
            "is_sprint": t.is_sprint,
            "length_m": t.length_m,
            "corners": t.corners,
            "track_type": t.track_type,
        }
        for t in ALL_TRACKS
    ]
    _print(tracks, args.json)
    return 0


def cmd_teams_list(args: argparse.Namespace) -> int:
    """List all 11 teams and 22 drivers (F1 2026 grid)."""
    from f1opt.data.drivers_2026 import all_drivers_2026
    from f1opt.data.teams_2026 import all_teams_2026_profiles

    teams = all_teams_2026_profiles()
    drivers = all_drivers_2026()
    result = {
        "n_teams": len(teams),
        "n_drivers": len(drivers),
        "teams": [
            {
                "team_id": t.team_id,
                "team_name": t.team_name,
                "full_name": t.full_name,
                "power_unit_supplier": t.power_unit_supplier,
                "pace_offset_s": t.pace_offset_s,
                "drivers": [
                    {"driver_id": d.driver_id, "driver_name": d.driver_name}
                    for d in drivers
                    if d.team_id == t.team_id
                ],
            }
            for t in teams
        ],
    }
    _print(result, args.json)
    return 0


def cmd_tracks_info(args: argparse.Namespace) -> int:
    """Show track details."""
    track = TRACKS_BY_ID.get(args.track)
    if track is None:
        _err(f"unknown track: {args.track}")
        return 1
    info = {
        "track_id": track.track_id,
        "official_name": track.official_name,
        "circuit_name": track.circuit_name,
        "city": track.city,
        "country": track.country,
        "country_code": track.country_code,
        "round_number": track.round_number,
        "date_range": track.date_range,
        "is_sprint": track.is_sprint,
        "length_m": track.length_m,
        "corners": track.corners,
        "elevation_change_m": track.elevation_change_m,
        "track_type": track.track_type,
        "notes": track.notes,
    }
    # Iter-274: 附带该场 Pirelli 2026 选胎方案 (soft/medium/hard -> C0-C5).
    try:
        from f1opt.model.pirelli_2026 import tire_compound_for_track

        sel = tire_compound_for_track(track.track_id)
        info["pirelli_compounds"] = {
            "soft": sel.soft_code, "medium": sel.medium_code, "hard": sel.hard_code,
        }
    except Exception:
        info["pirelli_compounds"] = None
    _print(info, args.json)
    return 0


def cmd_setup_default(args: argparse.Namespace) -> int:
    """Show default setup."""
    _print(DEFAULT_SETUP.model_dump(), args.json)
    return 0


def cmd_setup_validate(args: argparse.Namespace) -> int:
    """Validate a setup against the schema."""
    try:
        setup = _parse_setup_json(args.setup_json)
    except ValueError as exc:
        _err(str(exc))
        return 1
    output = {
        "valid": True,
        "setup": setup.model_dump(),
        "n_fields": len(SETUP_FIELDS),
    }
    _print(output, args.json)
    return 0


def _feedback_examples_text() -> str:
    """Render feedback examples for CLI display."""
    from f1opt.feedback.prompts import FEEDBACK_EXAMPLES
    lines = [f"Driver feedback examples ({len(FEEDBACK_EXAMPLES)} total):", ""]
    grouped: dict[str, list[dict[str, Any]]] = {"corner": [], "sector": [], "overall": []}
    for ex in FEEDBACK_EXAMPLES:
        g = ex.get("granularity", "overall")
        grouped.setdefault(g, []).append(ex)
    for g in ["corner", "sector", "overall"]:
        items = grouped.get(g, [])
        if not items:
            continue
        labels = {"corner": "[corner] 精确到弯道", "sector": "[sector] 某一段/扇区", "overall": "[overall] 整体感受"}
        lines.append(f"  {labels.get(g, g)}:")
        for ex in items:
            lines.append(f"    Q: {ex['question']}")
            lines.append(f"    A: {ex['example_answer'][:120]}...")
            lines.append("")
    return "\n".join(lines)


def cmd_feedback(args: argparse.Namespace) -> int:
    """Generate driver feedback (rule-based or LLM-enhanced).

    Iter-177: --style preset overrides --driver-style.
    """
    from f1opt.driver.profile import (
        AGGRESSIVE_PROFILE,
        CONSERVATIVE_PROFILE,
    )
    from f1opt.feedback.prompts import FEEDBACK_EXAMPLES

    # --list-examples: print grouped examples and exit
    if args.list_examples:
        print(_feedback_examples_text())
        return 0

    # --json mode list-examples
    if args.list_examples and args.json:
        grouped: dict[str, list[dict[str, Any]]] = {"corner": [], "sector": [], "overall": []}
        for ex in FEEDBACK_EXAMPLES:
            g = ex.get("granularity", "overall")
            grouped.setdefault(g, []).append(ex)
        _print({"n_examples": len(FEEDBACK_EXAMPLES), "examples_by_granularity": grouped}, True)
        return 0

    # Resolve driver profile: --style > --driver-style > None
    dp: DriverProfile | None = None
    if args.style:
        dp = _resolve_style_profile(args.style)
    elif args.driver_style == "aggressive":
        dp = AGGRESSIVE_PROFILE
    elif args.driver_style == "conservative":
        dp = CONSERVATIVE_PROFILE

    # Load frames if provided
    frames: list[dict] = []
    if args.frames_json:
        import json as _json
        with open(args.frames_json, encoding="utf-8") as f:
            frames = _json.load(f)

    try:
        from f1opt.data.setup_schema import DEFAULT_SETUP
        from f1opt.feedback.engine import generate_feedback

        result = generate_feedback(
            frames=frames if frames else [],
            setup=DEFAULT_SETUP.model_dump(),
            track_id=args.track or "",
            question=args.question,
            driver_profile=dp,
            session_id=args.session_id,
        )
        _print(result, args.json)
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def cmd_template(args: argparse.Namespace) -> int:
    """Generate driver feedback template (Iter-183)."""
    from f1opt.feedback.prompts import (
        DRIVER_FEEDBACK_TEMPLATES,
        FEEDBACK_TEMPLATE_GROUPS,
        render_feedback_template,
    )

    try:
        if args.id:
            # 单个模板
            template_ids = [args.id]
        else:
            template_ids = FEEDBACK_TEMPLATE_GROUPS.get(args.group, FEEDBACK_TEMPLATE_GROUPS["all"])

        if args.json:
            # JSON 格式输出所有模板
            templates = {}
            for tid in template_ids:
                t = DRIVER_FEEDBACK_TEMPLATES.get(tid)
                if t is None:
                    _err(f"unknown template: {tid}")
                    return 1
                templates[tid] = {
                    "id": t["id"],
                    "granularity": t["granularity"],
                    "category": t["category"],
                    "text": render_feedback_template(tid, args.lang),
                }
            _print(templates, True)
        else:
            # 文本格式输出
            for tid in template_ids:
                t = DRIVER_FEEDBACK_TEMPLATES.get(tid)
                if t is None:
                    _err(f"unknown template: {tid}")
                    return 1
                print(f"--- {t['id']} [{t['granularity']}/{t['category']}] ---")
                print(render_feedback_template(tid, args.lang))
                print()
        return 0
    except Exception as exc:
        _err(str(exc))
        return 1


def _launch_gui(host: str = "127.0.0.1", port: int = 8000) -> int:
    """Double-click entry: start the API server and open the browser.

    When ``f1opt.exe`` is launched with no arguments (double-click), we start
    the API server in the main thread and open the UI in the default browser.
    The server stays alive until the user closes the console window or presses
    Ctrl+C.  On any startup error the window stays open so the user can read
    the message.
    """
    import threading
    import webbrowser

    from f1opt.api.extended_app import create_extended_app

    url = f"http://{host}:{port}/"
    print("=" * 72)
    print("  F1 2026 Setup Optimizer — 正在启动 ...")
    print(f"  浏览器打开后访问: {url}")
    print("  按 Ctrl+C 或关闭本窗口即可退出。")
    print("=" * 72)
    sys.stdout.flush()

    # 在后台线程打开浏览器 (等服务器就绪后)
    def _open_browser() -> None:
        import socket
        import time

        for _ in range(200):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.2)
                    if sock.connect_ex((host, port)) == 0:
                        break
            except OSError:
                pass
            time.sleep(0.1)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open_browser, name="f1opt-browser", daemon=True).start()

    app = create_extended_app(start_listener=True)
    try:
        import uvicorn

        if sys.platform == "win32":
            uvicorn.run(app, host=host, port=port, loop="asyncio")
        else:
            uvicorn.run(app, host=host, port=port)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        _err(f"服务器启动失败: {exc}")
        try:
            input("按回车键退出...")
        except (EOFError, OSError):
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, dispatch to handler, print result."""
    # Windows: multiprocessing freeze_support 必须在任何 multiprocessing 使用前调用
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()
        # Windows: ProactorEventLoop 自 Python 3.8 起即为默认事件循环，
        # 无需显式设置；显式设置 WindowsProactorEventLoopPolicy 在 3.14+ 已弃用。
        # Windows: 注册信号处理 (SIGINT=SIGBREAK, SIGTERM 不可用)
        import signal as _signal
        def _win_signal_handler(signum, frame):
            print("\n收到中断信号，正在退出...", file=sys.stderr)
            sys.exit(0)
        _signal.signal(_signal.SIGINT, _win_signal_handler)
        if hasattr(_signal, "SIGBREAK"):
            _signal.signal(_signal.SIGBREAK, _win_signal_handler)
        # Windows: 启用控制台 ANSI 转义序列支持 (颜色输出)
        try:
            import ctypes as _ctypes
            _kernel32 = _ctypes.windll.kernel32
            _kernel32.SetConsoleMode(_kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
        # Windows: 强制 UTF-8 编码 (Python 3.7+ PEP 540)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # PyInstaller: 将 _MEIPASS 临时目录加入 PATH 以便找到 DLL
    if getattr(sys, "frozen", False):
        _meipass = getattr(sys, "_MEIPASS", None)
        if _meipass:
            if _meipass not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _meipass + os.pathsep + os.environ.get("PATH", "")
            # Windows: os.add_dll_directory 确保 PyInstaller 打包的 DLL 可被加载
            if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_meipass)
    parser = build_parser()
    # 无参数启动 (如双击 f1opt.exe): 直接打开智能分析中心, 而不是打印帮助后
    # 退出 — 否则控制台窗口一闪而过, 表现为"打不开"。显式传入空列表 (如测试
    # 中的 ``main([])``) 仍保留打印帮助的旧行为。
    if argv is None:
        argv = sys.argv[1:]
        if not argv:
            return _launch_gui()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
