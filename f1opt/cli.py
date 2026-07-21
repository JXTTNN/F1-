"""Command-line interface for the F1 setup optimizer.

Provides subcommands for training, prediction, setup search, validation,
serving the API, and inspecting tracks/setups. Uses argparse (no external
CLI framework). Each subcommand parses args, calls the relevant function,
and prints results as formatted output (table or JSON with ``--json`` flag).
Exits 0 on success, 1 on error (error message to stderr).

Public API:
    - :func:`build_parser` — construct the argparse parser.
    - :func:`main` — entry point, returns process exit code.
    - :func:`format_output` — format data as table or JSON.
    - ``cmd_*`` functions — one handler per subcommand.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

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
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    # --- bayesian ----------------------------------------------------------
    p_bayes = subparsers.add_parser("bayesian", help="Bayesian search")
    p_bayes.add_argument("--track", required=True)
    p_bayes.add_argument("--iterations", type=int, default=15)
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

    # --- feedback ----------------------------------------------------------
    p_feedback = subparsers.add_parser(
        "feedback",
        help="generate driver feedback from telemetry frames",
        description=(
            "Generate rule-based / LLM-enhanced feedback for a driver.\n\n"
            "Driver feedback with three precision levels:\n"
            "  [corner]  精确到某个弯道 (T1、T130R、发卡弯):\n"
            "    f1opt feedback --track suzuka --question '为什么 T1 入弯总推头?'\n"
            "    f1opt feedback --track silverstone "
            "--question 'How should I adjust the diff for T8?'\n"
            "    f1opt feedback --track suzuka --question 'T3 弯心的时候后轮总滑, 不敢加油'\n"
            "    f1opt feedback --track suzuka --question 'T130R 出弯速度上不去, 总被甩开'\n"
            "  [sector]  某一段/扇区 (S2、直道段、连续弯段):\n"
            "    f1opt feedback --track suzuka --question 'S2 连续弯那一段车头太钝, 指向性差'\n"
            "    f1opt feedback --track monza --question 'S3 高速段车身不稳, 像在飘'\n"
            "    f1opt feedback --track monza --question '出弯时车尾总往外甩'\n"
            "    f1opt feedback --track monaco --question '刹车点晚一点就锁死前轮'\n"
            "  [overall] 整体感受 (全圈、整车平衡、总体策略):\n"
            "    f1opt feedback --track spa --question '轮胎温度左边比右边高很多'\n"
            "    f1opt feedback --track bahrain --question '圈速能再快多少?'\n"
            "    f1opt feedback --track suzuka --question 'ERS 怎么部署最快?'\n"
            "    f1opt feedback --track suzuka --question '感觉车还行, 还能优化吗?'\n"
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
    p_feedback.add_argument("--frames-json", default=None,
                            help="JSON file with telemetry frames (default: synthetic)")
    p_feedback.add_argument("--list-examples", action="store_true",
                            help="print driver feedback question examples and exit")
    p_feedback.add_argument("--json", action="store_true")
    p_feedback.set_defaults(func=cmd_feedback)

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
    print(f"error: {msg}", file=sys.stderr)


def _parse_setup_json(setup_json: str) -> CarSetup:
    """Parse a setup JSON string into a :class:`CarSetup`.

    Raises :class:`ValueError` on invalid JSON or schema validation failure.
    """
    try:
        setup_dict = json.loads(setup_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid setup JSON: {exc}") from exc
    if not isinstance(setup_dict, dict):
        raise ValueError("setup JSON must be a JSON object")
    return CarSetup(**setup_dict)


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Train the surrogate model."""
    from f1opt.model.train import train

    try:
        model = train(iterations=args.iterations, log=args.log, save=False)
        summary = {
            "status": "trained",
            "iterations": args.iterations,
            "model_version": model.model_version,
        }
        _print(summary, args.json)
        return 0
    except Exception as exc:  # noqa: BLE001 — surface any failure as exit 1
        _err(str(exc))
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict lap time for a setup."""
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
    except Exception as exc:  # noqa: BLE001 — model not ready / call-shape mismatch
        _err(str(exc))
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for optimal setup (differential or bayesian method)."""
    try:
        if args.method == "bayesian":
            from f1opt.model.bayesian import bayesian_search_setup

            result = bayesian_search_setup(
                args.track,
                DEFAULT_SETUP,
                n_iterations=args.iterations,
                acquisition="ei",
            )
            output = {
                "method": "bayesian",
                "track": args.track,
                "recommended_setup": result["recommended_setup"].model_dump(),
                "recommended_lap_time": result["recommended_lap_time"],
                "baseline_lap_time": result["baseline_lap_time"],
                "predicted_gain_s": result["predicted_gain_s"],
                "iterations": result["iterations"],
            }
        else:
            from f1opt.model.optimizer import search_setup

            result = search_setup(
                args.track,
                iterations=args.iterations,
                tire_wear_weight=args.tire_wear_weight,
            )
            output = {
                "method": "differential",
                "track": args.track,
                "recommended": result.recommended,
                "baseline": result.baseline,
                "predicted_gain_s": result.predicted_gain_s,
                "baseline_lap_time": result.baseline_lap_time,
                "recommended_lap_time": result.recommended_lap_time,
                "algorithm": result.algorithm,
                "iterations": result.iterations,
            }
        _print(output, args.json)
        return 0
    except Exception as exc:  # noqa: BLE001 — optimizer not available / failure
        _err(str(exc))
        return 1


def cmd_bayesian(args: argparse.Namespace) -> int:
    """Bayesian search."""
    try:
        from f1opt.model.bayesian import bayesian_search_setup

        result = bayesian_search_setup(
            args.track,
            DEFAULT_SETUP,
            n_iterations=args.iterations,
            acquisition=args.acquisition,
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
    except Exception as exc:  # noqa: BLE001 — BO not available / failure
        _err(str(exc))
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Run model validation."""
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
    except Exception as exc:  # noqa: BLE001 — validator not available / failure
        _err(str(exc))
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API server (uvicorn)."""
    try:
        import uvicorn

        if args.extended:
            from f1opt.api.extended_app import create_extended_app

            app = create_extended_app(start_listener=False)
        else:
            from f1opt.api.app import create_app

            app = create_app(start_listener=False)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except Exception as exc:  # noqa: BLE001 — server failure
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


def cmd_feedback(args: argparse.Namespace) -> int:
    """Generate driver feedback (rule-based or LLM-enhanced)."""
    from f1opt.driver.profile import DriverProfile
    from f1opt.feedback.prompts import FEEDBACK_EXAMPLES

    # --list-examples: 按 granularity 分组打印车手反馈示例并退出
    if args.list_examples:
        # 按 granularity 分组 (corner / sector / overall)
        grouped: dict[str, list[dict[str, str]]] = {
            "corner": [], "sector": [], "overall": [],
        }
        for ex in FEEDBACK_EXAMPLES:
            g = ex.get("granularity", "overall")
            grouped.setdefault(g, []).append(ex)
        examples_output = {
            "n_examples": len(FEEDBACK_EXAMPLES),
            "granularities": {
                "corner":  {"count": len(grouped["corner"]),
                            "desc": "精确到某个弯道 (T1、T130R、发卡弯)"},
                "sector":  {"count": len(grouped["sector"]),
                            "desc": "某一段/扇区 (S2、直道段、连续弯段)"},
                "overall": {"count": len(grouped["overall"]),
                            "desc": "整体感受 (全圈、整车平衡、总体策略)"},
            },
            "examples_by_granularity": grouped,
            "note": (
                "Driver feedback examples with three precision levels. "
                "Use --question '...' to ask. Supports Chinese and English. "
                "Granularity is automatically detected."
            ),
        }
        _print(examples_output, args.json)
        return 0

    if not args.track:
        _err("--track is required (unless --list-examples)")
        return 1
    if args.track not in TRACKS_BY_ID:
        _err(f"unknown track: {args.track}")
        return 1

    # Detect question granularity (corner/sector/overall)
    granularity_info: dict[str, Any] = {}
    if args.question:
        try:
            from f1opt.feedback.intent import classify_granularity
            gres = classify_granularity(args.question)
            granularity_info = {
                "granularity": gres.granularity,
                "confidence": gres.confidence,
                "corner_ref": gres.corner_ref,
                "matched_pattern": gres.matched_pattern,
            }
        except Exception as exc:
            granularity_info = {"error": str(exc)}

    # 车手画像
    if args.driver_style == "aggressive":
        dp = DriverProfile(
            brake_point_norm=0.85, throttle_smoothness=0.30,
            steer_smoothness=0.40, corner_balance_pref=0.70,
            aggression_score=0.90, consistency_score=0.60,
            ers_usage_intensity=0.85, drs_usage_efficiency=0.80,
        )
    elif args.driver_style == "conservative":
        dp = DriverProfile(
            brake_point_norm=0.40, throttle_smoothness=0.85,
            steer_smoothness=0.80, corner_balance_pref=0.30,
            aggression_score=0.30, consistency_score=0.85,
            ers_usage_intensity=0.40, drs_usage_efficiency=0.50,
        )
    else:
        dp = None

    # 遥测帧: 从 JSON 文件加载, 否则用合成帧
    if args.frames_json:
        try:
            with open(args.frames_json, encoding="utf-8") as f:
                frames = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            _err(f"failed to load frames: {exc}")
            return 1
    else:
        # 合成 60 帧 (1 秒) 默认遥测
        frames = [
            {
                "session_time": i / 60.0,
                "speed": 250.0 + 50.0 * (i % 60) / 60.0,
                "throttle": 0.8,
                "brake": 0.0,
                "steer": 0.1 * (i % 20) / 20.0,
                "g_lat": 1.5,
                "g_long": 0.3,
                "rpm": 10000,
                "tyre_temps": (95, 95, 95, 95),
                "tyre_wear": (5.0, 5.0, 5.0, 5.0),
            }
            for i in range(60)
        ]

    # 调用 FeedbackEngine
    from f1opt.feedback.engine import FeedbackEngine
    engine = FeedbackEngine()
    try:
        feedback = engine.run(
            frames, DEFAULT_SETUP.model_dump(), args.track,
            question=args.question,
            driver_profile=dp,
            session_id=args.session_id,
        )
    except Exception as exc:
        _err(f"feedback engine error: {exc}")
        return 1

    # Inject granularity info into feedback output (if not set by engine)
    if isinstance(feedback, dict):
        if "granularity" not in feedback and granularity_info:
            feedback["granularity"] = granularity_info.get("granularity", "overall")
        if "granularity_info" not in feedback:
            feedback["granularity_info"] = granularity_info

    _print(feedback, args.json)
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """Entry point; parse ``argv`` and dispatch to the matching subcommand.

    Returns the process exit code (0 on success, 1 on handler error, 2 on
    argparse usage error). ``--help`` and empty argv print help and return 0.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse raises SystemExit on --help (code 0) and on usage errors
        # (code 2). Surface the code so callers can distinguish.
        return exc.code if isinstance(exc.code, int) else 1
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
