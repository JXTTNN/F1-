"""Tests for :mod:`f1opt.cli` (argparse-based command-line interface).

Covers:
- :func:`build_parser` returns an :class:`argparse.ArgumentParser` and parses
  each subcommand's flags correctly.
- :func:`main` dispatches to subcommands, prints output, and returns 0 on
  success / non-zero on error.
- :func:`format_output` returns valid JSON in JSON mode and a non-JSON table
  otherwise.

The ``train`` subcommand uses ``save=False`` internally so it never overwrites
the on-disk surrogate; the ``predict`` test serializes
:data:`DEFAULT_SETUP` to JSON so the setup validates cleanly.
"""

from __future__ import annotations

import argparse
import json

import pytest

from f1opt.cli import build_parser, format_output, main
from f1opt.data.setup_schema import DEFAULT_SETUP


# --------------------------------------------------------------------------- #
# build_parser
# --------------------------------------------------------------------------- #
def test_build_parser_returns_argument_parser() -> None:
    """build_parser() returns an argparse.ArgumentParser instance."""
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parse_train_iterations() -> None:
    """'train --iterations 100' → args.iterations == 100."""
    parser = build_parser()
    args = parser.parse_args(["train", "--iterations", "100"])
    assert args.iterations == 100
    assert args.log is False


def test_parse_predict_track() -> None:
    """'predict --track melbourne --setup-json {}' → args.track == 'melbourne'."""
    parser = build_parser()
    args = parser.parse_args(
        ["predict", "--track", "melbourne", "--setup-json", "{}"]
    )
    assert args.track == "melbourne"
    assert args.setup_json == "{}"


def test_parse_search_method_bayesian() -> None:
    """'search --track melbourne --method bayesian' → args.method == 'bayesian'."""
    parser = build_parser()
    args = parser.parse_args(
        ["search", "--track", "melbourne", "--method", "bayesian"]
    )
    assert args.method == "bayesian"


def test_parse_bayesian_iterations_acquisition() -> None:
    """'bayesian --track melbourne --iterations 10 --acquisition ucb' parses."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "bayesian",
            "--track",
            "melbourne",
            "--iterations",
            "10",
            "--acquisition",
            "ucb",
        ]
    )
    assert args.track == "melbourne"
    assert args.iterations == 10
    assert args.acquisition == "ucb"


def test_parse_validate_track() -> None:
    """'validate --track melbourne' parses."""
    parser = build_parser()
    args = parser.parse_args(["validate", "--track", "melbourne"])
    assert args.track == "melbourne"


def test_parse_serve_port_extended() -> None:
    """'serve --port 9000 --extended' parses."""
    parser = build_parser()
    args = parser.parse_args(["serve", "--port", "9000", "--extended"])
    assert args.port == 9000
    assert args.extended is True


def test_parse_tracks_list() -> None:
    """'tracks list' parses and dispatches to cmd_tracks_list."""
    parser = build_parser()
    args = parser.parse_args(["tracks", "list"])
    assert args.tracks_command == "list"


def test_parse_tracks_info_track() -> None:
    """'tracks info --track madrid' parses."""
    parser = build_parser()
    args = parser.parse_args(["tracks", "info", "--track", "madrid"])
    assert args.track == "madrid"


def test_parse_setup_default() -> None:
    """'setup default' parses."""
    parser = build_parser()
    args = parser.parse_args(["setup", "default"])
    assert args.setup_command == "default"


def test_parse_setup_validate() -> None:
    """'setup validate --setup-json {}' parses."""
    parser = build_parser()
    args = parser.parse_args(["setup", "validate", "--setup-json", "{}"])
    assert args.setup_json == "{}"


# --------------------------------------------------------------------------- #
# main — dispatch + exit codes
# --------------------------------------------------------------------------- #
def test_main_train_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    """main(['train', '--iterations', '10', '--log']) returns 0.

    Uses save=False internally so the on-disk surrogate is not overwritten.
    """
    rc = main(["train", "--iterations", "10", "--log"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "trained" in out


def test_main_tracks_list_returns_0_with_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main(['tracks', 'list']) returns 0 and output contains track names."""
    rc = main(["tracks", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "melbourne" in out
    assert "madrid" in out


def test_main_tracks_info_returns_0(capsys: pytest.CaptureFixture[str]) -> None:
    """main(['tracks', 'info', '--track', 'melbourne']) returns 0."""
    rc = main(["tracks", "info", "--track", "melbourne"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "melbourne" in out
    assert "Australian Grand Prix" in out


def test_main_setup_default_returns_0_with_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main(['setup', 'default']) returns 0 and output has setup fields."""
    rc = main(["setup", "default"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "front_wing" in out
    assert "rear_wing" in out
    assert "fuel_load" in out


def test_main_predict_valid_setup_returns_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main(['predict', ...]) with a VALID full setup JSON returns 0.

    Uses DEFAULT_SETUP serialized as JSON so CarSetup validation succeeds.
    """
    setup_json = json.dumps(DEFAULT_SETUP.model_dump())
    rc = main(
        ["predict", "--track", "melbourne", "--setup-json", setup_json]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "lap_time" in out


def test_main_tracks_list_json_valid_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main(['tracks', 'list', '--json']) returns 0 and output is valid JSON."""
    rc = main(["tracks", "list", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 24
    assert any(t["track_id"] == "melbourne" for t in data)


def test_main_setup_default_json_valid_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main(['setup', 'default', '--json']) returns 0 and output is parseable JSON."""
    rc = main(["setup", "default", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, dict)
    assert data["front_wing"] == DEFAULT_SETUP.front_wing
    assert data["fuel_load"] == DEFAULT_SETUP.fuel_load


def test_main_no_args_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """main([]) prints help and returns 0 (or 2 — both acceptable per spec)."""
    rc = main([])
    assert rc in (0, 2)


def test_main_help_returns_0_or_2() -> None:
    """main(['--help']) returns 0 (argparse help exit code)."""
    rc = main(["--help"])
    assert rc in (0, 2)


def test_main_unknown_command_returns_nonzero() -> None:
    """main(['unknown-command']) returns a non-zero exit code."""
    rc = main(["unknown-command"])
    assert rc != 0


# --------------------------------------------------------------------------- #
# format_output
# --------------------------------------------------------------------------- #
def test_format_output_json_mode_returns_valid_json() -> None:
    """format_output(data, json_mode=True) returns a valid JSON string."""
    data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
    out = format_output(data, json_mode=True)
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed == data


def test_format_output_table_mode_returns_non_json() -> None:
    """format_output(data, json_mode=False) returns a non-JSON string."""
    data = {"a": 1, "b": 2}
    out = format_output(data, json_mode=False)
    assert isinstance(out, str)
    # Table format ("key: value") is not valid JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_format_output_list_of_dicts_table_has_headers() -> None:
    """Table mode for a list of dicts includes a header row and tab separators."""
    data = [{"track_id": "melbourne", "country": "Australia"},
            {"track_id": "madrid", "country": "Spain"}]
    out = format_output(data, json_mode=False)
    lines = out.splitlines()
    assert len(lines) == 3
    assert "track_id" in lines[0]
    assert "melbourne" in lines[1]
    assert "madrid" in lines[2]


def test_main_setup_validate_invalid_setup_returns_1(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main(['setup', 'validate', '--setup-json', '{"front_wing":25}']) returns 1.

    A partial setup missing the other 18 required fields fails CarSetup
    validation, so the handler prints to stderr and exits 1.
    """
    rc = main(["setup", "validate", "--setup-json", '{"front_wing": 25}'])
    assert rc == 1
