"""Tests for :mod:`f1opt.observability.audit`.

Covers the append-only JSONL :class:`AuditLogger`: record construction,
file persistence, the ``tail`` reader, best-effort error handling, the
process-wide singleton and the :func:`audit_log` convenience wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1opt.observability import audit
from f1opt.observability.audit import AuditLogger, audit_log, get_audit_logger


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #
def test_explicit_path(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    assert logger.path == log_path
    assert logger.count == 0


def test_env_path_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = tmp_path / "env" / "audit.jsonl"
    monkeypatch.setenv("F1OPT_AUDIT_PATH", str(env_path))
    logger = AuditLogger()
    assert logger.path == env_path


def test_default_path_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("F1OPT_AUDIT_PATH", raising=False)
    logger = AuditLogger()
    assert logger.path == Path("data_store") / "audit" / "audit.jsonl"


# --------------------------------------------------------------------------- #
# log()
# --------------------------------------------------------------------------- #
def test_log_returns_record_with_expected_fields(tmp_path: Path) -> None:
    logger = AuditLogger(path=tmp_path / "audit.jsonl")
    record = logger.log(
        actor="user@example.com",
        action="setup.change",
        resource="setup/suzuka",
        outcome="success",
        ip="192.0.2.1",
        user_agent="F1OPT-CLI/0.1",
        metadata={"field": "front_wing", "old": 25, "new": 30},
    )
    assert record["actor"] == "user@example.com"
    assert record["action"] == "setup.change"
    assert record["resource"] == "setup/suzuka"
    assert record["outcome"] == "success"
    assert record["ip"] == "192.0.2.1"
    assert record["user_agent"] == "F1OPT-CLI/0.1"
    assert record["metadata"] == {"field": "front_wing", "old": 25, "new": 30}
    assert "timestamp" in record


def test_log_defaults(tmp_path: Path) -> None:
    """Optional fields default sensibly (success outcome, empty metadata)."""
    logger = AuditLogger(path=tmp_path / "audit.jsonl")
    record = logger.log(actor="a", action="b", resource="c")
    assert record["outcome"] == "success"
    assert record["ip"] is None
    assert record["user_agent"] is None
    assert record["metadata"] == {}


def test_log_writes_jsonl_and_creates_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "dir" / "audit.jsonl"
    logger = AuditLogger(path=log_path)
    logger.log(actor="a", action="login", resource="session")
    logger.log(actor="b", action="logout", resource="session")

    assert log_path.is_file()
    assert logger.count == 2
    assert logger.last_error is None

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["actor"] == "a"
    assert first["action"] == "login"


def test_log_appends_across_instances(tmp_path: Path) -> None:
    """A second logger appends rather than truncating."""
    log_path = tmp_path / "audit.jsonl"
    AuditLogger(path=log_path).log(actor="a", action="x", resource="r")
    AuditLogger(path=log_path).log(actor="b", action="y", resource="r")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_log_best_effort_on_write_error(tmp_path: Path) -> None:
    """A write failure is captured in last_error and never raised."""
    # Point the audit path at a location whose parent is a file, so
    # mkdir(parents=True) fails with OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    logger = AuditLogger(path=blocker / "audit.jsonl")

    record = logger.log(actor="a", action="x", resource="r")
    # Record is still returned.
    assert record["actor"] == "a"
    # But the write failed and was recorded, not raised.
    assert logger.count == 0
    assert logger.last_error is not None


# --------------------------------------------------------------------------- #
# tail()
# --------------------------------------------------------------------------- #
def test_tail_missing_file_returns_empty(tmp_path: Path) -> None:
    logger = AuditLogger(path=tmp_path / "does_not_exist.jsonl")
    assert logger.tail() == []


def test_tail_returns_last_n_records(tmp_path: Path) -> None:
    logger = AuditLogger(path=tmp_path / "audit.jsonl")
    for i in range(10):
        logger.log(actor=f"user{i}", action="act", resource="r")

    last3 = logger.tail(n=3)
    assert len(last3) == 3
    assert [r["actor"] for r in last3] == ["user7", "user8", "user9"]


def test_tail_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text(
        '{"actor": "a"}\n'
        "\n"  # blank
        "not-json\n"
        '{"actor": "b"}\n',
        encoding="utf-8",
    )
    logger = AuditLogger(path=log_path)
    records = logger.tail()
    assert [r["actor"] for r in records] == ["a", "b"]


# --------------------------------------------------------------------------- #
# singleton + convenience wrapper
# --------------------------------------------------------------------------- #
def test_get_audit_logger_is_singleton() -> None:
    assert get_audit_logger() is get_audit_logger()


def test_audit_log_wrapper_uses_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reset the module-level singleton so it picks up our temp path.
    monkeypatch.setattr(audit, "_GLOBAL_LOGGER", None)
    monkeypatch.setenv("F1OPT_AUDIT_PATH", str(tmp_path / "audit.jsonl"))

    record = audit_log(actor="a", action="predict", resource="lap")
    assert record["action"] == "predict"
    assert get_audit_logger().count == 1
