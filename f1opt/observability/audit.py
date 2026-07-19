"""Audit log for security-sensitive operations (Iter-170).

Provides :class:`AuditLogger` — a lightweight, append-only audit trail for
security-sensitive API operations (login, setup changes, feedback
submission, model retraining, etc.). Each entry is a structured JSON
record with timestamp, actor, action, resource, outcome, and request
metadata (IP, user-agent).

Design goals:
- **Append-only**: records are appended to a JSONL file; no in-place
  modification or deletion. The file is opened in ``"a"`` mode and
  flushed after each write.
- **Structured**: each record is a JSON object with stable fields
  (``timestamp`` / ``actor`` / ``action`` / ``resource`` / ``outcome``
  / ``ip`` / ``user_agent`` / ``metadata``).
- **Best-effort**: writes never raise — a write failure is reported
  via :data:`AuditLogger.last_error` but does not interrupt the
  audited operation.
- **Process-safe**: a single process-wide :class:`AuditLogger`
  instance is exposed via :func:`get_audit_logger`; concurrent writes
  are serialized by an internal :class:`threading.Lock`.

The audit log path defaults to ``{data_dir}/audit/audit.jsonl`` and
is created on first write. The path can be overridden via the
``F1OPT_AUDIT_PATH`` environment variable.

Usage::

    from f1opt.observability.audit import get_audit_logger

    audit = get_audit_logger()
    audit.log(
        actor="user@example.com",
        action="setup.change",
        resource="setup/suzuka",
        outcome="success",
        ip="192.0.2.1",
        user_agent="F1OPT-CLI/0.1",
        metadata={"field": "front_wing", "old": 25, "new": 30},
    )

The audit log is exposed read-only via the ``GET /api/audit`` endpoint
(admin-only in production; not exposed in dev mode by default).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["AuditLogger", "get_audit_logger", "audit_log"]


class AuditLogger:
    """Append-only JSONL audit logger.

    Each call to :meth:`log` writes one JSON record (plus newline) to
    the configured file. Writes are serialized by a process-wide lock;
    write failures are captured in :attr:`last_error` and never raised.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            env_path = os.environ.get("F1OPT_AUDIT_PATH")
            if env_path:
                path = Path(env_path)
            else:
                # Default: {cwd}/data_store/audit/audit.jsonl
                # (matches f1opt.config.Settings.data_dir default).
                path = Path("data_store") / "audit" / "audit.jsonl"
        self._path = Path(path)
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self._count: int = 0

    @property
    def path(self) -> Path:
        """Absolute path to the audit JSONL file."""
        return self._path

    @property
    def count(self) -> int:
        """Number of records successfully written since instantiation."""
        return self._count

    def log(
        self,
        *,
        actor: str,
        action: str,
        resource: str,
        outcome: str = "success",
        ip: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one audit record. Returns the record dict.

        ``outcome`` should be one of ``"success"`` / ``"failure"`` /
        ``"denied"`` / ``"error"``. ``metadata`` is an arbitrary
        JSON-serializable dict (kept small — for large payloads, store
        out-of-band and reference by id).
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "actor": str(actor),
            "action": str(action),
            "resource": str(resource),
            "outcome": str(outcome),
            "ip": ip,
            "user_agent": user_agent,
            "metadata": metadata or {},
        }
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str, sort_keys=True))
                    fh.write("\n")
                    fh.flush()
                self._count += 1
                self.last_error = None
            except OSError as exc:
                # Best-effort: never raise from audit log.
                self.last_error = f"{type(exc).__name__}: {exc}"
        return record

    def tail(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the last ``n`` audit records (most-recent last).

        Reads the JSONL file from the end. Returns an empty list if the
        file does not exist or is unreadable.
        """
        if not self._path.is_file():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            return []
        # Parse only the last n non-empty lines.
        records: list[dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


# Process-wide singleton (lazy).
_GLOBAL_LOGGER: AuditLogger | None = None
_GLOBAL_LOCK = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Return the process-wide :class:`AuditLogger` singleton."""
    global _GLOBAL_LOGGER
    with _GLOBAL_LOCK:
        if _GLOBAL_LOGGER is None:
            _GLOBAL_LOGGER = AuditLogger()
        return _GLOBAL_LOGGER


def audit_log(
    *,
    actor: str,
    action: str,
    resource: str,
    outcome: str = "success",
    ip: str | None = None,
    user_agent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: ``get_audit_logger().log(...)``."""
    return get_audit_logger().log(
        actor=actor,
        action=action,
        resource=resource,
        outcome=outcome,
        ip=ip,
        user_agent=user_agent,
        metadata=metadata,
    )
