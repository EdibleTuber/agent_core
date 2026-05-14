"""Tests for AuditLog — append-only JSONL with daily rotation."""
import json
from datetime import datetime, timezone

import pytest

from agent_core.workers.audit import AuditLog
from agent_core.workers.types import AuditEntry


def _make_entry(session_guid: str = "11111111-1111-4111-9111-111111111111") -> AuditEntry:
    return AuditEntry(
        request_id="req-1",
        worker="android",
        tool="attach",
        args={"package": "com.example"},
        declared_tier="low",
        effective_tier="low",
        outcome="ok",
        latency_ms=15,
        session_guid=session_guid,
        worker_contract_version=1,
    )


def test_audit_log_writes_jsonl(tmp_path):
    log = AuditLog(directory=tmp_path)
    log.append(_make_entry())
    log.append(_make_entry())

    files = sorted(tmp_path.glob("audit-*.jsonl"))
    assert len(files) == 1  # same day → same file

    lines = files[0].read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert all(p["session_guid"] == "11111111-1111-4111-9111-111111111111" for p in parsed)


def test_audit_log_rotates_on_date_change(tmp_path, monkeypatch):
    """When the UTC date changes, AuditLog opens a new file."""
    log = AuditLog(directory=tmp_path)

    # First entry on day 1.
    monkeypatch.setattr(
        "agent_core.workers.audit._today_utc",
        lambda: "2026-05-13",
    )
    log.append(_make_entry())

    # Second entry on day 2.
    monkeypatch.setattr(
        "agent_core.workers.audit._today_utc",
        lambda: "2026-05-14",
    )
    log.append(_make_entry())

    files = sorted(tmp_path.glob("audit-*.jsonl"))
    assert len(files) == 2
    assert files[0].name == "audit-2026-05-13.jsonl"
    assert files[1].name == "audit-2026-05-14.jsonl"


def test_audit_log_creates_directory(tmp_path):
    """AuditLog creates the audit directory if it doesn't exist."""
    nested = tmp_path / "projects" / "scratch" / "audit"
    log = AuditLog(directory=nested)
    log.append(_make_entry())
    assert nested.exists()
    assert any(nested.glob("audit-*.jsonl"))
