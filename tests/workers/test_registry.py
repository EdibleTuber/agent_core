"""Tests for WorkerRegistry — workers.yaml loader and lookup."""
from pathlib import Path

import pytest

from agent_core.workers.registry import WorkerRegistry, WorkerNotFoundError
from agent_core.workers.types import WorkerSpec


SAMPLE_YAML = """\
workers:
  android:
    endpoint: http://localhost:9100/mcp
    transport: streamable_http
    risk_default: medium
    container: pare-android-worker
    capability_tags: [mobile, dynamic, android]
  static:
    endpoint: http://localhost:8000
    transport: http_job_api
    risk_default: low
  ghidra:
    endpoint: ${GHIDRA_MCP_URL}
    transport: streamable_http
    risk_default: medium
    kind: external_mcp
"""


def test_registry_loads_from_yaml(tmp_path):
    p = tmp_path / "workers.yaml"
    p.write_text(SAMPLE_YAML)

    reg = WorkerRegistry.load(p)

    assert {w.name for w in reg.all()} == {"android", "static", "ghidra"}
    android = reg.get("android")
    assert isinstance(android, WorkerSpec)
    assert android.transport == "streamable_http"
    assert android.capability_tags == ["mobile", "dynamic", "android"]


def test_registry_get_unknown_raises(tmp_path):
    p = tmp_path / "workers.yaml"
    p.write_text(SAMPLE_YAML)

    reg = WorkerRegistry.load(p)
    with pytest.raises(WorkerNotFoundError):
        reg.get("nonexistent")


def test_registry_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        WorkerRegistry.load(tmp_path / "nope.yaml")


def test_registry_external_mcp_kind(tmp_path):
    p = tmp_path / "workers.yaml"
    p.write_text(SAMPLE_YAML)

    reg = WorkerRegistry.load(p)
    assert reg.get("ghidra").kind == "external_mcp"
    assert reg.get("android").kind == "internal"  # default


def test_registry_add_appends(tmp_path):
    p = tmp_path / "workers.yaml"
    p.write_text(SAMPLE_YAML)
    reg = WorkerRegistry.load(p)

    reg.add(WorkerSpec(
        name="ios",
        endpoint="http://localhost:9101/mcp",
        transport="streamable_http",
        risk_default="medium",
    ))
    assert {w.name for w in reg.all()} == {"android", "static", "ghidra", "ios"}
