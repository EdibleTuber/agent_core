from agent_core.workers.registry import WorkerRegistry


def _write(tmp_path, text):
    p = tmp_path / "workers.yaml"
    p.write_text(text)
    return p


def test_risk_overrides_parsed_as_pattern_tier_pairs(tmp_path):
    path = _write(tmp_path, """
workers:
  frida:
    command: x
    transport: stdio
    risk_default: high
risk_overrides:
  - ["frida_execute_script", "critical"]
  - ["frida_write_memory", "high"]
""")
    reg = WorkerRegistry.load(path)
    assert reg.risk_overrides() == [
        ("frida_execute_script", "critical"),
        ("frida_write_memory", "high"),
    ]


def test_risk_overrides_absent_returns_empty(tmp_path):
    path = _write(tmp_path, """
workers:
  frida:
    command: x
    transport: stdio
    risk_default: high
""")
    reg = WorkerRegistry.load(path)
    assert reg.risk_overrides() == []
