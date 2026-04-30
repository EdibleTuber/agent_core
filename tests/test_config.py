"""Tests for agent_core.config.BaseConfig and load_config."""
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_core.config import BaseConfig, load_config


def test_baseconfig_defaults():
    cfg = BaseConfig()
    assert cfg.inference_url == "http://192.168.1.14:11434"
    assert cfg.history_depth == 50
    assert cfg.max_response_tokens == 4096
    assert cfg.batch_enabled is False
    assert cfg.socket_path is None  # Resolved by load_config based on agent_name.


def test_load_config_no_env_uses_defaults(monkeypatch, tmp_path):
    # Wipe any PAL_* env vars that might leak in.
    for key in list(os.environ):
        if key.startswith("PAL_"):
            monkeypatch.delenv(key, raising=False)
    cfg = load_config(BaseConfig, agent_name="pal")
    assert cfg.inference_url == "http://192.168.1.14:11434"
    # socket_path should be derived since not explicitly set.
    assert cfg.socket_path is not None
    assert str(cfg.socket_path).endswith("pal.sock")


def test_load_config_env_override_str(monkeypatch):
    monkeypatch.setenv("PAL_INFERENCE_URL", "http://example:1234")
    cfg = load_config(BaseConfig, agent_name="pal")
    assert cfg.inference_url == "http://example:1234"


def test_load_config_env_override_int(monkeypatch):
    monkeypatch.setenv("PAL_HISTORY_DEPTH", "200")
    cfg = load_config(BaseConfig, agent_name="pal")
    assert cfg.history_depth == 200


def test_load_config_env_override_bool(monkeypatch):
    monkeypatch.setenv("PAL_BATCH_ENABLED", "true")
    cfg = load_config(BaseConfig, agent_name="pal")
    assert cfg.batch_enabled is True


def test_load_config_env_override_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_VAULT_PATH", str(tmp_path))
    cfg = load_config(BaseConfig, agent_name="pal")
    assert cfg.vault_path == tmp_path


def test_load_config_prefix_derived_from_agent_name(monkeypatch):
    monkeypatch.setenv("RELAB_INFERENCE_URL", "http://relab:5555")
    cfg = load_config(BaseConfig, agent_name="re-lab")  # hyphen
    # Hyphens stripped entirely in prefix derivation: "re-lab" -> "RELAB_".
    assert cfg.inference_url == "http://relab:5555"


def test_load_config_prefix_explicit_override(monkeypatch):
    monkeypatch.setenv("MYPREFIX_INFERENCE_URL", "http://my:6666")
    cfg = load_config(BaseConfig, agent_name="ignored", env_prefix="MYPREFIX_")
    assert cfg.inference_url == "http://my:6666"


def test_load_config_socket_path_explicit(monkeypatch):
    monkeypatch.setenv("PAL_SOCKET_PATH", "/tmp/custom.sock")
    cfg = load_config(BaseConfig, agent_name="pal")
    assert cfg.socket_path == Path("/tmp/custom.sock")


def test_load_config_subclass_extra_field(monkeypatch):
    @dataclass
    class MyConfig(BaseConfig):
        my_extra: int = 42

    monkeypatch.setenv("PAL_MY_EXTRA", "999")
    cfg = load_config(MyConfig, agent_name="pal")
    assert cfg.my_extra == 999
    # BaseConfig fields still load correctly via the same prefix.
    assert cfg.inference_url == "http://192.168.1.14:11434"  # unchanged from default


def test_load_config_invalid_int_raises_with_field_name(monkeypatch):
    """A malformed int env var raises ValueError mentioning the env var name,
    so deploy-time misconfigs surface clearly."""
    monkeypatch.setenv("PAL_HISTORY_DEPTH", "banana")
    with pytest.raises(ValueError, match="PAL_HISTORY_DEPTH"):
        load_config(BaseConfig, agent_name="pal")
