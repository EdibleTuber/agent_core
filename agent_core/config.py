"""Base configuration for agent_core agents.

Defines a dataclass with the universally-shared infrastructure fields and an
env-var loader that derives env-var prefix from the agent's name (e.g. `pal` ->
`PAL_`, `re-lab` -> `RELAB_`). Agents subclass `BaseConfig` to add domain
fields; the same loader supports any subclass.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_type_hints


def _default_socket_path(agent_name: str) -> Path:
    """Derive the default socket path: $XDG_RUNTIME_DIR/<agent_name>.sock."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime_dir) / f"{agent_name}.sock"


@dataclass
class BaseConfig:
    """Universally-shared agent configuration.

    Subclass to add domain fields. Field names map to env vars as
    `<PREFIX><FIELD_NAME_UPPER>`, where the prefix is derived from the agent
    name unless explicitly overridden.
    """
    inference_url: str = "http://192.168.1.14:11434"
    model: str = "Qwen3.5-35B-A3B-Q4_K_M"
    socket_path: Path | None = None
    history_depth: int = 50
    vault_path: Path = field(default_factory=lambda: Path.home() / "vault")
    collection_id: str = "vault"
    username: str = "user"
    searxng_url: str = "http://192.168.1.14:8080"
    fetch_max_bytes: int = 2_000_000
    fetch_timeout: int = 30
    max_response_tokens: int = 4096
    batch_enabled: bool = False
    batch_inference_url: str = "http://192.168.1.14:11434"
    batch_model: str = "gemma-4-E4B-it-Q4_K_M"
    scratchpad_max_bytes: int = 2048


def _coerce(field_type, raw: str):
    """Coerce a raw env-var string to a typed value based on the field type."""
    if isinstance(field_type, type):
        if field_type is int:
            return int(raw)
        if field_type is bool:
            return raw.strip().lower() in ("true", "1", "yes")
        if field_type is Path:
            return Path(raw)
        if field_type is str:
            return raw
    # Unions like `Path | None`: try each member.
    args = getattr(field_type, "__args__", ())
    if args:
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, raw)
            except (TypeError, ValueError):
                continue
    return raw  # fallback: treat as string


def load_config(
    config_cls: type[BaseConfig],
    agent_name: str,
    env_prefix: str | None = None,
) -> BaseConfig:
    """Load `config_cls` from env vars.

    The env-var prefix is `<agent_name.upper().replace('-', '')>_` (hyphens
    stripped entirely, e.g. `re-lab` -> `RELAB_`) unless `env_prefix` is
    supplied explicitly. The returned config has its `socket_path` derived
    from `agent_name` if not set via env var.
    """
    prefix = (
        env_prefix
        if env_prefix is not None
        else f"{agent_name.upper().replace('-', '')}_"
    )
    type_hints = get_type_hints(config_cls)
    kwargs: dict = {}
    for f in fields(config_cls):
        env_name = f"{prefix}{f.name.upper()}"
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        field_type = type_hints.get(f.name, str)
        try:
            kwargs[f.name] = _coerce(field_type, raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid value for {env_name}: {raw!r} ({exc})",
            ) from exc
    cfg = config_cls(**kwargs)
    if cfg.socket_path is None:
        cfg.socket_path = _default_socket_path(agent_name)
    return cfg
