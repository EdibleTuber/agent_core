"""WorkerRegistry — loads workers.yaml and exposes workers by name.

Phase 0 ships only the data layer: parsing, validation, lookup. A live
MCP client (initialize → list_tools → register_tools()) will land in a
later phase when an actual worker connects.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from agent_core.workers.types import WorkerSpec


class WorkerNotFoundError(KeyError):
    """Raised when WorkerRegistry.get() is called with an unknown name."""


class WorkerRegistry:
    """In-memory store of WorkerSpec entries loaded from a workers.yaml."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerSpec] = {}

    @classmethod
    def load(cls, path: Path | str) -> WorkerRegistry:
        """Parse workers.yaml at the given path. Raises FileNotFoundError
        if the file is missing, ValidationError if entries are malformed."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"workers.yaml not found at {path}")

        data = yaml.safe_load(path.read_text()) or {}
        raw_workers = data.get("workers", {}) or {}

        reg = cls()
        for name, fields in raw_workers.items():
            spec = WorkerSpec(name=name, **fields)
            reg._workers[spec.name] = spec
        return reg

    def get(self, name: str) -> WorkerSpec:
        if name not in self._workers:
            raise WorkerNotFoundError(f"no worker registered with name {name!r}")
        return self._workers[name]

    def all(self) -> list[WorkerSpec]:
        return list(self._workers.values())

    def add(self, spec: WorkerSpec) -> None:
        self._workers[spec.name] = spec
