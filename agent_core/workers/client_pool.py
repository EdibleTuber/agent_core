"""MCPClientPool — one MCPClient per worker, lazy connect, reused across calls.

A long-lived pool held on the Agent instance (constructed in setup()).
Tools created by the discovery driver call back into this pool to run
their actual MCP exchanges.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_core.workers.client import MCPClient
from agent_core.workers.types import WorkerSpec


class MCPClientPool:
    """Holds one MCPClient per worker name, lazy-connecting on first use."""

    def __init__(self, specs: list[WorkerSpec]) -> None:
        self._specs: dict[str, WorkerSpec] = {s.name: s for s in specs}
        self._clients: dict[str, MCPClient] = {}
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self, worker: str) -> MCPClient:
        if worker not in self._specs:
            raise KeyError(f"no worker named {worker!r} in this pool")
        async with self._connect_lock:
            if worker not in self._clients:
                spec = self._specs[worker]
                client = MCPClient.from_spec(spec)
                await client.connect()
                await client.initialize()
                self._clients[worker] = client
        return self._clients[worker]

    async def list_tools(self, worker: str):
        client = await self._ensure_connected(worker)
        return await client.list_tools()

    async def call_tool(self, worker: str, tool: str, arguments: dict[str, Any], ctx: Any = None):
        client = await self._ensure_connected(worker)
        return await client.call_tool(tool, arguments)

    async def close_all(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except BaseException:
                pass
        self._clients.clear()
