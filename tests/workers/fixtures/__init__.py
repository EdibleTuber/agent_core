"""Pytest fixtures for live Streamable HTTP testing."""
import asyncio
import contextlib
import socket

import pytest
import uvicorn

from tests.workers.fixtures.streamable_http_stub import build_stub


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def streamable_http_fixture():
    """Start the FastMCP stub on a free port and yield its base URL.

    Teardown stops the uvicorn server cleanly."""
    port = _free_port()
    stub = build_stub()
    # http_app() confirmed via inspection to return the Starlette ASGI app
    # with a single route mounted at /mcp.
    app = stub.http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    # Give uvicorn a moment to bind.
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("uvicorn did not start within 2.5s")

    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await server_task
