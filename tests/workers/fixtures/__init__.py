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



def _reset_sse_shutdown_flag() -> None:
    """Undo sse-starlette's PROCESS-GLOBAL shutdown latch after each server.

    sse_starlette.sse.AppStatus.should_exit is a class attribute -- one value
    for the whole process -- and nothing ever sets it back to False. A
    background watcher polls the running uvicorn server's `should_exit` every
    0.5s and latches the global when it sees it. Once latched,
    _listen_for_exit_signal() returns immediately for every LATER SSE
    response, so the stream ends the moment it starts: the ASGI app sends
    http.response.start and no body, and uvicorn logs

        ASGI callable returned without completing response.

    That is fatal here because MCP's Streamable HTTP replies to `initialize`
    over SSE. The client gets headers, waits for a body that never comes, and
    times out -- while the server looks healthy and every other test that
    touched the fixture earlier passed.

    This fixture starts and stops one server PER TEST in a single process, so
    the first teardown that outlives a 0.5s watcher tick kills the transport
    for every test after it. That is why the failure was ordering-dependent
    and why it showed up on CI first: the trigger was the one test slow enough
    to span a tick (it spawns a stdio subprocess), and a slower machine makes
    more tests slow enough.
    """
    try:
        from sse_starlette.sse import AppStatus
    except ImportError:      # pragma: no cover - sse-starlette is an mcp dep
        return
    AppStatus.should_exit = False


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
        _reset_sse_shutdown_flag()
