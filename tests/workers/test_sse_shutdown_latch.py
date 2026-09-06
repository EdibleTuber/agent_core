"""The fixture must not let one server's teardown kill the next one's SSE.

This is a regression test for a failure that was CI-only for days:
tests/workers/test_client_pool.py::test_pool_mixed_transports would pass,
and then the next two tests that used the Streamable HTTP fixture would fail
with the server sending response headers and no body.
"""
import asyncio

import pytest
import uvicorn
from sse_starlette.sse import AppStatus

from agent_core.workers.client import MCPClient
from tests.workers.fixtures import _free_port, _reset_sse_shutdown_flag, build_stub


@pytest.fixture(autouse=True)
def _restore_global():
    """Never let this file's own poking leak into the rest of the session."""
    before = AppStatus.should_exit
    yield
    AppStatus.should_exit = before


async def _serve_and_initialize(linger: float) -> str:
    """Start a fixture server, do the MCP handshake, tear down."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(
        build_stub().http_app(), host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    client = MCPClient(endpoint=f"http://127.0.0.1:{port}/mcp")
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=5)
        outcome = "ok"
    except asyncio.TimeoutError:
        outcome = "headers but no body"
    except BaseException as exc:
        outcome = type(exc).__name__
    finally:
        try:
            await client.close()
        except BaseException:
            pass
        server.should_exit = True
        if linger:
            # Outlive sse-starlette's 0.5s watcher tick -- the tick is what
            # latches the global. A test that spawns a subprocess does this
            # without meaning to.
            await asyncio.sleep(linger)
        await task
        _reset_sse_shutdown_flag()
    return outcome



async def test_a_slow_teardown_does_not_poison_the_next_server():
    """The property, stated as a relationship rather than a magic number:
    a server started AFTER a slow teardown must serve exactly as well as one
    started before it.

    Without _reset_sse_shutdown_flag() the third call returns
    'headers but no body' -- verified by removing the reset and re-running.
    """
    assert await _serve_and_initialize(linger=0) == "ok"
    assert await _serve_and_initialize(linger=1.2) == "ok"
    assert await _serve_and_initialize(linger=0) == "ok", (
        "a previous server's teardown latched sse-starlette's process-global "
        "AppStatus.should_exit, so this server ended its SSE stream before "
        "sending a body")


async def test_the_reset_clears_a_latched_flag():
    """The narrow unit: AppStatus.should_exit is process-global and nothing in
    sse-starlette ever sets it back."""
    AppStatus.should_exit = True
    _reset_sse_shutdown_flag()
    assert AppStatus.should_exit is False
