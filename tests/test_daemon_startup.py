"""The daemon must finish astartup() before it dispatches anything.

start_unix_server defaults to start_serving=True, so the naive "bind, then
astartup" ordering already accepts connections and spawns handler tasks during
startup -- a chat turn can land against a half-populated tool executor.

Note on the discriminator below: `Server._start_serving()` is what calls
`sock.listen()` -- with the default `start_serving=True` that happens
synchronously inside `start_unix_server()`, before astartup() is ever
awaited, so a client can connect (and get serviced) the instant the socket
file exists. With `start_serving=False`, `sock.listen()` is deferred until
`serve_forever()` runs, i.e. strictly after astartup() returns in this
implementation. So a connect attempt made while astartup() is still running
must be refused (verified empirically: it raises ConnectionRefusedError,
not queued-then-serviced) -- that refusal is the proof that astartup() is
gating acceptance, not merely finishing before some unrelated log line.
"""
import asyncio

import pytest

from agent_core.agent import Agent
from agent_core.daemon import Daemon


class _Agent(Agent):
    name = "startup-probe"

    def __init__(self, socket_path):
        super().__init__()
        self.events = []
        self._socket_path = socket_path
        self.config = type("C", (), {"socket_path": socket_path})()

    async def astartup(self):
        self.events.append("astartup-begin")
        await asyncio.sleep(0.2)
        self.events.append("astartup-end")

    async def ashutdown(self):
        self.events.append("ashutdown")

    def system_prompt(self, ctx):
        return ""


async def test_astartup_completes_before_first_connection(tmp_path):
    sock = tmp_path / "probe.sock"
    agent = _Agent(sock)
    daemon = Daemon(agent)
    server_task = asyncio.create_task(daemon.serve())

    # Wait for the socket file to exist (bind() has happened). astartup()
    # is a 0.2s sleep, so there is a wide margin before it can have finished.
    for _ in range(100):
        if sock.exists():
            break
        await asyncio.sleep(0.01)
    assert sock.exists(), "socket should be bound before astartup finishes"

    # While astartup() is still running, sock.listen() has not been called
    # (see module docstring) -- a connect attempt now must be refused. If
    # this instead succeeds, start_serving=False regressed (or astartup is
    # no longer awaited before serve_forever()), and the test must fail.
    with pytest.raises(ConnectionRefusedError):
        await asyncio.open_unix_connection(str(sock))

    # Retry-connect until the daemon actually starts listening. In this
    # implementation that can only happen once astartup() has returned and
    # serve_forever() has run sock.listen() -- so success here is itself
    # proof that astartup already completed.
    writer = None
    for _ in range(200):
        try:
            _reader, writer = await asyncio.open_unix_connection(str(sock))
            break
        except ConnectionRefusedError:
            await asyncio.sleep(0.02)
    assert writer is not None, "daemon never started listening"
    agent.events.append("connected")
    writer.close()

    server_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await server_task

    assert "ashutdown" in agent.events

    assert agent.events.index("astartup-end") < agent.events.index("connected"), (
        f"a connection was serviced before astartup finished: {agent.events}")


def test_default_hooks_are_noops():
    class _Bare(Agent):
        name = "bare"
        def system_prompt(self, ctx):
            return ""

    a = _Bare()
    asyncio.run(a.astartup())
    asyncio.run(a.ashutdown())
