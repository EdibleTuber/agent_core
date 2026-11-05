"""The manager token and localhost defaults (inference-server phase 6, Q15 and Q16).

The manager will refuse any request without its token (except /health), so
every call an agent makes to it must carry `Authorization: Bearer <token>`,
and nothing else the agent talks to (SearXNG, fetched pages) may see it.

The header tests swap a MockTransport into every httpx.AsyncClient the code
under test constructs, keeping whatever headers the code itself passes. They
check what goes on the wire, so they hold whether the header is set as a
client default or per request.
"""
import asyncio
from unittest.mock import patch

import httpx
import pytest

from agent_core.agent import Agent
from agent_core.config import BaseConfig, load_config
from agent_core.inference import InferenceClient
from agent_core.retrieval import RetrievalClient
from agent_core.runtime import run_daemon

TOKEN = "s3cret-manager-token"

_COMPLETION = {"choices": [{"message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop"}]}
_SSE = ('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n').encode()


def _manager_answer(request: httpx.Request) -> httpx.Response:
    """A minimal manager plus SearXNG: whatever each client call needs to succeed."""
    path = request.url.path
    if path == "/v1/chat/completions":
        if b'"stream": true' in request.content or b'"stream":true' in request.content:
            return httpx.Response(200, content=_SSE,
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_COMPLETION)
    if path.endswith("/search") and path.startswith("/collections/"):
        return httpx.Response(200, json={"results": []})
    if "/docs/" in path:
        return httpx.Response(200, json={"id": "d", "content": "c"})
    if path.endswith("/reindex") and request.method == "POST":
        return httpx.Response(202, json={"job_id": "j1"})
    if path.endswith("/reindex/status") or "/reindex/" in path:
        return httpx.Response(200, json={"job_id": "j1", "state": "done"})
    if path == "/search":  # SearXNG
        return httpx.Response(200, json={"results": []})
    return httpx.Response(404)


@pytest.fixture()
def wire(monkeypatch):
    """Record every request any httpx.AsyncClient built during the test sends."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _manager_answer(request)

    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return seen


def _auth(request: httpx.Request) -> str | None:
    return request.headers.get("authorization")


async def _drive_inference(client: InferenceClient, path: str) -> None:
    msgs = [{"role": "user", "content": "hi"}]
    if path == "complete":
        await client.complete(messages=msgs)
    else:
        async for _ in client.stream(messages=msgs):
            pass


async def _drive_retrieval(client: RetrievalClient) -> None:
    await client.search("q")
    await client.get_document("notes/a.md")
    assert await client.trigger_reindex() is not None
    assert await client.get_reindex_status() is not None
    assert await client.get_reindex_job("j1") is not None


# --- the header on the wire --------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["complete", "stream"])
async def test_every_inference_call_sends_the_token(wire, path):
    client = InferenceClient(base_url="http://mgr", model="m", token=TOKEN)
    await _drive_inference(client, path)
    assert wire, "no request reached the transport"
    assert [_auth(r) for r in wire] == [f"Bearer {TOKEN}"] * len(wire)


@pytest.mark.asyncio
async def test_a_503_retry_sends_the_token_again(monkeypatch):
    """The retry is a second request; it needs the header as much as the first."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_COMPLETION)

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real(
        *a, **{**kw, "transport": httpx.MockTransport(handler)}))
    client = InferenceClient(base_url="http://mgr", model="m", token=TOKEN)
    await _drive_inference(client, "complete")
    assert len(seen) == 2
    assert [_auth(r) for r in seen] == [f"Bearer {TOKEN}"] * 2


@pytest.mark.asyncio
async def test_every_retrieval_call_sends_the_token(wire):
    client = RetrievalClient(base_url="http://mgr", collection_id="vault", token=TOKEN)
    await _drive_retrieval(client)
    assert len(wire) == 5, [r.url.path for r in wire]
    assert [_auth(r) for r in wire] == [f"Bearer {TOKEN}"] * 5


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["complete", "stream", "retrieval"])
async def test_no_token_sends_no_authorization_header(wire, kind):
    if kind == "retrieval":
        await _drive_retrieval(RetrievalClient(base_url="http://mgr", collection_id="vault"))
    else:
        await _drive_inference(InferenceClient(base_url="http://mgr", model="m"), kind)
    assert wire
    assert [_auth(r) for r in wire] == [None] * len(wire)


def test_the_inference_clients_token_is_readable_and_read_only():
    client = InferenceClient(base_url="http://mgr", model="m", token=TOKEN)
    assert client.token == TOKEN
    assert InferenceClient(base_url="http://mgr", model="m").token is None
    with pytest.raises(AttributeError):
        client.token = "other"


# --- configuration -------------------------------------------------------------

@pytest.mark.parametrize("agent", ["pal", "pare"])
def test_the_token_is_read_from_the_agents_env(monkeypatch, agent):
    monkeypatch.setenv(f"{agent.upper()}_MANAGER_TOKEN", TOKEN)
    other = "PARE" if agent == "pal" else "PAL"
    monkeypatch.setenv(f"{other}_MANAGER_TOKEN", "someone-elses")
    monkeypatch.delenv("MANAGER_TOKEN", raising=False)
    assert load_config(BaseConfig, agent_name=agent).manager_token == TOKEN


def test_no_token_env_means_no_token(monkeypatch):
    monkeypatch.delenv("PAL_MANAGER_TOKEN", raising=False)
    monkeypatch.setenv("MANAGER_TOKEN", TOKEN)  # the manager's own name is not the agent's
    assert BaseConfig().manager_token is None
    assert load_config(BaseConfig, agent_name="pal").manager_token is None


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_an_empty_token_env_means_no_token(monkeypatch, raw):
    """An empty value would otherwise send `Authorization: Bearer ` and be refused."""
    monkeypatch.setenv("PAL_MANAGER_TOKEN", raw)
    assert load_config(BaseConfig, agent_name="pal").manager_token is None


def test_the_token_is_not_in_the_config_repr(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class AgentConfig(BaseConfig):  # the way PAL and PARE subclass it
        extra: int = 1

    monkeypatch.setenv("PAL_MANAGER_TOKEN", TOKEN)
    for cls in (BaseConfig, AgentConfig):
        cfg = load_config(cls, agent_name="pal")
        assert cfg.manager_token == TOKEN
        assert TOKEN not in repr(cfg)
        assert TOKEN not in str(cfg)


def test_the_default_inference_urls_are_localhost():
    cfg = BaseConfig()
    assert cfg.inference_url == "http://127.0.0.1:11434"
    assert cfg.batch_inference_url == "http://127.0.0.1:11434"


# --- the runtime wires it ----------------------------------------------------------

class _TokenAgent(Agent):
    name = "tokenprobe"

    def setup(self):
        pass


def test_the_runtime_passes_the_token_to_both_clients(monkeypatch, tmp_path, wire):
    monkeypatch.setenv("TOKENPROBE_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("TOKENPROBE_SOCKET_PATH", str(tmp_path / "tokenprobe.sock"))
    monkeypatch.setenv("TOKENPROBE_INFERENCE_URL", "http://mgr:11434")
    monkeypatch.setenv("TOKENPROBE_SEARXNG_URL", "http://searx:8888")
    monkeypatch.setenv("TOKENPROBE_MANAGER_TOKEN", TOKEN)

    agent = _TokenAgent()
    # Stand in for asyncio.run: close the daemon coroutine instead of serving it.
    with patch("agent_core.runtime.asyncio.run", side_effect=lambda coro: coro.close()):
        run_daemon(agent)

    async def calls():
        await _drive_inference(agent.inference, "complete")
        await _drive_retrieval(agent.retrieval)
        await agent.websearch.search("q")

    asyncio.run(calls())

    by_host: dict[str, list[str | None]] = {}
    for r in wire:
        by_host.setdefault(r.url.host, []).append(_auth(r))
    assert set(by_host) == {"mgr", "searx"}, by_host
    assert by_host["mgr"] == [f"Bearer {TOKEN}"] * 6
    # The token is the manager's; SearXNG never sees it.
    assert by_host["searx"] == [None]
