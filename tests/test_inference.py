"""Tests for the inference server HTTP client."""
import json

import httpx
import pytest

from agent_core.inference import InferenceClient, CompletionResult, StreamEnd, ToolCall, Usage


# Minimal tool definitions for tests. Mirrors the shape that PAL's full
# TOOL_DEFINITIONS uses but only includes one entry, since the inference
# tests only verify dispatch behavior, not which tools exist.
_TEST_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the vault.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


@pytest.mark.asyncio
async def test_complete_non_streaming(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "hello world"}],
    )
    assert result.type == "text"
    assert result.content == "echo: hello world"


@pytest.mark.asyncio
async def test_complete_streaming(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    tokens = []
    async for token in client.stream(
        messages=[{"role": "user", "content": "hello world"}],
    ):
        if isinstance(token, StreamEnd):
            continue
        tokens.append(token)
    full = "".join(tokens)
    assert full == "echo: hello world"


@pytest.mark.asyncio
async def test_complete_streaming_empty_response(mock_inference_server):
    """Streaming an empty user message still produces output."""
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    tokens = []
    async for token in client.stream(
        messages=[{"role": "user", "content": ""}],
    ):
        if isinstance(token, StreamEnd):
            continue
        tokens.append(token)
    full = "".join(tokens)
    assert full == "echo:"


@pytest.mark.asyncio
async def test_complete_returns_text_result(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "hello"}],
        tools=_TEST_TOOL_DEFINITIONS,
    )
    assert isinstance(result, CompletionResult)
    assert result.type == "text"
    assert result.content == "echo: hello"
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_complete_returns_tool_calls(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "TOOLCALL:read_file"}],
        tools=_TEST_TOOL_DEFINITIONS,
    )
    assert isinstance(result, CompletionResult)
    assert result.type == "tool_calls"
    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "Research/quantum.md"}


@pytest.mark.asyncio
async def test_complete_without_tools_returns_text(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert isinstance(result, CompletionResult)
    assert result.type == "text"
    assert result.content == "echo: hello"


@pytest.mark.asyncio
async def test_stream_returns_text_tokens(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    items = []
    async for item in client.stream(
        messages=[{"role": "user", "content": "hello"}],
        tools=_TEST_TOOL_DEFINITIONS,
    ):
        items.append(item)
    # Final item is a StreamEnd sentinel; preceding items are text tokens.
    assert isinstance(items[-1], StreamEnd)
    text_items = items[:-1]
    assert all(isinstance(item, str) for item in text_items)
    assert "".join(text_items) == "echo: hello"


@pytest.mark.asyncio
async def test_stream_detects_tool_calls(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    items = []
    async for item in client.stream(
        messages=[{"role": "user", "content": "TOOLCALL:read_file"}],
        tools=_TEST_TOOL_DEFINITIONS,
    ):
        items.append(item)
    assert len(items) == 1
    assert isinstance(items[0], list)
    assert len(items[0]) == 1
    assert items[0][0].name == "read_file"
    assert items[0][0].arguments == {"path": "Research/quantum.md"}


@pytest.mark.asyncio
async def test_complete_extracts_reasoning(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "REASON:deep question"}],
    )
    assert result.type == "text"
    assert result.content == "answer: deep question"
    assert result.reasoning == "thinking about deep question"


@pytest.mark.asyncio
async def test_complete_no_reasoning_returns_none(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert result.reasoning is None


@pytest.mark.asyncio
async def test_complete_uses_override_model(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="default-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "hello"}],
        model="override-model",
    )
    assert result.type == "text"


@pytest.mark.asyncio
async def test_default_model_attribute(mock_inference_server):
    client = InferenceClient(base_url=mock_inference_server, model="my-model")
    assert client.default_model == "my-model"


@pytest.mark.asyncio
async def test_complete_includes_max_tokens_when_set():
    """When max_tokens is passed, it appears in the request payload."""
    captured: dict = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    await client.complete(messages=[{"role": "user", "content": "hi"}], max_tokens=512)

    assert captured["body"].get("max_tokens") == 512


@pytest.mark.asyncio
async def test_complete_omits_max_tokens_when_none():
    """When max_tokens is None (default), no max_tokens key is in the payload."""
    captured: dict = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    await client.complete(messages=[{"role": "user", "content": "hi"}])

    assert "max_tokens" not in captured["body"]


def _sse_response(chunks: list[dict]) -> httpx.Response:
    """Build a streaming Response that emits the given chunk objects as SSE lines."""
    body_lines = []
    for c in chunks:
        body_lines.append(f"data: {json.dumps(c)}")
    body_lines.append("data: [DONE]")
    body = "\n".join(body_lines).encode("utf-8")
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_stream_includes_max_tokens_when_set():
    captured: dict = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _sse_response([
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

    transport = httpx.MockTransport(mock_handler)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    out = []
    async for item in client.stream(messages=[{"role": "user", "content": "hi"}], max_tokens=256):
        out.append(item)

    assert captured["body"].get("max_tokens") == 256


@pytest.mark.asyncio
async def test_stream_yields_streamend_with_stop_reason():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return _sse_response([
            {"choices": [{"delta": {"content": "alpha"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "beta"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

    transport = httpx.MockTransport(mock_handler)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    out = []
    async for item in client.stream(messages=[{"role": "user", "content": "hi"}]):
        out.append(item)

    assert out[:-1] == ["alpha", "beta"]
    assert isinstance(out[-1], StreamEnd)
    assert out[-1].finish_reason == "stop"
    assert out[-1].chunks_yielded == 2


@pytest.mark.asyncio
async def test_stream_yields_streamend_with_length_reason_when_capped():
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return _sse_response([
            {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "y"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ])

    transport = httpx.MockTransport(mock_handler)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    out = []
    async for item in client.stream(messages=[{"role": "user", "content": "hi"}], max_tokens=2):
        out.append(item)

    end = out[-1]
    assert isinstance(end, StreamEnd)
    assert end.finish_reason == "length"
    assert end.chunks_yielded == 2


@pytest.mark.asyncio
async def test_stream_tool_calls_does_not_yield_streamend():
    """When the model emits tool_calls, no StreamEnd follows."""
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return _sse_response([
            {"choices": [{"delta": {
                "tool_calls": [{
                    "index": 0, "id": "tc1",
                    "function": {"name": "search", "arguments": "{\"q\":\"x\"}"},
                }]
            }, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])

    transport = httpx.MockTransport(mock_handler)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    out = []
    async for item in client.stream(messages=[{"role": "user", "content": "hi"}]):
        out.append(item)

    # Last item is the tool-calls list, not a StreamEnd.
    assert isinstance(out[-1], list)
    assert all(not isinstance(x, StreamEnd) for x in out)


# ---------------------------------------------------------------------------
# Usage tracking (per-turn token counts).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_attaches_usage(mock_inference_server):
    """Non-streaming complete() returns Usage from the response's usage block."""
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert isinstance(result.usage, Usage)
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 50
    assert result.usage.model == "test-model"


@pytest.mark.asyncio
async def test_complete_tool_call_attaches_usage(mock_inference_server):
    """Tool-call responses also carry usage."""
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    result = await client.complete(
        messages=[{"role": "user", "content": "TOOLCALL:cat"}],
        tools=_TEST_TOOL_DEFINITIONS,
    )
    assert result.type == "tool_calls"
    assert isinstance(result.usage, Usage)
    assert result.usage.total_tokens == 50


@pytest.mark.asyncio
async def test_stream_request_includes_stream_options(mock_inference_server):
    """stream() opts the request into include_usage so the server emits usage."""
    from tests.conftest import REQUEST_LOG
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    async for _ in client.stream(messages=[{"role": "user", "content": "hi"}]):
        pass
    assert REQUEST_LOG[-1].get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_attaches_usage_to_streamend(mock_inference_server):
    """The StreamEnd sentinel carries Usage when the server emits a usage chunk."""
    client = InferenceClient(base_url=mock_inference_server, model="test-model")
    end = None
    async for item in client.stream(messages=[{"role": "user", "content": "hello"}]):
        if isinstance(item, StreamEnd):
            end = item
    assert end is not None
    assert isinstance(end.usage, Usage)
    assert end.usage.prompt_tokens == 42
    assert end.usage.completion_tokens == 8
    assert end.usage.total_tokens == 50


@pytest.mark.asyncio
async def test_complete_handles_missing_usage():
    """When the server omits usage, result.usage is None (not an error)."""
    transport = httpx.MockTransport(lambda req: httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    ))
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    result = await client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.type == "text"
    assert result.usage is None


@pytest.mark.asyncio
async def test_stream_handles_missing_usage():
    """When the server omits the usage chunk, StreamEnd.usage is None."""
    async def fake_stream(request):
        body = (
            b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n\n'
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )
    transport = httpx.MockTransport(fake_stream)
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=transport)

    end = None
    async for item in client.stream(messages=[{"role": "user", "content": "hi"}]):
        if isinstance(item, StreamEnd):
            end = item
    assert end is not None
    assert end.usage is None


# ---------------------------------------------------------------------------
# Transient connection-drop resilience.
#
# A model-swapping/keep-alive backend can close a pooled connection between
# turns; the client sees httpx.RemoteProtocolError ("Server disconnected
# without sending a response"). Such drops are transient and must be retried,
# not surfaced as a fatal chat error. Genuine client errors (4xx) must NOT be.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_retries_on_remote_protocol_error(monkeypatch):
    """A dropped connection is retried and the next attempt succeeds."""
    import agent_core.inference as inference

    async def _no_sleep(*_a, **_k):
        return
    monkeypatch.setattr(inference.asyncio, "sleep", _no_sleep)

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.", request=request
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await client.complete(messages=[{"role": "user", "content": "hi"}])

    assert result.type == "text"
    assert result.content == "ok"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_complete_does_not_retry_on_client_error(monkeypatch):
    """A 4xx is a permanent error: raise immediately, do not retry."""
    import agent_core.inference as inference

    async def _no_sleep(*_a, **_k):
        return
    monkeypatch.setattr(inference.asyncio, "sleep", _no_sleep)

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        await client.complete(messages=[{"role": "user", "content": "hi"}])
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_stream_retries_on_remote_protocol_error_before_first_chunk(monkeypatch):
    """A connection drop before any token is retried, then streams normally."""
    import agent_core.inference as inference

    async def _no_sleep(*_a, **_k):
        return
    monkeypatch.setattr(inference.asyncio, "sleep", _no_sleep)

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.", request=request
            )
        return _sse_response([
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    out = []
    async for item in client.stream(messages=[{"role": "user", "content": "hi"}]):
        out.append(item)

    assert attempts["n"] == 2
    assert out[:-1] == ["hi"]
    assert isinstance(out[-1], StreamEnd)


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_first_chunk(monkeypatch):
    """A mid-stream drop (tokens already emitted) must propagate, not retry —
    retrying would replay already-yielded tokens to the caller."""
    import agent_core.inference as inference

    async def _no_sleep(*_a, **_k):
        return
    monkeypatch.setattr(inference.asyncio, "sleep", _no_sleep)

    attempts = {"n": 0}

    class _DropAfterFirst(httpx.AsyncByteStream):
        def __init__(self, request):
            self._request = request

        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n\n'
            raise httpx.RemoteProtocolError("peer closed mid-stream", request=self._request)

        async def aclose(self):
            return

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DropAfterFirst(request),
        )

    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    out = []
    with pytest.raises(httpx.RemoteProtocolError):
        async for item in client.stream(messages=[{"role": "user", "content": "hi"}]):
            out.append(item)

    assert attempts["n"] == 1
    assert out == ["hi"]


# --- 409 model_not_loaded is recoverable, not terminal ----------------------
#
# The model manager fronting several llama.cpp backends refuses implicit model
# swaps: a completion for a model loaded on no slot gets 409 with
# {"error": {"type": "model_not_loaded", "message": "... use POST /swap"}}.
# The remedy is named in the error. This client used to raise instead of taking
# it, so the first turn of a session after a manager restart died -- observed in
# PARE's log as `Chat error: Client error '409 Conflict'`.
#
# 503 ("loaded but slot not ready") was already retried with backoff. The
# asymmetry was backwards: 409 has a deterministic fix, 503 is only hope.

_NOT_LOADED = {"error": {"type": "model_not_loaded",
                         "message": "model m not loaded on any slot; use POST /swap"}}
_OK_COMPLETION = {"choices": [{"message": {"role": "assistant", "content": "ok"},
                               "finish_reason": "stop"}]}


def _client_with(handler):
    client = InferenceClient(base_url="http://test", model="m")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.asyncio
async def test_a_model_not_loaded_409_triggers_a_swap_and_one_retry():
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.url.path == "/swap":
            return httpx.Response(200, json={"slot": "main", "model": "m", "status": "ok"})
        if seen.count("POST /v1/chat/completions") == 1:
            return httpx.Response(409, json=_NOT_LOADED)
        return httpx.Response(200, json=_OK_COMPLETION)

    result = await _client_with(handler).complete(
        messages=[{"role": "user", "content": "hi"}])
    assert result.type == "text" and result.content == "ok"
    assert seen == ["POST /v1/chat/completions", "POST /swap",
                    "POST /v1/chat/completions"], seen


@pytest.mark.asyncio
async def test_the_swap_asks_for_the_model_the_completion_wanted():
    """And omits `target`: the client cannot know which slot a model belongs on,
    so it lets the manager apply its own default rather than guessing and
    evicting whatever is on main."""
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/swap":
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})
        if not bodies:
            return httpx.Response(409, json=_NOT_LOADED)
        return httpx.Response(200, json=_OK_COMPLETION)

    await _client_with(handler).complete(
        messages=[{"role": "user", "content": "hi"}], model="some-other-model")
    assert bodies == [{"model": "some-other-model"}], bodies


@pytest.mark.asyncio
async def test_an_unrelated_409_does_NOT_trigger_a_swap():
    """Keyed on the error body, not the status code. A 409 from some other
    backend for some other reason must not provoke a POST to an admin endpoint."""
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(409, json={"error": {"type": "something_else",
                                                   "message": "nope"}})

    with pytest.raises(httpx.HTTPStatusError):
        await _client_with(handler).complete(messages=[{"role": "user", "content": "hi"}])
    assert "/swap" not in seen, seen


@pytest.mark.asyncio
async def test_a_409_with_an_unparseable_body_does_not_trigger_a_swap():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/swap", "swapped on a body it could not read"
        return httpx.Response(409, content=b"<html>gateway</html>")

    with pytest.raises(httpx.HTTPStatusError):
        await _client_with(handler).complete(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_it_swaps_at_most_once_and_then_gives_up():
    """A swap that reports success while the completion still 409s must not
    loop. One recovery attempt, then the caller hears about it."""
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/swap":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(409, json=_NOT_LOADED)

    with pytest.raises(httpx.HTTPStatusError):
        await _client_with(handler).complete(messages=[{"role": "user", "content": "hi"}])
    assert seen.count("/swap") == 1, seen


@pytest.mark.asyncio
async def test_a_failed_swap_surfaces_the_original_409_not_the_swap_error():
    """The caller needs to know the model was not loaded. A 404 from /swap on a
    backend that has no such endpoint must not replace that diagnosis."""
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/swap":
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(409, json=_NOT_LOADED)

    with pytest.raises(httpx.HTTPStatusError) as e:
        await _client_with(handler).complete(messages=[{"role": "user", "content": "hi"}])
    assert e.value.response.status_code == 409


@pytest.mark.asyncio
async def test_the_streaming_path_recovers_from_model_not_loaded_too():
    """`complete` and `stream` have SEPARATE retry loops, both of which handled
    only 503. Fixing one would have left the bug reachable through the other,
    which is worse than not fixing it -- it looks fixed."""
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/swap":
            return httpx.Response(200, json={"status": "ok"})
        if seen.count("/v1/chat/completions") == 1:
            return httpx.Response(409, json=_NOT_LOADED)
        body = ('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                'data: [DONE]\n\n')
        return httpx.Response(200, content=body.encode(),
                              headers={"Content-Type": "text/event-stream"})

    chunks = [c async for c in _client_with(handler).stream(
        messages=[{"role": "user", "content": "hi"}])]
    assert any(getattr(c, "content", None) == "ok" or c == "ok" for c in chunks), chunks
    assert seen == ["/v1/chat/completions", "/swap", "/v1/chat/completions"], seen


@pytest.mark.asyncio
async def test_the_streaming_path_does_not_swap_on_an_unrelated_409():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/swap", "swapped on an unrelated 409"
        return httpx.Response(409, json={"error": {"type": "other", "message": "no"}})

    with pytest.raises(httpx.HTTPStatusError):
        async for _ in _client_with(handler).stream(
                messages=[{"role": "user", "content": "hi"}]):
            pass
