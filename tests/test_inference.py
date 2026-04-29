"""Tests for the inference server HTTP client."""
import json

import httpx
import pytest

from agent_core.inference import InferenceClient, CompletionResult, StreamEnd, ToolCall


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
