"""HTTP client for the inference server's OpenAI-compatible API.

Supports both streaming (SSE) and non-streaming completions via
POST /v1/chat/completions. Tool-aware: can pass tool definitions
and parse tool-call responses.
"""
import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import httpx

from agent_core.reasoning import shape_request, extract_reasoning

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_INITIAL_BACKOFF = 2.0
_MAX_BACKOFF = 30.0


class BatchUnavailableError(RuntimeError):
    """Raised when a batch-mode InferenceClient cannot reach the batch
    backend (connection error, repeated 503, or timeout past retries).

    Callers distinguish this from other RuntimeErrors to decide between
    silent-skip (background scanners) or user-facing fallback proposals
    (interactive callers).
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class CompletionResult:
    type: str  # "text" or "tool_calls"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning: str | None = None


@dataclass
class StreamEnd:
    """Sentinel yielded as the final item by `InferenceClient.stream()` after the
    SSE stream completes (text-output path only). Not yielded when the model
    emitted tool calls; the tool-call list itself signals end-of-stream there.
    """
    finish_reason: str   # "stop" | "length" | "tool_calls" | "content_filter" | "unknown"
    chunks_yielded: int


class InferenceClient:
    def __init__(self, base_url: str, model: str, is_batch: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.is_batch = is_batch
        self._client = httpx.AsyncClient(timeout=600.0)

    async def close(self) -> None:
        await self._client.aclose()

    @asynccontextmanager
    async def _stream_with_retry(
        self, url: str, payload: dict
    ) -> AsyncGenerator[httpx.Response, None]:
        """Open a streaming POST, retrying on 503 before yielding.

        For batch clients, any unrecoverable failure (connection error,
        repeated 503, timeout) is re-raised as BatchUnavailableError so
        callers can distinguish batch-backend outages from other errors.
        """
        backoff = _INITIAL_BACKOFF
        try:
            for attempt in range(_MAX_RETRIES):
                async with self._client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 503:
                        resp.raise_for_status()
                        yield resp
                        return
                    retry_after = resp.headers.get("Retry-After")
                wait = min(float(retry_after) if retry_after else backoff, _MAX_BACKOFF)
                logger.warning(
                    "503 from inference server on stream (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, _MAX_BACKOFF)
            # Final attempt
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                yield resp
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if self.is_batch:
                raise BatchUnavailableError(f"{type(exc).__name__}: {exc}") from exc
            raise

    async def _post_with_retry(self, payload: dict) -> httpx.Response:
        """POST to /v1/chat/completions with exponential backoff on 503.

        For batch clients, any unrecoverable failure (connection error,
        repeated 503, timeout) is re-raised as BatchUnavailableError so
        callers can distinguish batch-backend outages from other errors.
        """
        url = f"{self.base_url}/v1/chat/completions"
        backoff = _INITIAL_BACKOFF
        try:
            for attempt in range(_MAX_RETRIES):
                resp = await self._client.post(url, json=payload)
                if resp.status_code != 503:
                    resp.raise_for_status()
                    return resp
                retry_after = float(resp.headers.get("Retry-After", backoff))
                wait = min(retry_after, _MAX_BACKOFF)
                logger.warning(
                    "503 from inference server (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, _MAX_BACKOFF)
            # Final attempt - let it raise on any error
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if self.is_batch:
                raise BatchUnavailableError(f"{type(exc).__name__}: {exc}") from exc
            raise

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        reasoning: Literal["on", "off"] | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Send a non-streaming completion request.

        Returns a CompletionResult indicating either a text response
        or a list of tool calls the model wants to make.
        """
        resolved_model = model or self.default_model
        payload: dict = {"model": resolved_model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if reasoning is not None:
            payload = shape_request(payload, resolved_model, reasoning)
            if reasoning == "on" and "chat_template_kwargs" not in payload:
                logger.debug("reasoning control requested but no-op for model %s", resolved_model)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = await self._post_with_retry(payload)
        data = resp.json()
        message = data["choices"][0]["message"]

        raw_calls = message.get("tool_calls")
        if raw_calls:
            parsed = []
            for tc in raw_calls:
                func = tc["function"]
                args = func["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                parsed.append(ToolCall(
                    id=tc["id"],
                    name=func["name"],
                    arguments=args,
                ))
            return CompletionResult(type="tool_calls", tool_calls=parsed)

        reasoning_text = extract_reasoning(data)
        return CompletionResult(
            type="text",
            content=message.get("content", ""),
            reasoning=reasoning_text,
        )

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        reasoning: Literal["on", "off"] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str | list[ToolCall] | StreamEnd, None]:
        """Send a streaming completion request.

        Yields str tokens for text responses. If the model returns tool calls
        instead, accumulates all tool-call deltas and yields a single
        list[ToolCall] as the only item.

        For text responses, after the SSE stream ends, yields a final
        `StreamEnd(finish_reason, chunks_yielded)` sentinel. Tool-call
        responses do not get a trailing StreamEnd; the tool-call list itself
        signals end-of-stream.
        """
        resolved_model = model or self.default_model
        payload: dict = {"model": resolved_model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if reasoning is not None:
            payload = shape_request(payload, resolved_model, reasoning)
            if reasoning == "on" and "chat_template_kwargs" not in payload:
                logger.debug("reasoning control requested but no-op for model %s", resolved_model)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Accumulators for tool-call deltas
        tool_call_acc: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        is_tool_response = False
        chunks_yielded = 0
        finish_reason = "unknown"
        url = f"{self.base_url}/v1/chat/completions"

        async with self._stream_with_retry(url, payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choice = chunk["choices"][0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})

                # Check for tool call deltas
                tc_deltas = delta.get("tool_calls")
                if tc_deltas is not None:
                    is_tool_response = True
                    for tcd in tc_deltas:
                        idx = tcd.get("index", 0)
                        if idx not in tool_call_acc:
                            tool_call_acc[idx] = {
                                "id": tcd.get("id", ""),
                                "name": "",
                                "arguments_str": "",
                            }
                        acc = tool_call_acc[idx]
                        if tcd.get("id"):
                            acc["id"] = tcd["id"]
                        func = tcd.get("function", {})
                        if func.get("name"):
                            acc["name"] = func["name"]
                        if func.get("arguments"):
                            acc["arguments_str"] += func["arguments"]
                    continue

                # Regular text content
                content = delta.get("content")
                if content is not None:
                    chunks_yielded += 1
                    yield content

        # If we accumulated tool calls, yield them as a single list
        if is_tool_response and tool_call_acc:
            calls = []
            for idx in sorted(tool_call_acc):
                acc = tool_call_acc[idx]
                args = json.loads(acc["arguments_str"]) if acc["arguments_str"] else {}
                calls.append(ToolCall(
                    id=acc["id"],
                    name=acc["name"],
                    arguments=args,
                ))
            yield calls
        else:
            # Text-output path: emit a sentinel describing how the stream ended.
            yield StreamEnd(finish_reason=finish_reason, chunks_yielded=chunks_yielded)
