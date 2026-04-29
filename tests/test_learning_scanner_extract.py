import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_core.learning_scanner import extract_candidate


def test_extract_candidate_parses_json():
    stub = AsyncMock(return_value='{"title": "T", "body": "B"}')
    result = asyncio.run(extract_candidate(
        recent_turns=[{"role": "user", "content": "you always merge"}],
        trigger_message="you always merge",
        inference_call=stub,
    ))
    assert result == {"title": "T", "body": "B"}


def test_extract_candidate_returns_none_on_null():
    stub = AsyncMock(return_value="null")
    result = asyncio.run(extract_candidate(
        recent_turns=[],
        trigger_message="actually never mind",
        inference_call=stub,
    ))
    assert result is None


def test_extract_candidate_returns_none_on_malformed_json():
    stub = AsyncMock(return_value="this is not json at all")
    result = asyncio.run(extract_candidate(
        recent_turns=[],
        trigger_message="you always",
        inference_call=stub,
    ))
    assert result is None


def test_extract_candidate_returns_none_on_timeout():
    async def slow(prompt):
        await asyncio.sleep(30)
        return "{}"
    result = asyncio.run(extract_candidate(
        recent_turns=[],
        trigger_message="x",
        inference_call=slow,
        timeout=0.1,
    ))
    assert result is None


def test_extract_candidate_returns_none_on_missing_fields():
    stub = AsyncMock(return_value='{"title": "only-title"}')
    result = asyncio.run(extract_candidate(
        recent_turns=[],
        trigger_message="you always",
        inference_call=stub,
    ))
    assert result is None


def test_extract_candidate_returns_none_on_empty_fields():
    stub = AsyncMock(return_value='{"title": "", "body": ""}')
    result = asyncio.run(extract_candidate(
        recent_turns=[],
        trigger_message="you always",
        inference_call=stub,
    ))
    assert result is None


def test_extract_candidate_strips_whitespace():
    stub = AsyncMock(return_value='  {"title": "T", "body": "B"}  ')
    result = asyncio.run(extract_candidate(
        recent_turns=[],
        trigger_message="you always",
        inference_call=stub,
    ))
    assert result == {"title": "T", "body": "B"}
