"""Tests for LearningScanner silent-skip on batch unavailable."""
import logging

import pytest
from unittest.mock import AsyncMock

from agent_core.inference import BatchUnavailableError
from agent_core.learning_scanner import extract_candidate


@pytest.mark.asyncio
async def test_extract_candidate_silent_skip_on_batch_unavailable(caplog):
    """When the inference call raises BatchUnavailableError, extract_candidate
    logs a warning and returns None, no exception propagation.
    """
    inference_call = AsyncMock(side_effect=BatchUnavailableError("batch down"))

    with caplog.at_level(logging.WARNING):
        result = await extract_candidate(
            recent_turns=[{"role": "user", "content": "hi"}],
            trigger_message="hi",
            inference_call=inference_call,
        )

    assert result is None
    assert any(
        "batch unavailable" in r.getMessage().lower()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_extract_candidate_propagates_other_exceptions():
    """Non-batch, non-timeout exceptions should propagate, not be swallowed."""
    inference_call = AsyncMock(side_effect=RuntimeError("some other error"))

    with pytest.raises(RuntimeError, match="some other error"):
        await extract_candidate(
            recent_turns=[{"role": "user", "content": "hi"}],
            trigger_message="hi",
            inference_call=inference_call,
        )
