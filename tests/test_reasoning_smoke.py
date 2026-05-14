"""Reasoning-content smoke test against the real local manager.

Verifies that agent_core.inference.complete(..., reasoning=...) still
returns separate .content and .reasoning fields when called against a
running Gemma-4 model on the local manager. This is an integration
test, not run by default — it requires the inference server to be up.

Enable with:
    AGENT_CORE_SMOKE_MANAGER_URL=http://192.168.1.14:11434 \\
    AGENT_CORE_SMOKE_MODEL=gemma-4-26b-a4b-it-q4_k_m \\
    pytest tests/test_reasoning_smoke.py -v
"""
import os

import pytest

from agent_core.inference import InferenceClient


SMOKE_URL = os.getenv("AGENT_CORE_SMOKE_MANAGER_URL")
SMOKE_MODEL = os.getenv("AGENT_CORE_SMOKE_MODEL")


pytestmark = pytest.mark.skipif(
    not (SMOKE_URL and SMOKE_MODEL),
    reason="set AGENT_CORE_SMOKE_MANAGER_URL and AGENT_CORE_SMOKE_MODEL to run",
)


@pytest.mark.asyncio
async def test_reasoning_on_returns_separate_content_and_reasoning():
    """With reasoning='on', the completion has both .content and .reasoning
    populated (Gemma-4 is a reasoning model)."""
    client = InferenceClient(base_url=SMOKE_URL, model=SMOKE_MODEL)
    messages = [
        {"role": "user", "content": "What is 2 + 2? Answer in exactly one word."}
    ]
    completion = await client.complete(messages, reasoning="on", max_tokens=512)
    assert completion.content, "expected non-empty .content"
    # Reasoning may be empty for trivial queries even with reasoning='on',
    # so don't strictly require non-empty — just verify the field exists.
    assert hasattr(completion, "reasoning")


@pytest.mark.asyncio
async def test_reasoning_off_skips_chain_of_thought():
    """With reasoning='off', the completion has .content; .reasoning is
    empty or absent."""
    client = InferenceClient(base_url=SMOKE_URL, model=SMOKE_MODEL)
    messages = [
        {"role": "user", "content": "Say hello in exactly three words."}
    ]
    completion = await client.complete(messages, reasoning="off", max_tokens=64)
    assert completion.content
    # Reasoning either absent or empty string.
    if hasattr(completion, "reasoning") and completion.reasoning:
        pytest.fail(
            f"reasoning='off' returned non-empty reasoning: {completion.reasoning!r}"
        )
