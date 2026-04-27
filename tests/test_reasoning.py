# tests/test_reasoning.py
"""Tests for the reasoning module."""
from dataclasses import dataclass
from typing import Literal

from agent_core.reasoning import shape_request, extract_reasoning, decide_mode, _identify_family


@dataclass
class _StubConversation:
    """Minimal stand-in for an agent's Conversation type.

    Matches the duck-typed _ConversationLike Protocol that decide_mode reads.
    """
    reasoning_override: Literal["on", "off"] | None = None


def test_identify_family_gemma4():
    assert _identify_family("gemma-4-26b-a4b-it-q4_k_m") == "gemma"


def test_identify_family_gemma3():
    assert _identify_family("gemma-3-27b-it-q4_k_m") == "gemma"


def test_identify_family_qwen3():
    assert _identify_family("qwen3-35b-a3b-q4_k_m") == "qwen3"


def test_identify_family_unknown():
    assert _identify_family("llama-3.1-8b") is None


def test_shape_request_gemma_on():
    body = {"model": "gemma-4-26b", "messages": []}
    result = shape_request(body, "gemma-4-26b-a4b-it-q4_k_m", "on")
    assert result["chat_template_kwargs"]["enable_thinking"] is True


def test_shape_request_gemma_off():
    body = {"model": "gemma-4-26b", "messages": []}
    result = shape_request(body, "gemma-4-26b-a4b-it-q4_k_m", "off")
    assert result["chat_template_kwargs"]["enable_thinking"] is False


def test_shape_request_preserves_existing_kwargs():
    body = {"model": "gemma-4-26b", "messages": [], "chat_template_kwargs": {"other": 42}}
    result = shape_request(body, "gemma-4-26b-a4b-it-q4_k_m", "on")
    assert result["chat_template_kwargs"]["other"] == 42
    assert result["chat_template_kwargs"]["enable_thinking"] is True


def test_shape_request_unknown_model_noop():
    body = {"model": "llama-3.1-8b", "messages": []}
    original = dict(body)
    result = shape_request(body, "llama-3.1-8b", "on")
    assert result == original


def test_shape_request_qwen3_noop_for_now():
    body = {"model": "qwen3-35b", "messages": []}
    original = dict(body)
    result = shape_request(body, "qwen3-35b-a3b-q4_k_m", "on")
    assert result == original


def test_extract_reasoning_present():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think about this step by step...",
            }
        }]
    }
    assert extract_reasoning(response) == "Let me think about this step by step..."


def test_extract_reasoning_absent():
    response = {
        "choices": [{
            "message": {"role": "assistant", "content": "hello"}
        }]
    }
    assert extract_reasoning(response) is None


def test_extract_reasoning_empty_string():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "hello",
                "reasoning_content": "",
            }
        }]
    }
    assert extract_reasoning(response) is None


def test_decide_mode_override_on():
    conv = _StubConversation(reasoning_override="on")
    assert decide_mode(conv) == "on"


def test_decide_mode_override_off():
    conv = _StubConversation(reasoning_override="off")
    assert decide_mode(conv) == "off"


def test_decide_mode_no_override():
    conv = _StubConversation()
    assert decide_mode(conv) == "off"


def test_shape_request_does_not_mutate_input():
    body = {"model": "gemma-4-26b", "messages": []}
    shape_request(body, "gemma-4-26b-a4b-it-q4_k_m", "on")
    assert "chat_template_kwargs" not in body
