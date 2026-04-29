# tests/test_learning_scanner_prefilter.py
import pytest

from agent_core.learning_scanner import has_signal


@pytest.mark.parametrize("msg", [
    "actually, you got that wrong",
    "No, don't do that",
    "stop, I meant the other one",
    "you always try to merge these",
    "you never cite sources",
    "you should use DOMPurify",
    "you shouldn't rely on auto-escape",
    "exactly, that's what I meant",
    "perfect, keep doing that",
    "thank you for the correction",
    "you're right about that",
    "that's wrong",
    "you tend to over-consolidate",
])
def test_has_signal_matches(msg: str):
    assert has_signal(msg) is True


@pytest.mark.parametrize("msg", [
    "tell me about IoT security",
    "what does OpenOCD do?",
    "Can we research compilers next?",
    "cool",
    "ok",
])
def test_has_signal_ignores_neutral(msg: str):
    assert has_signal(msg) is False
