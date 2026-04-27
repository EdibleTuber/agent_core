# agent_core/reasoning.py
"""Reasoning model control -- per-request toggle and response extraction.

Maps model names to families and dispatches reasoning control per family.
Today: Gemma family uses chat_template_kwargs.enable_thinking.
"""
from __future__ import annotations

from typing import Literal, Protocol


class _ConversationLike(Protocol):
    """Duck-typed contract for what `decide_mode` reads from its argument.

    Any object with an optional `reasoning_override` attribute set to
    "on", "off", or None will satisfy this. Concrete agents (e.g. PAL's
    Conversation class) match without explicit subclassing.
    """
    reasoning_override: Literal["on", "off"] | None


_MODEL_FAMILIES: dict[str, str] = {
    "gemma-4": "gemma",
    "gemma-3": "gemma",
    "qwen3":   "qwen3",
}


def _identify_family(model: str) -> str | None:
    for prefix, family in _MODEL_FAMILIES.items():
        if model.startswith(prefix):
            return family
    return None


def shape_request(body: dict, model: str, mode: Literal["on", "off"]) -> dict:
    body = dict(body)
    if "chat_template_kwargs" in body:
        body["chat_template_kwargs"] = dict(body["chat_template_kwargs"])
    match _identify_family(model):
        case "gemma":
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = (mode == "on")
        case "qwen3":
            pass
        case None:
            pass
    return body


def extract_reasoning(response: dict) -> str | None:
    msg = response["choices"][0]["message"]
    return msg.get("reasoning_content") or None


def decide_mode(conversation: _ConversationLike) -> Literal["on", "off"]:
    override = getattr(conversation, "reasoning_override", None)
    if override in ("on", "off"):
        return override
    return "off"
