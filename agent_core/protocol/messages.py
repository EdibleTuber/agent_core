"""Generic agent message primitives. Domain-specific messages are registered by
each agent's own protocol module."""
from dataclasses import dataclass

from agent_core.protocol.transport import register_message


@register_message
@dataclass
class ChatMessage:
    text: str
    channel_id: str | None = None
    type: str = "chat"


@register_message
@dataclass
class CommandMessage:
    name: str
    args: str
    channel_id: str | None = None
    type: str = "command"


@register_message
@dataclass
class StreamChunkMessage:
    token: str
    type: str = "stream_chunk"


@register_message
@dataclass
class ResponseMessage:
    text: str
    command: str = ""
    reasoning: str = ""
    type: str = "response"


@register_message
@dataclass
class ErrorMessage:
    error: str
    type: str = "error"


@register_message
@dataclass
class ToolProgressMessage:
    tool: str
    arguments: dict
    type: str = "tool_progress"


@register_message
@dataclass
class LearningCandidateProposalMessage:
    proposal_id: str
    title: str
    body: str
    trigger_excerpt: str  # user-message fragment that triggered the scan
    type: str = "learning_candidate_proposal"
