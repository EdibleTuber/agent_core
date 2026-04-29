# Changelog

## [0.4.0] - 2026-04-29

### Added
- `agent_core.protocol` package: `transport` (encode_message/decode_message/register_message/STREAM_BUFFER_LIMIT) and `messages` (ChatMessage, CommandMessage, StreamChunkMessage, ResponseMessage, ErrorMessage, ToolProgressMessage, LearningCandidateProposalMessage). Self-registering registry for downstream protocols.
- `agent_core.conversation.Conversation`: rolling in-memory message buffer with optional JSONL persistence and a generic `overrides: dict[str, Any]` field for per-conversation toggles. Replaces PAL's `reasoning_override` field with a forward-compatible dict.
- `agent_core.channels.ChannelStore`: per-channel Conversation cache, vault-rooted at `<vault>/_channels/<agent_name>/`.
- `agent_core.scratchpad.Scratchpad`: free-form per-channel markdown file with optional `commit_callback: Callable[[Path, str], None]` for git tracking. Replaces PAL's WikiManager dependency.
- `agent_core.git_helpers.make_commit_callback`: helper factory for agents that want bare git tracking on scratchpad writes. Locale-independent diff check, GPG-sign bypass, dash-prefixed path safety.
- `agent_core.learning_scanner`: signal detection, extraction, dedupe, and proposal emission pipeline. Consumes `LearningCandidateProposalMessage` from `agent_core.protocol`.

### Notes
- All Phase D modules use the per-agent vault layout: `<vault>/_<thing>/<agent_name>/...`. Consumers pass `agent_name` at construction.
- The Conversation field rename from `reasoning_override` to `overrides` is a breaking change for any caller reading/writing that field directly. PAL's call sites are updated in its Phase D consumer-side migration.

## [0.3.1] - 2026-04-29

### Added
- `InferenceClient.complete()` and `InferenceClient.stream()` accept an optional `max_tokens: int | None` parameter that flows into the request payload when set.
- New `agent_core.inference.StreamEnd` dataclass yielded as the final item by `InferenceClient.stream()` on the text-output path. Carries `finish_reason` (one of "stop", "length", "tool_calls", "content_filter", "unknown") and `chunks_yielded`.

### Notes
- Tool-call streams continue to yield the assembled `list[ToolCall]` as their final item; no `StreamEnd` follows tool calls. Existing consumers that break on `isinstance(item, list)` are unaffected.
- Consumers that previously joined every yielded item (e.g. `"".join(items)`) must now filter out `StreamEnd` from the iteration.
