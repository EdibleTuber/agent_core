# Changelog

## [0.3.1] - 2026-04-29

### Added
- `InferenceClient.complete()` and `InferenceClient.stream()` accept an optional `max_tokens: int | None` parameter that flows into the request payload when set.
- New `agent_core.inference.StreamEnd` dataclass yielded as the final item by `InferenceClient.stream()` on the text-output path. Carries `finish_reason` (one of "stop", "length", "tool_calls", "content_filter", "unknown") and `chunks_yielded`.

### Notes
- Tool-call streams continue to yield the assembled `list[ToolCall]` as their final item; no `StreamEnd` follows tool calls. Existing consumers that break on `isinstance(item, list)` are unaffected.
- Consumers that previously joined every yielded item (e.g. `"".join(items)`) must now filter out `StreamEnd` from the iteration.
