# Changelog

## [0.5.1] - 2026-04-30

### Added
- `Agent.handle_other(msg, ctx)` — synchronous default-noop hook for messages that aren't ChatMessage or CommandMessage. Lets agents route domain-specific message types (approval responses, batch fallback choices) without subclassing the daemon.
- `Daemon._handle_connection` dispatches non-Chat/non-Command messages to `agent.handle_other` instead of emitting an "unexpected message type" error.

### Notes
- This unblocks PAL's Phase E migration: PAL's CLI sends `ResearchApprovalResponseMessage` and `BatchFallbackApprovalMessage` directly (not via Chat/Command), and these need agent-side routing.
- Default `handle_other` is a no-op so existing minimal agents don't break.

## [0.5.0] - 2026-04-30

### Added
- `agent_core.config.BaseConfig`: dataclass with universally-shared agent fields, plus `load_config()` with name-derived env-var prefix machinery (e.g. agent name "pal" reads `PAL_*` env vars; "re-lab" reads `RELAB_*`). Subclasses extend with domain fields. Coercion errors annotate the offending env-var name.
- `agent_core.agent.Agent`: base class with extension points `setup()`, `system_prompt()`, `handle_chat()`, `handle_command()`, `decide_mode()`. Framework managers (profile, wisdom, learning, allowlist, approval_registry, channels, inference, retrieval, websearch) are populated by `run_daemon()` before `setup()` runs.
- `agent_core.agent.HandlerContext`: per-turn dataclass carrying conversation, channel_id, writer.
- `agent_core.client.DaemonConnection`: async unix-socket client (lifted from PAL with the asyncio.Lock for concurrent-read safety, `is_connected` property, and `chat()`/`command()`/`command_stream()` end-of-turn helpers).
- `agent_core.daemon.Daemon`: transport-only daemon. Connection lifecycle, NDJSON decode, dispatch to agent handlers, NDJSON encode, disconnect cleanup. Reserves `_chat_tasks` for the deferred per-channel preemption safety fix.
- `agent_core.runtime.run_daemon()`: entry point that wires framework managers onto the agent and starts the daemon.
- `agent_core.adapters.cli.run_repl()`: generic REPL with a `Renderer` Protocol for agent-specific message rendering. Default rendering covers the seven generic message types.
- Contract tests in `tests/test_contract.py` pinning the API surface.

### Notes
- This release is the first that lets a new agent ship as just a small `Agent` subclass plus a `run_daemon(MyAgent())` entry point. PAL adopts in PAL's Phase E migration.
- Phase F (next) extracts tool/command/prompt scaffolding so default `handle_chat` / `handle_command` implementations can use registries instead of every agent reimplementing dispatch.

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
