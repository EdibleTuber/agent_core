# Changelog

## [1.5.0] - 2026-05-28

### Added
- `agent_core.workers.risk_pool.RiskAwareToolPool` — composes an `MCPClientPool` and enforces `RiskGate` at the `call_tool` chokepoint. `low`/`medium` tiers auto-execute; `high`/`critical` block on operator approval; every dispatch emits one `AuditEntry`. `list_tools`/`close_all` proxy to the inner pool ungated (discovery stays read-only). Approval delivery is fail-closed: if the approval channel raises, the call is denied (`outcome="approval_undeliverable"`) rather than waiting out the timeout.
- `agent_core.workers.tool_approval` — `ToolApprovalRegistry` (per-call HITL gates keyed by `proposal_id`, with idempotent `discard()` cleanup and a timeout that auto-denies), plus `ToolCallSpec` and `ToolDecision` (carries `approved`, `justification`, and an `once`/`session` scope).
- `ToolApprovalRequestMessage` / `ToolApprovalResponseMessage` protocol messages. The CLI renders an inline `[y/n/j/a]` approval prompt (`a` = approve-for-session; `critical` tier forces a justification and disallows bare `y`/`a`; Ctrl-C/EOF denies). The daemon routes responses to the registry at chassis level (independent of any agent's `handle_other`).
- `AuditEntry.detail` field — carries the operator justification, an exception class name, or a session-approval note, keeping `override_reason` reserved for the RiskGate escalation reason.
- `"approval_undeliverable"` member added to the audit `Outcome` literal.

### Notes
- `worker_contract_version` stays at `1`. **Purely additive** — no existing public signatures changed. `make_tool_class` and `discover_and_register` are untouched; an agent opts into enforcement by passing a `RiskAwareToolPool` where it previously passed an `MCPClientPool`.
- `risk_default` in `workers.yaml` becomes live policy when dispatch goes through `RiskAwareToolPool`. `kind: external_mcp` auto-bump and `risk_overrides:` patterns remain unimplemented (tracked follow-ups).
- Audit `outcome` reflects execution truth: an approved call whose inner result errors is audited `"error"`, not `"hitl_approved"`.
- PAL consumers can bump the pin transparently; PAL doesn't use MCP workers in v1.

## [1.4.0] - 2026-05-25

### Added
- `agent_core.workers.MCPClient` now supports stdio transport alongside the existing Streamable HTTP. Constructor accepts either `endpoint` (HTTP, existing positional) or keyword-only `command` / `args` / `env` / `cwd` (stdio, new). `connect()` branches on whichever is configured.
- `MCPClient.from_spec(spec)` classmethod — dispatches to the right transport from a `WorkerSpec`. Used internally by `MCPClientPool`.
- `agent_core.workers.types.WorkerSpec` gains optional `command`, `args`, `env`, `cwd` fields. A `model_validator` enforces "endpoint required for streamable_http/http_job_api, command required for stdio."
- `assert_stdio_conformance(spec)` — companion to `assert_streamable_http_conformance(endpoint)`. Workers run this against their own stdio configuration.

### Notes
- `worker_contract_version` stays at `1`. v1.4.0 is purely additive: the new transport opens up the community ecosystem of stdio MCP servers (Frida MCP, Ghidra MCP, mitmdump wrappers, etc.) without changing the contract surface.
- Existing Streamable HTTP callers are unaffected. `MCPClient("http://...")` positional usage continues to work.
- PAL consumers can bump the pin transparently; PAL doesn't use MCP workers in v1.

## [1.3.0] - 2026-05-16

### Added
- `agent_core.workers.MCPClient` — thin async wrapper over the official `mcp` Python SDK (v1.27.x) Streamable HTTP transport. Methods: `connect`, `initialize`, `list_tools`, `call_tool`, `close`.
- `agent_core.workers.MCPClientPool` — per-worker client cache, lazy-connect, reused across all dynamic Tool calls.
- `agent_core.workers.discover_and_register(specs, pool)` — discovers tools across all workers in a registry and produces ready-to-register `Tool` subclasses (name-prefixed `{worker}_{tool}`). The natural return shape for an agent's `register_tools()`.
- `agent_core.workers.make_tool_class(worker_spec, tool_def, pool)` — dynamic Tool subclass factory.
- `agent_core.workers.conformance.assert_streamable_http_conformance(endpoint)` — live-transport conformance check. Workers import this into their own test suites.

### Dependencies
- Added `mcp>=1.27.0` (official Anthropic MCP Python SDK) to dependencies.
- Added `fastmcp>=0.2.0` to dev dependencies (for the Streamable HTTP test fixture).

### Notes
- `worker_contract_version` stays at `1`. v1.3.0 is purely additive on top of the v1.2.x data layer; no fields removed, no schemas changed, no behaviour broken.
- PAL consumers can bump the pin transparently; PAL doesn't use MCP workers in v1.

## [1.2.0] - 2026-05-13

### Added
- `agent_core.boundary` module with `generate_guid()`, `wrap_untrusted()`, and `SANITIZATION_SYSTEM_PROMPT` extracted from PAL.
- `agent_core.workers` subpackage: `types` (RiskTier, WorkerSpec, WorkerError, AuditEntry, WORKER_CONTRACT_VERSION), `registry` (WorkerRegistry, WorkerNotFoundError), `risk` (RiskGate, TierDecision), `audit` (AuditLog), `conformance` (assert_conformance, MockWorkerContract).
- `Agent.register_tools()` lifecycle hook for dynamic tool registration at runtime; unioned with declarative `cls.tools` by `runtime._attach_registries`.
- Env-gated `test_reasoning_smoke.py` for verifying `reasoning_content` handling against a real local-manager model.

### Dependencies
- Added `pydantic>=2.0` as an explicit dependency (previously transitive).

### Notes
- PAL consumers should update their pin to `v1.2.0` and switch `pal.boundary` imports to `agent_core.boundary`. The PAL update PR is a no-op functionally — boundary helpers and reasoning API are unchanged.
- No breaking changes. Existing declarative `tools = [...]` works exactly as before.

## [1.0.0] - 2026-05-09

### Milestone

The agent_core extraction is complete. The framework's API surface — `Agent` base class, `BaseConfig`, `run_daemon`, `Tool` / `Command` / `SystemPromptBuilder`, the per-channel managers, the protocol layer, the Discord gateway helpers — is stable enough to depend on. PAL has been migrated end-to-end and is the canonical consumer; `Agent_Template` (https://github.com/EdibleTuber/agent_template) provides a starter scaffold for the next agent.

This is a tag, not a feature release. Code is identical to v0.7.0 except for the version bump and this entry. The 1.0 number reserves the right to break the API in a future major; from here, minor bumps preserve the public contract.

### What's in v1.0

- `Agent` base class with `tools` / `commands` / `disabled_builtins` ClassVars + `setup()` / `system_prompt(ctx)` / `handle_chat` / `handle_command` / `handle_other` extension points.
- `BaseConfig` (16 fields covering inference, retrieval, allowlist/web-search, channels, scratchpad, fetcher, batch).
- `run_daemon(agent)`: constructs framework managers + URLFetcher, calls `agent.setup()`, attaches `tool_executor` / `command_registry` / `prompt_builder` (via `_attach_registries`), then serves the daemon.
- `agent_core.tools`: Tool base, ToolExecutor with `requires`-validation + exception containment + asyncio.CancelledError reraise; 12 builtins (cat / head / tail / ls / grep / find / read_lines / fetch_url / search_vault / search_web / update_scratch / add_learning).
- `agent_core.commands`: Command base, CommandRegistry, 12 builtins (help / clear / status / profile / scratch / wisdom / learnings / promote / rate / model / think / quit).
- `agent_core.prompts.SystemPromptBuilder` with `render_profile` / `render_wisdom` / `render_scratchpad(channel_id)` / `render_commands_catalog` / `render_tools_catalog`.
- `agent_core.adapters.discord_gateway`: pure-Python composition helpers for Discord-using agents (UserConnectionManager, parse, split, rewrite, format).
- `agent_core.protocol`: NDJSON transport, message primitives, agent-extensible registry.
- `agent_core.channels` / `scratchpad` / `conversation` / `learning_scanner` / `git_helpers`: per-channel state.
- `agent_core.profile` / `wisdom` / `learning` / `allowlist` / `approval_registry`: per-agent stateful managers.
- `agent_core.inference` / `retrieval` / `websearch`: stateless HTTP clients.
- `agent_core.utils`: frontmatter, chunker, sanitizer, fetcher (URLFetcher with allowlist + content-type gating), converter.

### Test count

525 tests passing (340 baseline pre-extraction → +185 across Phases A-G).

### Phases recap

| Phase | Tag | What |
|---|---|---|
| A | v0.1.x | leaf utilities |
| B | v0.2.0 | stateless clients (inference / retrieval / websearch / reasoning) |
| C | v0.3.0 | stateful managers (wisdom / learning / profile / allowlist / approval_registry) |
| D | v0.4.0 | per-channel state (conversation / channels / scratchpad / learning_scanner) + protocol package |
| E | v0.5.x | runtime infrastructure (Agent / BaseConfig / run_daemon / Daemon / DaemonConnection / CLI adapter) |
| F | v0.6.x | tool/command/prompt scaffolding + 12 tool + 12 command builtins; v0.6.1 fixes registries-before-setup ordering |
| G | v0.7.0 | discord_gateway helpers |
| H | (consumer-side) | Agent_Template scaffold, validates the API surface |

### What's next

- Phase I (burn-in): use the framework + template for real work; file follow-ups for rough edges.
- The `_BasePALPromptAdapter` shim in PAL (PR6) is a known smell — Compiler/Consolidator are constructed in `setup()` before `_attach_registries` and need a base-prompt builder at construction. A future Agent extension point (`pre_setup`?) might let it go away.
- Inference safety guards (per-channel preemption, structured logging, forensic JSONL) deferred from Phase D — resumes after burn-in.

## [0.7.0] - 2026-05-07

### Added
- `agent_core.adapters.discord_gateway` module: composition helpers lifted from PAL's discord_adapter. Includes `UserConnectionManager` (per-user DaemonConnection lifecycle), `parse_discord_message` (`!cmd` vs chat split), `rewrite_slash_prefixes` (`/cmd` -> `!cmd` with code-fence protection), `split_message` (2000-char Discord limit), `format_tool_progress` (italic-wrapped tool label with optional per-tool custom_formatters dict).
- The module is pure-Python helpers -- no `import discord` at module level. Bot classes (discord.Client subclasses) stay in consumer code; this module is composition primitives.

### Notes
- Phase G ships the generic ~250 LOC. PAL retains `PalDiscordBot` plus its approval UX (button/modal handlers, proposal threads) since those are PAL-specific. A second Discord-using agent supplies its own bot class and uses these helpers directly.
- The `agent_core[discord]` extras_require remains for consumers that want discord.py installed transitively; this module itself doesn't depend on it.

## [0.6.1] - 2026-05-07

### Changed
- `run_daemon` now calls `Agent.setup()` BEFORE `_attach_registries`. Tool and command `requires` declarations can now reference domain managers constructed in `setup()` (e.g. PAL's `wiki`, `reorganizer`, `compiler`), not just framework managers wired by `run_daemon`. Registration-time validation still fails fast -- before any user message -- but now validates against the agent's full attribute set.

### Notes
- Trade-off: `Agent.setup()` no longer sees `tool_executor` / `command_registry` / `prompt_builder` attached. Setup is for constructing domain resources; the registries are built afterwards from the union of framework + domain state.
- Downstream impact: agents whose `setup()` was *reading* from the registries (none known to do this -- PAL doesn't) need to move that logic into a new override point or after `run_daemon` returns. Agents that simply *construct* domain resources in setup (PAL, the canonical case) work better than before -- domain-dependency tools now validate at boot.

## [0.6.0] - 2026-05-06

### Added
- `agent_core.tools` subpackage: `Tool` base class with `name`/`description`/`parameters`/`requires` ClassVars and `async def run(args, ctx) -> str`. `to_openai_schema()` classmethod returns OpenAI function-calling format.
- `agent_core.tools.executor.ToolExecutor` with `build()` classmethod that unions builtins with agent-supplied tool classes, validates `requires` against agent attributes via `hasattr`, and instantiates the surviving classes. Generic dispatch with exception containment; `asyncio.CancelledError` re-raised explicitly so cancellation propagates.
- 12 builtin tools in `agent_core.tools.builtin.BUILTIN_TOOLS`:
  - 7 read-only shell tools, vault-scoped, pure-Python: `cat`, `head`, `tail`, `ls`, `grep`, `find`, `read_lines`. 32 KB output cap (`OUTPUT_CAP_BYTES`); system paths (`_`-prefixed) rejected.
  - 5 framework-manager-backed: `fetch_url`, `search_vault`, `search_web`, `update_scratch`, `add_learning`.
- `agent_core.commands` subpackage: `Command` base class and `CommandRegistry` with the same `build()` validation pattern. `dispatch()` yields zero or more messages from the chosen command's async generator.
- 12 builtin commands in `agent_core.commands.builtin.BUILTIN_COMMANDS`: `help`, `clear`, `status`, `profile`, `scratch`, `wisdom`, `learnings`, `promote`, `rate`, `model`, `think`, `quit`. Each is a thin wrapper over a framework manager.
- `agent_core.prompts.builder.SystemPromptBuilder` with section render helpers: `render_profile()`, `render_wisdom()`, `render_scratchpad(channel_id)`, `render_commands_catalog()`, `render_tools_catalog()`. Each returns an empty string when underlying data is empty so callers can `filter(None, ...)` freely.
- `Agent` ClassVars: `tools = []`, `commands = []`, `disabled_builtins = frozenset()` for declarative registration with opt-out by name.
- `HandlerContext` fields: `agent` (back-reference for tools accessing `ctx.agent.X`) and `emit` (NDJSON-encoding writer callback for tools sending out-of-band messages mid-call). Both default to `None`; populated per turn by `Daemon._handle_connection`.
- `agent_core.runtime._attach_registries(agent)` builds `tool_executor`, `command_registry`, and `prompt_builder` from the agent's class-level registration, validates `requires` against the agent's attributes, and attaches all three before `Agent.setup()` runs. Misconfiguration fails fast at boot.
- `agent_core.runtime.run_daemon` now constructs a `URLFetcher(max_bytes=config.fetch_max_bytes, timeout=config.fetch_timeout)` and attaches it as `agent.fetcher` alongside the other framework managers, so the `fetch_url` builtin works out of the box.
- Contract tests pinning the registration API surface from a consumer's perspective.

### Notes
- Builtin tools and commands are opt-out via `Agent.disabled_builtins`; the default set is enabled in every agent that doesn't override it.
- `requires` validation is shallow (`hasattr`-based). Type-safe validation via Protocols is deferred to a future phase.
- Downstream agents that previously constructed their own `URLFetcher` in `setup()` can now drop that and rely on the framework default. Agents that want different fetch settings can still override `self.fetcher` in `setup()`.
- The `requires` of `Help` is `("command_registry",)` and the registry isn't attached when `CommandRegistry.build()` validates — `_attach_registries` sets a `None` sentinel before building so the `hasattr` check passes; the real registry overwrites it on the next line. The `Help` command only reads `command_registry` at dispatch time, so the sentinel is never user-visible.
- Phase G (next) is the Discord gateway adapter extraction.

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
