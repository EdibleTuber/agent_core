# agent_core

Shared library for the agent ecosystem. Provides daemon runtime, socket protocol, inference client, retrieval, wisdom, learning, channels, scratchpad, fetcher/chunker/converter, CLI REPL, and an opt-in Discord gateway adapter.

## Status

Under active extraction from PAL. See `docs/superpowers/specs/2026-04-25-agent-core-extraction-design.md` in the PAL repo for the design.

## Installation

Private repo, install via git:

```bash
pip install "agent_core @ git+https://github.com/EdibleTuber/agent_core.git@v0.1.1"
```

For development against a local checkout:

```bash
pip install -e /path/to/agent_core
```

## Building an agent

Subclass `agent_core.agent.Agent` and implement the handlers the daemon dispatches to.

**At minimum you must implement `handle_chat`** — it is the per-turn loop (call the
model, stream the reply, dispatch tool calls). The base class is a bare
`raise NotImplementedError`, so an agent that implements only
`setup`/`system_prompt`/`register_tools` will boot fine, register its tools, and serve
its socket — then fail on the **first chat message** with an empty `NotImplementedError`.
Don't get caught by this (PARE did).

Overrides:

- `handle_chat(self, msg, ctx) -> AsyncIterator[object]` — **required to converse.** Yield
  `StreamChunkMessage` / `ResponseMessage` / `ErrorMessage` / `ToolProgressMessage`. Use
  `agent_core.inference.InferenceClient` (`.stream()` / `.complete()`, with tool-call
  support) to talk to your model, and dispatch any tool calls through your tool pool so they
  are risk-gated + audited.
- `system_prompt(self, ctx) -> str` — **required** (also `NotImplementedError` in the base).
- `handle_command(self, msg, ctx)` — implement if you accept `/commands` (also a stub).
- `setup(self)` — optional; construct domain resources (framework managers are already populated).
- `register_tools(self)` — optional; return tool classes (e.g. from `discover_and_register`)
  to expose MCP workers.

The terminal REPL (`agent_core.adapters.cli.run_repl`) is a **library, not an entry point**.
Each agent ships its own launcher that calls `run_repl(config.socket_path, renderer)` with a
`Renderer` (two methods: `splash()` and `format_message()`).

### Gotchas

- **Worker discovery isn't resilient to unreachable workers (as of v1.6.0).**
  `discover_and_register` runs workers sequentially; an unreachable `streamable_http` worker's
  cancellation can cascade and cancel *sibling* workers' discovery, and closing clients logs a
  noisy but harmless anyio "cancel scope" traceback. If a healthy worker isn't registering,
  check whether an unreachable worker precedes it in the registry. Fix tracked for a later release.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
