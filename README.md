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

## Tests

```bash
pip install -e ".[dev]"
pytest
```
