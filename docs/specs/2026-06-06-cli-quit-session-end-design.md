# `/quit` ends the CLI loop via a protocol session-end signal

**Date:** 2026-06-06
**Status:** Proposed design (awaiting review)
**Scope:** `agent_core` — shared CLI adapter, builtin `Quit` command, protocol

## Problem

In `agent_core`'s shared REPL (`adapters/cli.py::run_repl`), typing `/quit`
prints "Goodbye." but does not exit — the prompt returns and the loop continues.
Observed in PARE (which consumes `run_repl` directly).

### Root cause (established)

`/quit` has no loop-termination logic anywhere in the shared path:

- **Client** (`adapters/cli.py:127` `run_repl`): `/quit` matches the generic
  `line.startswith("/")` branch and is sent as `CommandMessage(name="quit")`. The
  client drains responses until a `ResponseMessage`/`ErrorMessage`, `break`s the
  **inner** receive loop (`cli.py:160-161`), then returns to the top of
  `while True` and prompts again. The outer loop only exits on
  `EOFError`/`KeyboardInterrupt` (Ctrl-D / Ctrl-C).
- **Daemon** (`daemon.py:101`): dispatches `quit` like any command — no teardown.
- **Command** (`commands/_builtin_impls.py:326` `Quit.run`): yields exactly
  `ResponseMessage(text="Goodbye.")`. No terminal marker (`ResponseMessage` has
  no such field), no exit signal.

So the goodbye prints, the inner loop breaks, and the prompt returns — exactly
the reported symptom.

### Why fix it in `agent_core` (not copy PAL)

PAL's CLI (`pal/cli.py`) works because it **forked the entire loop** and hardcodes
`if cmd_name in ("quit","exit"): break` (`pal/cli.py:327`), breaking *before* it
ever contacts the daemon. That is the outlier: `agent_core.adapters.cli` is the
loop *meant* to be reused ("a library; each agent provides its entry"), and it is
the one that's broken. Fixing it there fixes PARE and every future consumer.

The deeper smell PAL exposes: **"does this command end the session?" is knowledge
that currently lives nowhere shared** — each loop must independently hardcode the
command names. PAL remembered; the shared loop forgot; nothing prevents them
drifting again. The fix puts that knowledge in **one** place — the command/
protocol — so no loop hardcodes command names.

## Design

A response may carry a `end_session` flag. The `Quit` command sets it. Interactive
loops break when they observe it; long-running adapters ignore it.

### 1. Protocol — `ResponseMessage.end_session`

`protocol/messages.py`:

```python
@register_message
@dataclass
class ResponseMessage:
    text: str
    command: str = ""
    reasoning: str = ""
    end_session: bool = False   # ← new; True asks an interactive client to exit
    type: str = "response"
```

The flag lives on the message instance, so the *command* decides per-invocation
(a future `/logout` could set it too). Default `False` keeps every existing
response non-terminal.

**Serialization:** `encode_message` (`asdict`→JSON) and `decode_message`
(`cls(**obj)`) handle the new field transparently when client and daemon share the
schema — which PARE/PAL always do (client + daemon ship from one `agent_core`
install and upgrade together).

### 2. Command — `Quit` marks its response terminal

`commands/_builtin_impls.py`:

```python
class Quit(Command):
    name = "quit"
    args = ""
    description = "End the session"

    async def run(self, raw_args: str, ctx) -> AsyncIterator:
        yield ResponseMessage(text="Goodbye.", end_session=True)
```

The daemon path is otherwise unchanged: it still dispatches and streams the
response normally; the only difference is the flag on the final message.

### 3. Client — `run_repl` breaks on the signal

`adapters/cli.py`, inside the inner receive loop: when a message carries
`end_session`, remember it; after the inner loop ends (so "Goodbye." has already
printed), break the outer `while True`.

```python
should_exit = False
async for msg in conn.receive():
    if isinstance(msg, ToolApprovalRequestMessage):
        ...
        continue
    rendered = renderer.format_message(msg)
    if rendered is None:
        rendered = _default_format(msg)
    if isinstance(msg, StreamChunkMessage):
        print(rendered, end="", flush=True)
    else:
        print(rendered, flush=True)
    if getattr(msg, "end_session", False):
        should_exit = True
    if isinstance(msg, (ResponseMessage, ErrorMessage)):
        break
if should_exit:
    break
```

`getattr(msg, "end_session", False)` keeps it safe for message types without the
field. The daemon's "Goodbye." still prints first, then the loop exits cleanly
through the existing `finally: await conn.close()`. No hardcoded command names in
the client.

### 4. Gateway — no change (ignores the flag)

`adapters/discord_gateway.py` has no receive loop that inspects `ResponseMessage`
(its only `break` is inside `split_message`). It never reads `end_session`, so a
Discord `/quit` just relays "Goodbye." text and the gateway keeps running — the
correct behavior for a long-running multi-user service. The signal is
"this interactive session ends," and each adapter decides what to do with it
(CLI breaks; gateway ignores).

### PAL interaction

PAL is unaffected: its CLI breaks client-side *before* sending `quit`, so its
daemon never runs the modified `Quit`, and `pal/cli.py` reads only `resp.text`/
`resp.reasoning`, never `end_session`. **Follow-up (out of scope):** PAL could
later honor `end_session` and delete its hardcoded `("quit","exit")` match,
collapsing the duplication this design removes from the shared loop.

## What changes

- `protocol/messages.py` — add `end_session: bool = False` to `ResponseMessage`.
- `commands/_builtin_impls.py` — `Quit.run` yields `end_session=True`.
- `adapters/cli.py` — `run_repl` breaks the outer loop on an `end_session` message.
- No change to `daemon.py` or `discord_gateway.py`.

## Testing

- **Unit — command** (`tests/` builtin): `Quit.run` yields one `ResponseMessage`
  with `text == "Goodbye."` and `end_session is True`.
- **Unit — loop exit (failing test first)** (`tests/test_cli.py`): monkeypatch
  `cli.DaemonConnection` with a fake whose `receive()` yields
  `ResponseMessage(text="Goodbye.", end_session=True)`, and `cli.PromptSession`
  with a fake whose `prompt_async` returns `"/quit"` on the first call and raises
  if called again. Assert `run_repl` **returns** (no second prompt) and that
  "Goodbye." was printed. This fails on current code (loop re-prompts).
- **Unit — non-terminal response keeps looping** (guards over-broad break): a
  `ResponseMessage` with default `end_session=False` does not exit; the loop
  prompts again (second `prompt_async` returns EOF to end the test).
- **Unit — serialization round-trip:** `decode_message(encode_message(
  ResponseMessage(text="x", end_session=True)))` preserves the flag.

## Out of scope / noted

- **Protocol version skew.** `decode_message` does `cls(**obj)` — strict: a newer
  daemon emitting `end_session` to an *older* client lacking the field would raise
  `TypeError: unexpected keyword argument`. Safe for PARE/PAL (single install,
  versioned together). A general hardening — filtering `obj` to the dataclass's
  known fields in `decode_message` — is a separate robustness improvement, not
  required here.
- **Converging PAL onto the shared loop.** PAL's loop carries real extra UX (rich
  console, live-markdown streaming) the minimal shared loop lacks; collapsing the
  two is a larger refactor for separate payoff, deliberately not attempted here.

## Rollout

The change is in `agent_core`, which is installed as a **copied (non-editable)**
package in PARE's venv (v1.6.0, `site-packages`; source byte-identical). The fix
will not reach the running `pare-cli` until agent_core is reinstalled into PARE's
venv — or flipped to an editable install (`pip install -e
/home/edible/Projects/agent_core`) so future fixes propagate without a reinstall.
PAL's venv also carries agent_core but is unaffected by this fix at runtime.
