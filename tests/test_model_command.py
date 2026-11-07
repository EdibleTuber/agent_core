"""/model speaks the manager's phase 5 contract (inference-server phase 6, Q18).

- No argument: one line per slot /status reports, whatever the slots are called.
- A swap POSTs /swap with an X-Request-Id, sending "target" only when given.
  200 means already loaded. 202 means the swap is queued: the command polls
  /status until that slot's last_swap carries its own request id, and reports
  that entry's outcome.
- The session is never repointed: a swap changes a slot for every client, and
  which model this agent sends stays its configuration's business.
- While it waits, the command yields ToolProgressMessage, never an interim
  ResponseMessage: every reader (DaemonConnection.command, the CLI REPL, PAL's
  CLI and Discord) stops at the first ResponseMessage, so an interim one would
  leave the real outcome in the socket to be read as the next command's answer.

The manager is a MockTransport swapped into every httpx.AsyncClient the code
builds, so the tests hold whether /model uses the inference client's own
httpx client or builds its own.
"""
import re
from unittest.mock import MagicMock

import httpx
import pytest

from agent_core.commands import _builtin_impls
from agent_core.commands._builtin_impls import Model
from agent_core.conversation import Conversation
from agent_core.inference import InferenceClient
from agent_core.protocol.messages import (
    ErrorMessage,
    ResponseMessage,
    ToolProgressMessage,
)

TOKEN = "s3cret-manager-token"
SESSION = "session-model"


class FakeManager:
    """A scripted manager: /status, /swap and /v1/models.

    After a 202, /status shows `older` as the swap slot's last_swap for
    `polls_before` polls, then this request's own entry with `outcome`.
    `never` keeps showing `older` forever. `status_errors` makes that many
    polls after the 202 fail with a 503 first.
    """

    def __init__(self, *, swap_status=202, swap_slot="egpu", outcome="ok",
                 message=None, polls_before=0, older=None, never=False,
                 status_errors=0, refuse=None):
        self.swap_status = swap_status
        self.swap_slot = swap_slot
        self.outcome = outcome
        self.message = message
        self.polls_before = polls_before
        self.older = older
        self.never = never
        self.status_errors = status_errors
        self.refuse = refuse
        self.requests: list[httpx.Request] = []
        self.swap_bodies: list[dict] = []
        self.rid: str | None = None
        self.polls_after_swap = 0

    def slots(self, last_swap):
        return {
            "main": {"loaded_model": "qwen-3.8", "state": "idle", "last_swap": None},
            "egpu": {"loaded_model": "gemma-4", "state": "swapping",
                     "last_swap": last_swap if self.swap_slot == "egpu" else None},
            "spare": {"loaded_model": None, "state": "unhealthy",
                      "last_swap": last_swap if self.swap_slot == "spare" else None},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        import json
        self.requests.append(request)
        path = request.url.path
        if path == "/status":
            if self.rid is None:
                return httpx.Response(200, json={"slots": self.slots(self.older)})
            self.polls_after_swap += 1
            if self.status_errors:
                self.status_errors -= 1
                return httpx.Response(503, json={"error": {"type": "busy"}})
            if self.never or self.polls_after_swap <= self.polls_before:
                return httpx.Response(200, json={"slots": self.slots(self.older)})
            own = {"id": self.rid, "model": self.swap_bodies[-1]["model"],
                   "trial": False, "outcome": self.outcome,
                   "message": self.message, "at_utc": "2026-11-05T00:00:00+00:00"}
            return httpx.Response(200, json={"slots": self.slots(own)})
        if path == "/swap":
            body = json.loads(request.content)
            self.swap_bodies.append(body)
            if self.refuse is not None:
                status, err_type, msg = self.refuse
                return httpx.Response(status, json={"error": {"type": err_type,
                                                              "message": msg}})
            self.rid = request.headers.get("x-request-id")
            payload = {"slot": self.swap_slot, "model": body["model"],
                       "status": "ok" if self.swap_status == 200 else "swapping"}
            return httpx.Response(self.swap_status, json=payload,
                                  headers={"x-request-id": self.rid or "x"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "gemma-4"}]})
        return httpx.Response(404)


@pytest.fixture()
def fast(monkeypatch):
    """Poll every 10 ms; give up after 0.3 s."""
    monkeypatch.setattr(_builtin_impls, "MODEL_SWAP_POLL_SECONDS", 0.01)
    monkeypatch.setattr(_builtin_impls, "MODEL_SWAP_WAIT_SECONDS", 0.3)


def _install(monkeypatch, manager: FakeManager):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real(
        *a, **{**kw, "transport": httpx.MockTransport(manager)}))


def _ctx(inf):
    class _C:
        pass
    c = _C()
    c.agent = MagicMock(inference=inf, config=MagicMock(model="config-model"))
    c.channel_id = "default"
    c.conversation = Conversation(history_depth=10)
    return c


async def _run(monkeypatch, manager, args, token=None):
    _install(monkeypatch, manager)
    inf = InferenceClient(base_url="http://mgr", model=SESSION, token=token)
    msgs = [m async for m in Model().run(args, _ctx(inf))]
    return inf, msgs


def _final(msgs):
    assert msgs, "no message"
    finals = [m for m in msgs if isinstance(m, (ResponseMessage, ErrorMessage))]
    assert len(finals) == 1 and finals[0] is msgs[-1], (
        "exactly one ResponseMessage or ErrorMessage, and it comes last", msgs)
    return msgs[-1]


def _text(m):
    return m.text if isinstance(m, ResponseMessage) else m.error


# --- showing ---------------------------------------------------------------------

async def test_status_lists_every_slot_the_manager_reports(monkeypatch):
    inf, msgs = await _run(monkeypatch, FakeManager(), "")
    final = _final(msgs)
    assert isinstance(final, ResponseMessage)
    lines = final.text.splitlines()
    slot_lines = [ln for ln in lines if ln.startswith("  ")]
    assert slot_lines == [
        "  main: qwen-3.8 (idle)",
        "  egpu: gemma-4 (swapping)",
        "  spare: none (unhealthy)",
    ]
    assert lines[-1] == f"session sends: {SESSION}"


# --- swapping ---------------------------------------------------------------------

async def test_a_swap_without_target_sends_no_target(monkeypatch, fast):
    manager = FakeManager()
    await _run(monkeypatch, manager, "gemma-4")
    assert manager.swap_bodies == [{"model": "gemma-4"}]


async def test_a_swap_with_target_sends_it_unchecked(monkeypatch, fast):
    """Slot names are the manager's to check: any name goes through as given."""
    manager = FakeManager(swap_slot="spare")
    await _run(monkeypatch, manager, "--target spare gemma-4")
    assert manager.swap_bodies == [{"model": "gemma-4", "target": "spare"}]


async def test_a_swap_carries_a_fresh_request_id(monkeypatch, fast):
    manager = FakeManager(swap_status=200)
    _install(monkeypatch, manager)
    inf = InferenceClient(base_url="http://mgr", model=SESSION)
    for _ in range(2):
        [m async for m in Model().run("gemma-4", _ctx(inf))]
    ids = [r.headers.get("x-request-id") for r in manager.requests
           if r.url.path == "/swap"]
    assert len(ids) == 2
    assert all(rid and re.fullmatch(r"[0-9a-f]{32}", rid) for rid in ids), ids
    assert ids[0] != ids[1]


async def test_a_200_is_reported_at_once(monkeypatch, fast):
    manager = FakeManager(swap_status=200, swap_slot="main")
    _, msgs = await _run(monkeypatch, manager, "qwen-3.8")
    final = _final(msgs)
    assert isinstance(final, ResponseMessage)
    assert "already loaded" in final.text
    assert "qwen-3.8" in final.text
    assert not any(isinstance(m, ToolProgressMessage) for m in msgs)
    assert manager.polls_after_swap == 0, "a 200 needs no polling"


async def test_a_202_is_polled_until_its_own_last_swap(monkeypatch, fast):
    """An older swap's entry is on the slot first; it is not this swap's answer."""
    older = {"id": "an-older-swap", "model": "other", "trial": False,
             "outcome": "failed", "message": "old failure",
             "at_utc": "2026-11-04T00:00:00+00:00"}
    manager = FakeManager(polls_before=3, older=older)
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    final = _final(msgs)
    assert isinstance(final, ResponseMessage), _text(final)
    assert "gemma-4" in final.text and "egpu" in final.text
    assert "old failure" not in final.text
    assert manager.polls_after_swap >= 4


async def test_the_slot_polled_is_the_one_the_202_names(monkeypatch, fast):
    """No target was sent; the manager chose 'spare'. Only that slot's
    last_swap answers this swap."""
    manager = FakeManager(swap_slot="spare")
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    final = _final(msgs)
    assert isinstance(final, ResponseMessage), _text(final)
    assert "spare" in final.text
    # The swap landed: read on the first poll of 'spare', not waited out to
    # the cap (which also names the slot, and is also a ResponseMessage).
    assert "still running" not in final.text
    assert manager.polls_after_swap == 1


async def test_a_202_yields_progress_not_an_interim_response(monkeypatch, fast):
    _, msgs = await _run(monkeypatch, FakeManager(polls_before=2), "gemma-4")
    progress = [m for m in msgs if isinstance(m, ToolProgressMessage)]
    assert progress, "a queued swap tells the operator it is waiting"
    assert progress[0].tool == "model"
    assert progress[0].arguments["status"] == "swapping egpu -> gemma-4"
    _final(msgs)  # and only one Response/Error, last


async def test_a_status_error_while_polling_does_not_end_the_wait(monkeypatch, fast):
    """The manager may answer a poll 503 mid-swap; the next poll can still succeed."""
    manager = FakeManager(status_errors=2)
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    final = _final(msgs)
    assert isinstance(final, ResponseMessage), _text(final)
    assert manager.polls_after_swap == 3, "two failed polls, then the answer"


@pytest.mark.parametrize("outcome", ["ok", "already_loaded"])
async def test_a_landed_swap_is_reported_as_success(monkeypatch, fast, outcome):
    manager = FakeManager(outcome=outcome)
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    assert isinstance(_final(msgs), ResponseMessage)
    assert manager.polls_after_swap >= 1, "a 202 is not the outcome; last_swap is"


async def test_a_swap_that_fell_back_is_reported_as_a_failure(monkeypatch, fast):
    manager = FakeManager(outcome="fell_back",
                          message="gemma-4 did not come up; back on qwen-3.6")
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    final = _final(msgs)
    assert isinstance(final, ErrorMessage)
    assert "fell_back" in final.error
    assert "gemma-4 did not come up; back on qwen-3.6" in final.error


async def test_a_failed_swap_is_reported_as_a_failure(monkeypatch, fast):
    manager = FakeManager(outcome="failed", message="Model swap failed on slot egpu")
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    final = _final(msgs)
    assert isinstance(final, ErrorMessage)
    assert "failed" in final.error
    assert "Model swap failed on slot egpu" in final.error


async def test_a_swap_still_running_at_the_cap_says_so(monkeypatch, fast):
    manager = FakeManager(never=True)
    _, msgs = await _run(monkeypatch, manager, "gemma-4")
    final = _final(msgs)
    # A swap still running is not a failure: the plan makes the cap a
    # ResponseMessage, like success.
    assert isinstance(final, ResponseMessage), _text(final)
    text = final.text
    assert "still running" in text
    assert "/model" in text
    assert manager.polls_after_swap >= 2


async def test_a_refused_swap_shows_the_managers_message(monkeypatch, fast):
    msg = "'target' must be one of: 'main', 'egpu', 'spare'"
    manager = FakeManager(refuse=(400, "invalid_target", msg))
    _, msgs = await _run(monkeypatch, manager, "--target nope gemma-4")
    final = _final(msgs)
    assert isinstance(final, ErrorMessage)
    assert msg in final.error
    assert manager.polls_after_swap == 0


async def test_target_without_a_name_shows_usage(monkeypatch, fast):
    manager = FakeManager()
    _, msgs = await _run(monkeypatch, manager, "--target egpu")
    assert "usage" in _text(_final(msgs)).lower()
    assert manager.swap_bodies == []


@pytest.mark.parametrize("case", ["200", "ok", "fell_back", "failed", "cap", "refused"])
async def test_no_swap_outcome_repoints_the_session(monkeypatch, fast, case):
    manager = {
        "200": FakeManager(swap_status=200, swap_slot="main"),
        "ok": FakeManager(outcome="ok"),
        "fell_back": FakeManager(outcome="fell_back", message="m"),
        "failed": FakeManager(outcome="failed", message="m"),
        "cap": FakeManager(never=True),
        "refused": FakeManager(refuse=(404, "model_not_found", "no profile")),
    }[case]
    inf, msgs = await _run(monkeypatch, manager, "gemma-4")
    assert inf.default_model == SESSION
    final = _final(msgs)
    if isinstance(final, ResponseMessage):
        assert final.text.splitlines()[-1] == f"session sends: {SESSION}"


# --- the token ------------------------------------------------------------------

@pytest.mark.parametrize("args", ["", "list", "gemma-4"])
async def test_every_model_call_sends_the_token(monkeypatch, fast, args):
    manager = FakeManager(polls_before=1)
    await _run(monkeypatch, manager, args, token=TOKEN)
    assert manager.requests
    assert [r.headers.get("authorization") for r in manager.requests] == (
        [f"Bearer {TOKEN}"] * len(manager.requests))


async def test_no_token_sends_no_authorization_header(monkeypatch, fast):
    manager = FakeManager()
    await _run(monkeypatch, manager, "gemma-4")
    assert manager.requests
    assert all("authorization" not in r.headers for r in manager.requests)


# --- a failed call names what went wrong --------------------------------------
# An httpx timeout stringifies to '' (str(httpx.ReadTimeout("")) == ""), so a
# message built from {exc} alone read "Swap failed (gemma-4): " -- blank. Main
# fixed this in c7a037a, whose /model this branch replaced; the fix returns
# here (phase 6, Task 3 review).

class _Raising(FakeManager):
    """Raises `exc` for requests to `path`; answers everything else."""

    def __init__(self, path, exc, **kwargs):
        super().__init__(**kwargs)
        self.raise_path, self.exc = path, exc

    def __call__(self, request):
        if request.url.path == self.raise_path:
            self.requests.append(request)
            raise self.exc
        return super().__call__(request)


@pytest.mark.parametrize("args,path", [("list", "/v1/models"), ("gemma-4", "/swap")],
                         ids=["list", "swap"])
@pytest.mark.parametrize("exc,shown", [
    (httpx.ReadTimeout(""), "ReadTimeout"),
    (httpx.ConnectError("All connection attempts failed"),
     "All connection attempts failed"),
], ids=["blank-timeout", "with-message"])
async def test_a_failed_call_says_what_went_wrong(monkeypatch, fast, args, path,
                                                  exc, shown):
    _, msgs = await _run(monkeypatch, _Raising(path, exc), args)
    final = _final(msgs)
    assert isinstance(final, ErrorMessage)
    assert shown in final.error
