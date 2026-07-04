"""The REPL must not render a streamed turn's text twice.

On a reasoning=off text-only turn the agent streams StreamChunkMessage tokens
(printed live) and then emits a final ResponseMessage carrying the SAME joined
text. The old loop printed both -> the whole answer appeared twice. _TurnPrinter
suppresses the final ResponseMessage's text when it merely repeats what was
already streamed, printing only the newline that closes the streamed line.
Non-streamed turns (reasoning=on -> a lone ResponseMessage) are unaffected.
"""
from agent_core.adapters.cli import _TurnPrinter, _default_format
from agent_core.protocol import (
    ErrorMessage,
    ResponseMessage,
    StreamChunkMessage,
)


class _NullRenderer:
    """format_message returns None for everything -> default formatting (PARE)."""
    def splash(self) -> str:
        return ""

    def format_message(self, msg) -> str | None:
        return None


def _tp():
    return _TurnPrinter(_NullRenderer())


def test_streamed_then_duplicate_response_prints_text_once():
    tp = _tp()
    a = tp.emit(StreamChunkMessage(token="hello "))
    b = tp.emit(StreamChunkMessage(token="world"))
    r = tp.emit(ResponseMessage(text="hello world"))
    assert a == ("hello ", "")          # streamed live, no newline
    assert b == ("world", "")
    assert r == ("", "\n")              # duplicate suppressed: newline only


def test_response_without_streaming_prints_full_text():
    tp = _tp()
    r = tp.emit(ResponseMessage(text="the answer"))
    assert r == ("the answer", "\n")   # reasoning=on path: single print


def test_streamed_then_divergent_response_is_not_suppressed():
    """If the final ResponseMessage differs from the stream, print it (no data loss)."""
    tp = _tp()
    tp.emit(StreamChunkMessage(token="partial"))
    r = tp.emit(ResponseMessage(text="completely different final"))
    assert r == ("completely different final", "\n")


def test_error_after_streaming_is_printed():
    tp = _tp()
    tp.emit(StreamChunkMessage(token="working..."))
    r = tp.emit(ErrorMessage(error="boom"))
    assert r[0] == _default_format(ErrorMessage(error="boom"))
    assert r[1] == "\n"


def test_new_turn_resets_stream_state():
    """A fresh _TurnPrinter per turn: a lone ResponseMessage prints in full."""
    tp1 = _tp()
    tp1.emit(StreamChunkMessage(token="turn one"))
    tp1.emit(ResponseMessage(text="turn one"))     # suppressed
    tp2 = _tp()                                     # next turn, fresh state
    r = tp2.emit(ResponseMessage(text="turn one"))  # same text, but not streamed now
    assert r == ("turn one", "\n")                 # printed, not suppressed
