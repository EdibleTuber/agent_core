"""Tests for the GUID boundary primitive extracted from PAL."""
import re

from agent_core.boundary import generate_guid, wrap_untrusted, SANITIZATION_SYSTEM_PROMPT


UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def test_generate_guid_returns_uuid4_string():
    guid = generate_guid()
    assert isinstance(guid, str)
    assert UUID4_RE.match(guid), f"not a UUID4: {guid!r}"


def test_generate_guid_returns_unique_values():
    guids = {generate_guid() for _ in range(100)}
    assert len(guids) == 100, "expected 100 unique GUIDs in 100 calls"


def test_wrap_untrusted_uses_supplied_guid():
    guid = "abc12345-6789-4abc-9def-0123456789ab"
    out = wrap_untrusted("hello world", guid)
    assert out.startswith(f'<untrusted-content id="{guid}">')
    assert out.endswith("</untrusted-content>")
    assert "hello world" in out


def test_wrap_untrusted_preserves_content_verbatim():
    guid = generate_guid()
    content = "line1\nline2\n\twith tab"
    out = wrap_untrusted(content, guid)
    # The content appears verbatim between the open and close tags,
    # surrounded by newlines (readability).
    assert f'<untrusted-content id="{guid}">\n{content}\n</untrusted-content>' == out


def test_sanitization_prompt_mentions_untrusted_content_tag():
    assert "<untrusted-content" in SANITIZATION_SYSTEM_PROMPT
    assert "DATA" in SANITIZATION_SYSTEM_PROMPT  # the "treat as data" rule
    assert "instruction" in SANITIZATION_SYSTEM_PROMPT.lower()
