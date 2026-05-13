"""Tests for the GUID boundary primitive extracted from PAL."""
import re

import pytest

from agent_core.boundary import (
    generate_guid,
    wrap_untrusted,
    SANITIZATION_SYSTEM_PROMPT,
)


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
