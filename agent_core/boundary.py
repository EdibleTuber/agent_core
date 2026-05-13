"""GUID boundary wrapping for untrusted content.

When agent_core feeds untrusted content (worker output, fetched web
content, vault content from untrusted sources) to a model, it is wrapped
in <untrusted-content id="{guid}"> ... </untrusted-content>. The GUID is
randomly generated per request (or per session, depending on the consuming
agent's policy) — an attacker can't craft content that closes the
boundary because they don't know the GUID.

Paired with SANITIZATION_SYSTEM_PROMPT, which tells the model explicitly
to treat wrapped content as data, not instructions.

This module is the canonical location for the primitive. Extracted from
PAL's pal/boundary.py during agent_core v1.2.0; PAL re-imports from here.
"""
import uuid


def generate_guid() -> str:
    """Return a random UUID4 string for boundary tagging."""
    return str(uuid.uuid4())


# Exported symbols for Task 2 (will be fully implemented then)
wrap_untrusted = None
SANITIZATION_SYSTEM_PROMPT = None
