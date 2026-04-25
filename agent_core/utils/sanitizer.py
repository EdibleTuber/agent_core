"""Content sanitization for untrusted text fed to the local model.

Defense-in-depth alongside GUID boundaries. Sanitization is not a silver
bullet — it reduces attack surface by removing known injection vectors.

Pipeline:
  1. Unicode NFC normalization (collapse homoglyph variants)
  2. Strip zero-width characters (U+200B-D, U+FEFF)
  3. Strip bidirectional control characters (Trojan Source)
  4. Strip model special tokens (<|im_start|>, <|endoftext|>, etc.)
  5. Remove GUID echoes (paranoid — GUID is per-request and unpredictable)
  6. Min length check (warn, do not reject)
  7. Token budget truncation (char-count estimate, ~4 chars per token)
"""
import re
import unicodedata
from dataclasses import dataclass, field


ZERO_WIDTH = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # byte-order mark
}

BIDI_CONTROLS = {
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # deprecated bidi
    "\u2066", "\u2067", "\u2068", "\u2069",            # isolates
}

SPECIAL_TOKEN_RE = re.compile(
    r"<\|[a-zA-Z_][a-zA-Z0-9_]*\|>"
)

CHARS_PER_TOKEN = 4  # Rough estimate for English text


@dataclass
class SanitizationResult:
    text: str
    issues: list[str] = field(default_factory=list)
    original_length: int = 0
    sanitized_length: int = 0
    token_count_estimate: int = 0
    truncated: bool = False


def sanitize(
    text: str,
    guid: str,
    min_chars: int = 10,
    max_tokens: int = 32_000,
) -> SanitizationResult:
    """Sanitize untrusted text before feeding it to a model.

    Args:
        text: raw content to sanitize
        guid: the boundary GUID that will wrap this content (for echo check)
        min_chars: warn if content is shorter than this
        max_tokens: truncate if estimated tokens exceeds this

    Returns a SanitizationResult with cleaned text and a list of issues.
    """
    original_length = len(text)
    issues: list[str] = []

    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Strip zero-width characters
    zw_count = sum(text.count(c) for c in ZERO_WIDTH)
    if zw_count > 0:
        for c in ZERO_WIDTH:
            text = text.replace(c, "")
        issues.append(f"stripped {zw_count} zero-width character(s)")

    # 3. Strip bidirectional controls
    bidi_count = sum(text.count(c) for c in BIDI_CONTROLS)
    if bidi_count > 0:
        for c in BIDI_CONTROLS:
            text = text.replace(c, "")
        issues.append(f"stripped {bidi_count} bidi control character(s)")

    # 4. Strip model special tokens
    matches = SPECIAL_TOKEN_RE.findall(text)
    if matches:
        text = SPECIAL_TOKEN_RE.sub("", text)
        issues.append(f"stripped {len(matches)} model special token(s)")

    # 5. Remove GUID echoes (shouldn't happen — GUID is per-request)
    if guid and guid in text:
        text = text.replace(guid, "[REDACTED]")
        issues.append("removed GUID echo from content (suspicious)")

    sanitized_length = len(text)

    # 6. Min length check (warn only)
    if sanitized_length < min_chars:
        issues.append(f"content too short ({sanitized_length} chars)")

    # 7. Token budget truncation
    token_count_estimate = sanitized_length // CHARS_PER_TOKEN
    truncated = False
    if token_count_estimate > max_tokens:
        max_chars = max_tokens * CHARS_PER_TOKEN
        text = text[:max_chars]
        sanitized_length = len(text)
        token_count_estimate = sanitized_length // CHARS_PER_TOKEN
        truncated = True
        issues.append(f"truncated to ~{max_tokens} tokens ({max_chars} chars)")

    return SanitizationResult(
        text=text,
        issues=issues,
        original_length=original_length,
        sanitized_length=sanitized_length,
        token_count_estimate=token_count_estimate,
        truncated=truncated,
    )
