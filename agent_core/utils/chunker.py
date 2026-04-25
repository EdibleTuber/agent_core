"""Markdown chunker -- split documents at top-level headings.

Used by /import to break large documents into separate articles.
Detects the highest heading level (H1 or H2) and splits there.
"""
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) byte ranges covering fenced code blocks.

    A fence opens on a line starting with three or more backticks or tildes
    and closes on the next line starting with the same fence character
    (length greater than or equal to the opener). An unterminated fence
    extends to end of text.
    """
    ranges: list[tuple[int, int]] = []
    pos = 0
    while True:
        opener = _FENCE_RE.search(text, pos)
        if not opener:
            return ranges
        fence = opener.group(1)
        char = fence[0]
        min_len = len(fence)
        start = opener.start()
        search_from = text.find("\n", opener.end())
        if search_from == -1:
            ranges.append((start, len(text)))
            return ranges
        search_from += 1
        closer_re = re.compile(rf"^{re.escape(char)}{{{min_len},}}\s*$", re.MULTILINE)
        closer = closer_re.search(text, search_from)
        if not closer:
            ranges.append((start, len(text)))
            return ranges
        ranges.append((start, closer.end()))
        pos = closer.end()


def _in_any_range(pos: int, ranges: list[tuple[int, int]]) -> bool:
    for start, end in ranges:
        if start <= pos < end:
            return True
    return False


@dataclass
class Chunk:
    title: str
    body: str


def chunk_markdown(text: str, fallback_title: str) -> list[Chunk]:
    """Split markdown text at the highest heading level found.

    Args:
        text: markdown content to split
        fallback_title: title to use for content before the first heading,
                        or for the whole document if no headings exist

    Returns:
        list of Chunk(title, body). Empty list if text is blank.
    """
    if not text or not text.strip():
        return []

    fenced = _fenced_ranges(text)

    # Find all headings, ignoring any whose `#` lives inside a fenced code block
    headings = [
        (m.start(), len(m.group(1)), m.group(2).strip())
        for m in _HEADING_RE.finditer(text)
        if not _in_any_range(m.start(), fenced)
    ]

    if not headings:
        return [Chunk(title=fallback_title, body=text.strip())]

    # Detect highest (smallest number) heading level present
    split_level = min(level for _, level, _ in headings)

    # Filter to only headings at the split level
    split_points = [(pos, title) for pos, level, title in headings if level == split_level]

    if len(split_points) <= 1 and split_points[0][0] == 0:
        # Single heading at the start -- no splitting needed
        return [Chunk(title=split_points[0][1], body=text[split_points[0][0]:].strip())]

    chunks: list[Chunk] = []

    # Content before first heading
    first_pos = split_points[0][0]
    if first_pos > 0:
        pre_content = text[:first_pos].strip()
        if pre_content:
            chunks.append(Chunk(title=fallback_title, body=pre_content))

    # Each heading starts a chunk that runs until the next heading
    for i, (pos, title) in enumerate(split_points):
        if i + 1 < len(split_points):
            end = split_points[i + 1][0]
        else:
            end = len(text)

        body = text[pos:end].strip()
        if not body or body == f"{'#' * split_level} {title}":
            continue  # Skip empty chunks

        chunks.append(Chunk(title=title, body=body))

    return chunks
