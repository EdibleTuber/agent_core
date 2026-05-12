"""Framework-manager-backed builtin tools.

Each tool only touches state already wired onto the Agent by run_daemon
(retrieval, websearch, allowlist, channels, learning, config) plus
agent.fetcher which agents must expose in setup() if they want FetchUrl.

Deviations from the original plan (method names verified against real modules):
  - RetrievalClient: .search(query, limit=N) not .query()
    Result dicts use keys: id, name, collection, summary, tags, score
  - LearningManager: .add(title, body, source) not .add_candidate()
    Returns a slug string.
  - FetchUrl: uses ctx.agent.fetcher.fetch(url) (URLFetcher instance the agent
    must wire in setup()) + ctx.agent.allowlist.is_allowed(url) check.
    There is no module-level fetch_and_extract() entry point.
  - UpdateScratch: ChannelStore has no .scratchpad() method; Scratchpad is
    constructed directly from ctx.agent.config attrs.
  - WebSearchClient: .search(query) -> list[SearchResult] — matches plan.
"""
from __future__ import annotations

from agent_core.scratchpad import Scratchpad, ScratchpadTooLarge
from agent_core.tools.base import Tool


def _truncate(s: str | None, n: int) -> str:
    """Normalize and truncate a string to n chars, with '…' suffix when cut.

    Newlines collapse to spaces; outer whitespace strips. If `s` exceeds
    `n` chars, cut at `n - 1` and append `…`. Any trailing whitespace at
    the cut point is trimmed before the ellipsis, so cuts that land right
    after a word produce a clean "word…" rather than "word …".

    Does NOT back up to the prior word boundary when the cut lands inside
    a word. Callers needing strict word-boundary truncation should use a
    different helper.
    """
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    # Reserve one char for the ellipsis; rstrip removes trailing space if the
    # cut landed right after a word.
    return s[: n - 1].rstrip() + "…"


class FetchUrl(Tool):
    """Fetch a URL through the agent's allowlist and return extracted text."""

    name = "fetch_url"
    description = (
        "Fetch a URL through the agent's allowlist and return extracted text content. "
        "Only URLs whose domain is on the allowlist are fetched."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
        },
        "required": ["url"],
    }
    requires = ("allowlist", "fetcher")

    async def run(self, args, ctx):
        url = (args.get("url") or "").strip()
        if not url:
            return "Error: 'url' parameter is required."
        if not ctx.agent.allowlist.is_allowed(url):
            return (
                f"Error: URL not allowed by allowlist: {url}\n"
                "Add its domain to the allowlist file in the vault, then retry."
            )
        try:
            doc = await ctx.agent.fetcher.fetch(url)
        except Exception as exc:
            return f"Fetch error: {exc}"
        title = getattr(doc, "title", "") or ""
        body = getattr(doc, "text", "") or ""
        if title:
            return f"# {title}\n\n{body}"
        return body or "(empty page)"


class SearchVault(Tool):
    """Semantic search over the vault via the retrieval service."""

    name = "search_vault"
    description = (
        "Semantic search over the vault. "
        "Returns matching documents with name and summary snippet."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {
                "type": "integer",
                "description": "Cap on results (default 5, max 20).",
            },
        },
        "required": ["query"],
    }
    requires = ("retrieval",)

    async def run(self, args, ctx):
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' parameter is required."
        max_results = max(1, min(int(args.get("max_results", 5)), 20))
        try:
            # RetrievalClient.search(query, limit=N) -> list[dict]
            # Result dict keys: id, name, collection, summary, tags, score
            results = await ctx.agent.retrieval.search(query, limit=max_results)
        except Exception as exc:
            return f"Search error: {exc}"
        if not results:
            return f"No vault matches for: {query}"
        lines = [f"Found {len(results)} match(es) for '{query}':"]
        for r in results:
            if isinstance(r, dict):
                name = r.get("name") or r.get("id", "?")
                snippet = r.get("summary", "")
            else:
                name = getattr(r, "name", None) or getattr(r, "id", "?")
                snippet = getattr(r, "summary", "")
            lines.append(f"  {name}")
            if snippet:
                lines.append(f"    {str(snippet)[:200]}")
        return "\n".join(lines)


class SearchWeb(Tool):
    """Search the web via the agent's SearxNG instance."""

    name = "search_web"
    description = (
        "Search the web via the agent's SearxNG instance. "
        "Returns title, URL, and snippet for each result (no page content fetched)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {
                "type": "integer",
                "description": "Cap on results (default 5, max 10).",
            },
        },
        "required": ["query"],
    }
    requires = ("websearch",)

    async def run(self, args, ctx):
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: 'query' parameter is required."
        max_results = max(1, min(int(args.get("max_results", 5)), 10))
        try:
            results = await ctx.agent.websearch.search(query)
        except Exception as exc:
            return f"Search error: {exc}"
        results = results[:max_results]
        if not results:
            return f"No web results for: {query}"
        lines = [f"Found {len(results)} result(s) for '{query}':"]
        for r in results:
            lines.append(f"  {r.title}")
            lines.append(f"    {r.url}")
            snippet = (r.snippet or "").strip().replace("\n", " ")[:200]
            if snippet:
                lines.append(f"    {snippet}")
        return "\n".join(lines)


class UpdateScratch(Tool):
    """Replace the scratchpad for the current channel."""

    name = "update_scratch"
    description = (
        "Replace the scratchpad for the current channel. "
        "Persisted across turns; size-capped per config. "
        "REPLACES the scratchpad wholesale — prior content is discarded."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Replacement scratchpad content (markdown ok).",
            },
        },
        "required": ["text"],
    }
    requires = ("config",)

    async def run(self, args, ctx):
        text = args.get("text", "")
        cfg = ctx.agent.config
        try:
            sp = Scratchpad(
                vault_path=cfg.vault_path,
                agent_name=ctx.agent.name,
                channel_id=ctx.channel_id,
                max_bytes=cfg.scratchpad_max_bytes,
            )
            sp.write(text)
        except ScratchpadTooLarge as exc:
            return (
                f"Error: scratchpad too large. "
                f"Proposed {exc.proposed_bytes} bytes, cap is {exc.max_bytes}. "
                "Shorten the content and retry."
            )
        except Exception as exc:
            return f"Scratchpad error: {exc}"
        return f"Scratchpad updated ({len(text.encode())} bytes)."


class AddLearning(Tool):
    """Capture a durable lesson as a learning candidate."""

    name = "add_learning"
    description = (
        "Capture a durable lesson as a learning candidate. "
        "Requires a short title and a 1-3 sentence body."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the lesson."},
            "body": {"type": "string", "description": "1-3 sentence body."},
        },
        "required": ["title", "body"],
    }
    requires = ("learning",)

    async def run(self, args, ctx):
        title = (args.get("title") or "").strip()
        body = (args.get("body") or "").strip()
        if not title or not body:
            return "Error: 'title' and 'body' are both required."
        try:
            # LearningManager.add(title, body, source) -> slug string
            slug = ctx.agent.learning.add(title=title, body=body, source="agent_tool")
        except Exception as exc:
            return f"Learning error: {exc}"
        return f"Added learning: {slug}"
