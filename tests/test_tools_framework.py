"""Tests for framework-backed builtin tools.

Method-name deviations from the original plan:
  - RetrievalClient: .search(query, limit=N) not .query(query, top_k=N)
  - LearningManager:  .add(title, body, source) not .add_candidate(title, body)
  - FetchUrl:         uses ctx.agent.fetcher.fetch(url) + allowlist.is_allowed()
                      not a module-level fetch_and_extract(url, allowlist=...)
  - UpdateScratch:    constructs Scratchpad directly from ctx.agent.config attrs
                      (ChannelStore has no .scratchpad() method)
  - WebSearchClient:  .search(query) returning list[SearchResult] — matches plan
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_core.tools._framework import (
    AddLearning,
    FetchUrl,
    SearchVault,
    SearchWeb,
    UpdateScratch,
)


def _ctx(agent, channel_id="default"):
    class _C:
        pass
    c = _C()
    c.agent = agent
    c.channel_id = channel_id
    return c


# ---------------------------------------------------------------------------
# SearchWeb
# ---------------------------------------------------------------------------

async def test_search_web_formats_results():
    from agent_core.websearch import SearchResult
    websearch = MagicMock()
    websearch.search = AsyncMock(return_value=[
        SearchResult(url="http://a.com", title="A", snippet="snippet A"),
        SearchResult(url="http://b.com", title="B", snippet="snippet B"),
    ])
    agent = MagicMock(websearch=websearch)
    result = await SearchWeb().run({"query": "test"}, _ctx(agent))
    assert "A" in result and "http://a.com" in result and "snippet A" in result
    assert "B" in result and "http://b.com" in result


async def test_search_web_requires_query():
    agent = MagicMock()
    result = await SearchWeb().run({}, _ctx(agent))
    assert "query" in result.lower() and "required" in result.lower()


async def test_search_web_empty_results():
    websearch = MagicMock()
    websearch.search = AsyncMock(return_value=[])
    agent = MagicMock(websearch=websearch)
    result = await SearchWeb().run({"query": "nothing"}, _ctx(agent))
    assert "no" in result.lower()


# ---------------------------------------------------------------------------
# SearchVault
# ---------------------------------------------------------------------------

async def test_search_vault_calls_retrieval():
    retrieval = MagicMock()
    # Real method: .search(query, limit=N) returning list[dict] with keys
    # id, name, collection, summary, tags, score
    retrieval.search = AsyncMock(return_value=[
        {"name": "Notes/a.md", "summary": "matched content"},
    ])
    agent = MagicMock(retrieval=retrieval)
    result = await SearchVault().run({"query": "test"}, _ctx(agent))
    retrieval.search.assert_called_once_with("test", limit=5)
    assert "Notes/a.md" in result
    assert "matched" in result


async def test_search_vault_requires_query():
    agent = MagicMock()
    result = await SearchVault().run({}, _ctx(agent))
    assert "query" in result.lower() and "required" in result.lower()


async def test_search_vault_empty_results():
    retrieval = MagicMock()
    retrieval.search = AsyncMock(return_value=[])
    agent = MagicMock(retrieval=retrieval)
    result = await SearchVault().run({"query": "nothing"}, _ctx(agent))
    assert "no" in result.lower()


# ---------------------------------------------------------------------------
# FetchUrl
# ---------------------------------------------------------------------------

async def test_fetch_url_calls_fetcher():
    """FetchUrl checks allowlist then calls agent.fetcher.fetch(url)."""
    from agent_core.utils.fetcher import FetchResult
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value=FetchResult(
        url="http://example.com",
        title="Example Page",
        text="page body",
        content_hash="abc",
        byte_size=100,
    ))
    allowlist = MagicMock()
    allowlist.is_allowed = MagicMock(return_value=True)
    agent = MagicMock(fetcher=fetcher, allowlist=allowlist)
    result = await FetchUrl().run({"url": "http://example.com"}, _ctx(agent))
    allowlist.is_allowed.assert_called_once_with("http://example.com")
    fetcher.fetch.assert_called_once_with("http://example.com")
    assert "page body" in result


async def test_fetch_url_blocked_by_allowlist():
    allowlist = MagicMock()
    allowlist.is_allowed = MagicMock(return_value=False)
    agent = MagicMock(allowlist=allowlist)
    result = await FetchUrl().run({"url": "http://example.com"}, _ctx(agent))
    assert "allowlist" in result.lower() or "not allowed" in result.lower()


async def test_fetch_url_requires_url():
    agent = MagicMock()
    result = await FetchUrl().run({}, _ctx(agent))
    assert "url" in result.lower() and "required" in result.lower()


# ---------------------------------------------------------------------------
# UpdateScratch
# ---------------------------------------------------------------------------

async def test_update_scratch_writes():
    """UpdateScratch constructs a Scratchpad from config attrs and calls write()."""
    with patch("agent_core.tools._framework.Scratchpad") as MockScratchpad:
        mock_sp = MagicMock()
        mock_sp.write = MagicMock()
        MockScratchpad.return_value = mock_sp

        config = MagicMock()
        config.vault_path = "/fake/vault"
        config.scratchpad_max_bytes = 2048
        # MagicMock(name=...) sets the mock display name, not .name attr —
        # set .name explicitly after construction.
        agent = MagicMock(config=config)
        agent.name = "test-agent"
        result = await UpdateScratch().run({"text": "new content"}, _ctx(agent, "ch1"))

        MockScratchpad.assert_called_once_with(
            vault_path=config.vault_path,
            agent_name="test-agent",
            channel_id="ch1",
            max_bytes=2048,
        )
        mock_sp.write.assert_called_once_with("new content")
        assert "updated" in result.lower() or "ok" in result.lower()


async def test_update_scratch_too_large():
    """ScratchpadTooLarge is surfaced as an error string."""
    from agent_core.scratchpad import ScratchpadTooLarge
    with patch("agent_core.tools._framework.Scratchpad") as MockScratchpad:
        mock_sp = MagicMock()
        mock_sp.write = MagicMock(side_effect=ScratchpadTooLarge(100, 9999, 2048))
        MockScratchpad.return_value = mock_sp

        config = MagicMock()
        config.vault_path = "/fake/vault"
        config.scratchpad_max_bytes = 2048
        agent = MagicMock(config=config)
        agent.name = "test-agent"
        result = await UpdateScratch().run({"text": "x" * 9999}, _ctx(agent))
        assert "scratchpad" in result.lower() or "too large" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# AddLearning
# ---------------------------------------------------------------------------

async def test_add_learning_stores():
    """AddLearning calls learning.add(title, body, source) and returns slug."""
    learning = MagicMock()
    # Real method: .add(title, body, source) -> slug string
    learning.add = MagicMock(return_value="my-lesson-slug")
    agent = MagicMock(learning=learning)
    result = await AddLearning().run({"title": "T", "body": "B"}, _ctx(agent))
    learning.add.assert_called_once()
    call_kwargs = learning.add.call_args
    assert call_kwargs.kwargs.get("title") == "T" or call_kwargs.args[0] == "T"
    assert call_kwargs.kwargs.get("source") == "agent_tool"
    assert "my-lesson-slug" in result or "added" in result.lower()


async def test_add_learning_requires_title_and_body():
    agent = MagicMock()
    result = await AddLearning().run({"title": "T"}, _ctx(agent))
    assert "required" in result.lower() or "body" in result.lower()

    result2 = await AddLearning().run({"body": "B"}, _ctx(agent))
    assert "required" in result2.lower() or "title" in result2.lower()
