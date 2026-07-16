"""Tests for SearchVault's tags filter and doc_id direct-fetch path."""
from __future__ import annotations

import json

import pytest

from agent_core.tools._framework import SearchVault


class _FakeRetrieval:
    def __init__(self):
        self.search_calls = []
        self.doc_calls = []

    async def search(self, query, limit=5, tags=None):
        self.search_calls.append({"query": query, "limit": limit, "tags": tags})
        return [{"id": "raw/notes/android-vulnerable-sinks-reference",
                 "name": "sinks", "summary": "s", "score": 0.9}]

    async def get_document(self, doc_id):
        self.doc_calls.append(doc_id)
        return {"id": doc_id, "name": "sinks", "summary": "s",
                "content": "# sinks\n...", "metadata": {}}


class _FakeAgent:
    def __init__(self):
        self.retrieval = _FakeRetrieval()


class _FakeCtx:
    def __init__(self):
        self.agent = _FakeAgent()


@pytest.mark.asyncio
async def test_search_vault_forwards_tags():
    ctx = _FakeCtx()
    out = json.loads(await SearchVault().run({"query": "sinks", "tags": ["sinks"]}, ctx))
    assert out["status"] == "ok"
    assert ctx.agent.retrieval.search_calls[0]["tags"] == ["sinks"]


@pytest.mark.asyncio
async def test_search_vault_doc_id_fetches_document():
    ctx = _FakeCtx()
    out = json.loads(await SearchVault().run(
        {"doc_id": "raw/notes/android-vulnerable-sinks-reference"}, ctx))
    assert out["status"] == "ok"
    assert ctx.agent.retrieval.doc_calls == ["raw/notes/android-vulnerable-sinks-reference"]
    assert "content" in out


@pytest.mark.asyncio
async def test_search_vault_doc_id_takes_priority_over_query():
    """A doc_id is a direct fetch; it must short-circuit even if query is also set."""
    ctx = _FakeCtx()
    out = json.loads(await SearchVault().run(
        {"doc_id": "raw/notes/android-vulnerable-sinks-reference", "query": "ignored"}, ctx))
    assert out["status"] == "ok"
    assert ctx.agent.retrieval.doc_calls == ["raw/notes/android-vulnerable-sinks-reference"]
    assert ctx.agent.retrieval.search_calls == []


@pytest.mark.asyncio
async def test_search_vault_doc_id_fetch_error_returns_json_error():
    ctx = _FakeCtx()

    async def _boom(doc_id):
        raise FileNotFoundError(f"Document not found: {doc_id}")

    ctx.agent.retrieval.get_document = _boom
    out = json.loads(await SearchVault().run({"doc_id": "missing/doc"}, ctx))
    assert out["status"] == "error"
    assert out["doc_id"] == "missing/doc"
    assert "FileNotFoundError" in out["reason"]


@pytest.mark.asyncio
async def test_search_vault_no_query_or_doc_id_is_error():
    ctx = _FakeCtx()
    out = json.loads(await SearchVault().run({}, ctx))
    assert out["status"] == "error"
