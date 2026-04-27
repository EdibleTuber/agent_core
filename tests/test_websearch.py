"""Tests for WebSearchClient — SearxNG HTTP client."""
import pytest

from agent_core.websearch import WebSearchClient, SearchResult


@pytest.mark.asyncio
async def test_search_returns_results(mock_inference_server):
    client = WebSearchClient(base_url=mock_inference_server)
    results = await client.search("quantum computing")
    assert len(results) >= 3
    assert isinstance(results[0], SearchResult)
    assert results[0].title


@pytest.mark.asyncio
async def test_search_includes_snippets(mock_inference_server):
    client = WebSearchClient(base_url=mock_inference_server)
    results = await client.search("python")
    assert all(r.snippet for r in results)


@pytest.mark.asyncio
async def test_search_result_has_all_fields(mock_inference_server):
    client = WebSearchClient(base_url=mock_inference_server)
    results = await client.search("test")
    r = results[0]
    assert r.url
    assert r.title
    assert r.snippet
