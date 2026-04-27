"""Tests for the retrieval client — collection search and doc fetch."""
import pytest

from agent_core.retrieval import RetrievalClient


@pytest.mark.asyncio
async def test_search_returns_results(mock_inference_server):
    client = RetrievalClient(base_url=mock_inference_server, collection_id="vault")
    results = await client.search("quantum computing")
    assert len(results) == 3
    assert results[0]["id"] == "doc-0"
    assert "quantum computing" in results[0]["summary"]
    assert results[0]["score"] > results[1]["score"]


@pytest.mark.asyncio
async def test_search_respects_limit(mock_inference_server):
    client = RetrievalClient(base_url=mock_inference_server, collection_id="vault")
    results = await client.search("anything", limit=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_document(mock_inference_server):
    client = RetrievalClient(base_url=mock_inference_server, collection_id="vault")
    doc = await client.get_document("Projects/alpha.md")
    assert doc["id"] == "Projects/alpha.md"
    assert "Full content" in doc["content"]


@pytest.mark.asyncio
async def test_get_document_not_found(mock_inference_server):
    client = RetrievalClient(base_url=mock_inference_server, collection_id="vault")
    with pytest.raises(FileNotFoundError):
        await client.get_document("missing")


@pytest.mark.asyncio
async def test_get_document_rejects_path_traversal(mock_inference_server):
    client = RetrievalClient(base_url=mock_inference_server, collection_id="vault")
    with pytest.raises(ValueError, match="Invalid doc_id"):
        await client.get_document("../../etc/passwd")
    with pytest.raises(ValueError, match="Invalid doc_id"):
        await client.get_document("/absolute/path")


# ---------------------------------------------------------------------------
# Reindex methods
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_trigger_reindex_full_scan_posts_empty_body():
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=202)
    fake_response.json.return_value = {
        "job_id": "j1", "collection_id": "vault", "status": "queued",
        "paths": None, "stats": {"new": 0, "updated": 0, "removed": 0, "unchanged": 0},
        "started_at": "2026-04-15T00:00:00Z", "finished_at": None, "error": None,
    }
    client._client.post = AsyncMock(return_value=fake_response)

    result = await client.trigger_reindex()

    client._client.post.assert_awaited_once_with(
        "http://server/collections/vault/reindex",
        json={},
    )
    assert result == fake_response.json.return_value
    assert result["job_id"] == "j1"


@pytest.mark.asyncio
async def test_trigger_reindex_with_paths_posts_paths():
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=202)
    fake_response.json.return_value = {
        "job_id": "j2", "collection_id": "vault", "status": "queued",
        "paths": ["/abs/x.md"],
    }
    client._client.post = AsyncMock(return_value=fake_response)

    result = await client.trigger_reindex(paths=["/abs/x.md"])

    client._client.post.assert_awaited_once_with(
        "http://server/collections/vault/reindex",
        json={"paths": ["/abs/x.md"]},
    )
    assert result["paths"] == ["/abs/x.md"]


@pytest.mark.asyncio
async def test_trigger_reindex_returns_none_on_connection_error():
    """A downed inference server must not break the write path. trigger_reindex
    returns None and logs a warning."""
    import httpx
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    client._client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await client.trigger_reindex(paths=["/abs/x.md"])
    assert result is None


@pytest.mark.asyncio
async def test_trigger_reindex_returns_none_on_unexpected_status():
    """500 (or any non-202) returns None."""
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=500)
    fake_response.text = "internal error"
    client._client.post = AsyncMock(return_value=fake_response)

    result = await client.trigger_reindex()
    assert result is None


@pytest.mark.asyncio
async def test_get_reindex_status_returns_dict():
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "job_id": "j1", "status": "done", "collection_id": "vault",
    }
    client._client.get = AsyncMock(return_value=fake_response)

    result = await client.get_reindex_status()

    client._client.get.assert_awaited_once_with(
        "http://server/collections/vault/reindex/status",
    )
    assert result["status"] == "done"


@pytest.mark.asyncio
async def test_get_reindex_status_returns_none_on_404():
    """No job recorded yet returns None (not an error)."""
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=404)
    client._client.get = AsyncMock(return_value=fake_response)

    result = await client.get_reindex_status()
    assert result is None


@pytest.mark.asyncio
async def test_get_reindex_job_by_id():
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=200)
    fake_response.json.return_value = {
        "job_id": "abc", "status": "running", "collection_id": "vault",
    }
    client._client.get = AsyncMock(return_value=fake_response)

    result = await client.get_reindex_job("abc")

    client._client.get.assert_awaited_once_with(
        "http://server/collections/vault/reindex/abc",
    )
    assert result["status"] == "running"


@pytest.mark.asyncio
async def test_get_reindex_job_404_returns_none():
    client = RetrievalClient(base_url="http://server", collection_id="vault")
    fake_response = MagicMock(status_code=404)
    client._client.get = AsyncMock(return_value=fake_response)

    result = await client.get_reindex_job("missing")
    assert result is None
