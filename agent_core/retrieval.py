"""HTTP client for the inference server's collection search endpoints.

Thin wrapper over:
  POST /collections/{collection_id}/search
  GET  /collections/{collection_id}/docs/{doc_id}

The inference server handles embedding generation and vector search.
PAL's retrieval layer is used when the wiki outgrows index-file navigation
or for fuzzy/semantic queries.
"""
import logging

import httpx

logger = logging.getLogger(__name__)


class RetrievalClient:
    def __init__(self, base_url: str, collection_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection_id = collection_id
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        query: str,
        limit: int = 5,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Search the collection for documents matching the query.

        Returns a list of result dicts with keys: id, name, collection,
        summary, tags, score. Results are sorted by score (descending).
        """
        payload: dict = {"query": query, "limit": limit}
        if tags:
            payload["tags"] = tags
        resp = await self._client.post(
            f"{self.base_url}/collections/{self.collection_id}/search",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    async def get_document(self, doc_id: str) -> dict:
        """Fetch the full content of a document by its ID.

        Returns a dict with keys: id, name, collection, summary, content, metadata.
        Raises FileNotFoundError if the document doesn't exist.
        Raises ValueError if doc_id contains path traversal sequences.
        """
        if ".." in doc_id.split("/") or doc_id.startswith("/"):
            raise ValueError(f"Invalid doc_id: {doc_id}")
        resp = await self._client.get(
            f"{self.base_url}/collections/{self.collection_id}/docs/{doc_id}"
        )
        if resp.status_code == 404:
            raise FileNotFoundError(f"Document not found: {doc_id}")
        resp.raise_for_status()
        return resp.json()

    async def trigger_reindex(
        self,
        paths: list[str] | None = None,
    ) -> dict | None:
        """Ask the inference server to reindex the collection.

        With `paths` omitted: full incremental scan of the collection's
        source_dir. With `paths` provided: only those absolute paths are
        rescanned; stale-deletion is skipped.

        Returns the server's response dict on success (HTTP 202), or None
        on connection error or unexpected status. A None return is
        intentional best-effort: a downed inference server must never
        break the write path.
        """
        body: dict = {}
        if paths is not None:
            body["paths"] = list(paths)
        try:
            resp = await self._client.post(
                f"{self.base_url}/collections/{self.collection_id}/reindex",
                json=body,
            )
        except Exception as exc:
            logger.warning("trigger_reindex failed: %s", exc)
            return None
        if resp.status_code != 202:
            logger.warning(
                "trigger_reindex unexpected status %s: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        return resp.json()

    async def get_reindex_status(self) -> dict | None:
        """Fetch the current/most-recent reindex job for this collection.

        Returns the job dict or None (404 = no job yet, connection error,
        unexpected status).
        """
        try:
            resp = await self._client.get(
                f"{self.base_url}/collections/{self.collection_id}/reindex/status",
            )
        except Exception as exc:
            logger.warning("get_reindex_status failed: %s", exc)
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.warning("get_reindex_status status %s", resp.status_code)
            return None
        return resp.json()

    async def get_reindex_job(self, job_id: str) -> dict | None:
        """Fetch a specific job by id. Returns None on 404 or error."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/collections/{self.collection_id}/reindex/{job_id}",
            )
        except Exception as exc:
            logger.warning("get_reindex_job(%s) failed: %s", job_id, exc)
            return None
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.warning("get_reindex_job(%s) status %s", job_id, resp.status_code)
            return None
        return resp.json()
