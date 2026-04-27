"""WebSearchClient — thin HTTP client for SearxNG.

SearxNG is a self-hosted meta-search engine. This client hits its
JSON search endpoint and returns results as structured SearchResult objects.
"""
from dataclasses import dataclass

import httpx


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


class WebSearchClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def search(self, query: str) -> list[SearchResult]:
        """Query SearxNG, return raw results (no allowlist filtering applied)."""
        resp = await self._client.get(
            f"{self.base_url}/search",
            params={"q": query, "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
            ))
        return results
