"""Tests for URLFetcher — fetch + extract + validate."""
import pytest

from agent_core.utils.fetcher import URLFetcher, FetchResult, FetchError


@pytest.fixture(autouse=True)
def _disable_blocklist_for_mock(monkeypatch, request):
    """Disable blocklist for tests that use the mock server on 127.0.0.1."""
    if "mock_inference_server" in request.fixturenames:
        monkeypatch.setattr("agent_core.utils.fetcher.check_url_safety", lambda url: None)


# ---- Blocklist tests ----


@pytest.mark.asyncio
async def test_fetch_rejects_private_ip_127():
    """Fetcher must reject localhost URLs."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://127.0.0.1:8080/secret")


@pytest.mark.asyncio
async def test_fetch_rejects_private_ip_10():
    """Fetcher must reject 10.x.x.x range."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://10.0.0.1/internal")


@pytest.mark.asyncio
async def test_fetch_rejects_private_ip_172():
    """Fetcher must reject 172.16-31.x.x range."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://172.16.0.1/internal")


@pytest.mark.asyncio
async def test_fetch_rejects_private_ip_192_168():
    """Fetcher must reject 192.168.x.x range."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://192.168.1.1/admin")


@pytest.mark.asyncio
async def test_fetch_rejects_ipv6_localhost():
    """Fetcher must reject ::1."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://[::1]:8080/secret")


@pytest.mark.asyncio
async def test_fetch_rejects_file_scheme():
    """Fetcher must reject file:// URLs."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("file:///etc/passwd")


@pytest.mark.asyncio
async def test_fetch_rejects_ftp_scheme():
    """Fetcher must reject ftp:// URLs."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("ftp://internal-server/files")


@pytest.mark.asyncio
async def test_fetch_rejects_zero_ip():
    """Fetcher must reject 0.0.0.0 (commonly used SSRF bypass)."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://0.0.0.0:8080/secret")


@pytest.mark.asyncio
async def test_fetch_rejects_dns_rebinding():
    """Fetcher must reject hostnames that resolve to private IPs."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="blocked"):
        await fetcher.fetch("http://localhost:9999/page")


# ---- Existing tests ----


def test_fetcher_has_user_agent():
    """URLFetcher should configure a User-Agent header."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    assert fetcher.headers.get("User-Agent")
    assert "PAL" in fetcher.headers["User-Agent"]


@pytest.mark.asyncio
async def test_fetch_extracts_main_content(mock_inference_server):
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    result = await fetcher.fetch(f"{mock_inference_server}/page.html")
    assert isinstance(result, FetchResult)
    assert "main content" in result.text.lower() or "extract me" in result.text.lower()
    assert "nav junk" not in result.text.lower()
    assert result.url == f"{mock_inference_server}/page.html"
    assert result.title == "Test Page"


@pytest.mark.asyncio
async def test_fetch_rejects_too_large(mock_inference_server):
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="too large"):
        await fetcher.fetch(f"{mock_inference_server}/too-large")


@pytest.mark.asyncio
async def test_fetch_rejects_binary(mock_inference_server):
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="content type"):
        await fetcher.fetch(f"{mock_inference_server}/binary")


@pytest.mark.asyncio
async def test_fetch_404_raises(mock_inference_server):
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="404"):
        await fetcher.fetch(f"{mock_inference_server}/missing")


@pytest.mark.asyncio
async def test_fetch_result_has_hash(mock_inference_server):
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    result = await fetcher.fetch(f"{mock_inference_server}/page.html")
    assert result.content_hash
    assert len(result.content_hash) == 64  # sha256 hex


@pytest.mark.asyncio
async def test_fetch_respects_max_bytes_during_download(mock_inference_server):
    """If response streams more than max_bytes, fetch should abort."""
    fetcher = URLFetcher(max_bytes=1, timeout=10)
    with pytest.raises(FetchError, match="too large"):
        await fetcher.fetch(f"{mock_inference_server}/page.html")


@pytest.mark.asyncio
async def test_fetch_rejects_redirects(mock_inference_server):
    """Redirects must be rejected (SSRF protection)."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="redirect"):
        await fetcher.fetch(f"{mock_inference_server}/redirect")


@pytest.mark.asyncio
async def test_fetch_rejects_missing_content_type(mock_inference_server):
    """Missing Content-Type header must be rejected."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    with pytest.raises(FetchError, match="Content-Type"):
        await fetcher.fetch(f"{mock_inference_server}/no-content-type")


@pytest.mark.asyncio
async def test_fetch_preserves_code_blocks(mock_inference_server):
    """Trafilatura markdown output should preserve code fences."""
    fetcher = URLFetcher(max_bytes=2_000_000, timeout=10)
    result = await fetcher.fetch(f"{mock_inference_server}/page-with-code.html")
    assert "```" in result.text
