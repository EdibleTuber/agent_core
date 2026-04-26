"""URLFetcher — fetch URLs, extract main content, enforce limits.

Performs:
  1. Streamed download with byte cap (rejects oversized responses mid-stream)
  2. Content-Type validation (only text/html, text/plain, application/xhtml+xml)
  3. Content-Length header check where available
  4. trafilatura extraction (strips nav/footer/ads, keeps article body)
  5. SHA-256 hashing for provenance
  6. Private IP / reserved address blocklist (SSRF defense-in-depth)
"""
from dataclasses import dataclass
import hashlib
import ipaddress
import re
import socket as _socket
from urllib.parse import urlparse

import httpx
import trafilatura


_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)


class FetchError(Exception):
    """Raised when a URL can't be fetched for safety/correctness reasons."""


_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_ALLOWED_SCHEMES = ("http", "https")


def _is_private_ip(ip_str: str) -> bool:
    """Return True if ip_str falls in a private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return addr.is_private or addr.is_reserved or any(addr in net for net in _PRIVATE_NETWORKS)


def check_url_safety(url: str) -> None:
    """Raise FetchError if URL targets a private/reserved address or bad scheme.

    Resolves hostname via DNS to catch rebinding attacks.
    Note: TOCTOU gap exists between this DNS check and httpx's connection.
    A short-TTL record could return a public IP here and private IP at
    connect time. This is defense-in-depth, not a complete solution.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise FetchError(f"blocked: scheme '{parsed.scheme}' not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise FetchError("blocked: no hostname in URL")

    # Check if hostname is already a literal IP
    try:
        addr = ipaddress.ip_address(hostname)
        if _is_private_ip(hostname):
            raise FetchError(f"blocked: {hostname} is a private/reserved address")
        return
    except ValueError:
        pass  # Not a literal IP, resolve via DNS

    # DNS resolution check
    try:
        results = _socket.getaddrinfo(hostname, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in results:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                raise FetchError(f"blocked: {hostname} resolves to private address {ip_str}")
    except _socket.gaierror:
        pass  # DNS failure — let httpx handle it with a proper timeout error


ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
)


@dataclass
class FetchResult:
    url: str
    title: str
    text: str
    content_hash: str
    byte_size: int


class URLFetcher:
    def __init__(self, max_bytes: int, timeout: int) -> None:
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.headers = {"User-Agent": "PAL/1.0 (+personal knowledge base)"}

    async def fetch(self, url: str) -> FetchResult:
        """Fetch a URL and return extracted main content.

        Redirects are NOT followed — the caller has already validated the
        specific URL against the allowlist, and a redirect could land on a
        different host (SSRF risk). If the server returns a redirect, fetch
        fails and the caller can explicitly fetch the redirect target.
        """
        check_url_safety(url)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            headers=self.headers,
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "")
                    raise FetchError(
                        f"redirect not followed (HTTP {resp.status_code} to {location})"
                    )
                if resp.status_code >= 400:
                    raise FetchError(f"HTTP {resp.status_code} for {url}")

                ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if not ct:
                    raise FetchError("missing Content-Type header")
                if not any(ct.startswith(t) for t in ALLOWED_CONTENT_TYPES):
                    raise FetchError(f"rejected content type: {ct}")

                cl = resp.headers.get("content-length")
                if cl:
                    try:
                        cl_int = int(cl)
                    except ValueError:
                        raise FetchError(f"invalid Content-Length header: {cl}")
                    if cl_int > self.max_bytes:
                        raise FetchError(f"response too large (Content-Length: {cl})")

                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise FetchError(f"response too large (exceeded {self.max_bytes} bytes)")
                    chunks.append(chunk)
                raw = b"".join(chunks)

        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception as exc:
            raise FetchError(f"decode error: {exc}")

        text = trafilatura.extract(html, output_format="markdown") or ""

        # Prefer the HTML <title> tag; fall back to trafilatura metadata (h1, og:title, etc.)
        m = _TITLE_RE.search(html)
        if m:
            title = m.group(1).strip()
        else:
            metadata = trafilatura.extract_metadata(html)
            title = metadata.title if metadata and metadata.title else ""

        content_hash = hashlib.sha256(raw).hexdigest()

        return FetchResult(
            url=url,
            title=title,
            text=text,
            content_hash=content_hash,
            byte_size=len(raw),
        )
