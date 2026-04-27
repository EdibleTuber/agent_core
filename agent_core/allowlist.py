"""AllowlistManager — domain allowlist for web fetching.

The allowlist lives at _config/allowlist.md in the vault. It's a markdown
bullet list of domain patterns (one per line). Supports:
    exact.domain        — matches exactly that host
    *.subdomain.tld     — matches any subdomain + the bare domain

Only http:// and https:// URLs are ever allowed.
"""
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


STARTER_ALLOWLIST = """# Web Allowlist

Domains PAL is allowed to fetch. Edit this file to add or remove entries.
Patterns: `example.com` matches the exact host. `*.example.com` matches any subdomain AND the bare domain.

## Reference
- wikipedia.org
- *.wikipedia.org
- wiktionary.org
- *.wiktionary.org
- plato.stanford.edu

## Academic
- arxiv.org
- semanticscholar.org
- pubmed.ncbi.nlm.nih.gov

## Technical
- *.readthedocs.io
- docs.python.org
- developer.mozilla.org

## Code
- github.com
- stackoverflow.com
- stackexchange.com

## Standards
- rfc-editor.org
- w3.org
- ietf.org
"""


class AllowlistManager:
    def __init__(self, vault_path: Path, agent_name: str) -> None:
        self.vault_path = vault_path
        self.agent_name = agent_name

    @property
    def allowlist_path(self) -> Path:
        return self.vault_path / "_config" / self.agent_name / "allowlist.md"

    def seed(self) -> None:
        """Write the starter allowlist if no allowlist file exists yet."""
        if self.allowlist_path.exists():
            return
        self.allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        self.allowlist_path.write_text(STARTER_ALLOWLIST)
        logger.info("Seeded allowlist at %s", self.allowlist_path)

    def list(self) -> list[str]:
        """Parse the allowlist file, return all domain patterns."""
        if not self.allowlist_path.exists():
            return []
        patterns = []
        for line in self.allowlist_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                patterns.append(stripped[2:].strip())
        return patterns

    def is_allowed(self, url: str) -> bool:
        """Return True if the URL is http/https AND its host matches a pattern."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        for pattern in self.list():
            if pattern.startswith("*."):
                bare = pattern[2:]
                if host == bare or host.endswith("." + bare):
                    return True
            else:
                if host == pattern:
                    return True
        return False
