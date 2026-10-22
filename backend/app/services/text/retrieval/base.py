"""
Shared types and HTTP plumbing for evidence sources.

Every source returns a list of ``Evidence``. A source that fails -- network
error, rate limit, malformed response -- returns an empty list and records the
failure. One dead endpoint must never take the whole check down; it degrades
the evidence pool, and that degradation is reported to the user.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .. import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------
@dataclass
class Evidence:
    """One retrieved item that may bear on a claim."""

    title: str
    url: str
    domain: str
    source: str                       # which backend found it
    snippet: str = ""
    published: Optional[datetime] = None
    language: str = "en"
    outlet: str = ""                  # human-readable outlet name when known

    @property
    def premise(self) -> str:
        """The text handed to the NLI model as the premise."""
        parts = [self.title.strip()]
        if self.snippet and self.snippet.strip() != self.title.strip():
            parts.append(self.snippet.strip())
        joined = ". ".join(p.rstrip(".") for p in parts if p)
        return re.sub(r"\s+", " ", joined).strip()

    @property
    def age_days(self) -> Optional[float]:
        if self.published is None:
            return None
        now = datetime.now(timezone.utc)
        published = self.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        return max(0.0, (now - published).total_seconds() / 86400.0)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "outlet": self.outlet or self.domain,
            "source": self.source,
            "snippet": self.snippet[:400],
            "published": self.published.isoformat() if self.published else None,
            "age_days": round(self.age_days, 2) if self.age_days is not None else None,
        }


@dataclass
class RetrievalReport:
    """What each backend contributed, including its failures."""

    evidence: List[Evidence] = field(default_factory=list)
    per_source_counts: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    # Informational, NOT failures. Kept separate so the explanation does not
    # describe a working filter as a source that "did not respond".
    notices: Dict[str, str] = field(default_factory=dict)
    queries_used: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_items": len(self.evidence),
            "per_source_counts": self.per_source_counts,
            "errors": self.errors,
            "notices": self.notices,
            "queries_used": self.queries_used,
        }


# --------------------------------------------------------------------------
# Domain helpers
# --------------------------------------------------------------------------
def normalise_domain(url_or_domain: str) -> str:
    """Lowercase registrable-ish domain, with 'www.' and ports removed."""
    value = (url_or_domain or "").strip().lower()
    if "://" in value:
        value = urlparse(value).netloc
    value = value.split("@")[-1].split(":")[0]
    if value.startswith("www."):
        value = value[4:]
    return value


def canonical_url(url: str) -> str:
    """Strip tracking parameters and fragments so duplicates collapse."""
    if not url:
        return ""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{normalise_domain(parsed.netloc)}{parsed.path}".rstrip("/")


# --------------------------------------------------------------------------
# HTTP with retry + a small TTL cache
# --------------------------------------------------------------------------
class _TTLCache:
    """Tiny thread-safe TTL cache, so development does not hammer free APIs."""

    def __init__(self, ttl: int, max_entries: int):
        self.ttl = ttl
        self.max_entries = max_entries
        self._data: Dict[str, Tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        if self.ttl <= 0:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.time() - stored_at > self.ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            if len(self._data) >= self.max_entries:
                oldest = min(self._data.items(), key=lambda kv: kv[1][0])[0]
                self._data.pop(oldest, None)
            self._data[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CACHE = _TTLCache(config.CACHE_TTL_SECONDS, config.CACHE_MAX_ENTRIES)


def http_get(
    url: str,
    params: Optional[dict] = None,
    use_cache: bool = True,
    user_agent: Optional[str] = None,
) -> str:
    """
    GET a URL and return the body as text.

    Uses ``urllib`` from the standard library rather than httpx, for a reason
    found by testing rather than by preference: the Wikipedia API returns 403
    to httpx requests regardless of headers, while identical requests from
    urllib succeed. The block is not User-Agent based -- it was reproduced with
    minimal headers, several UA strings and various Accept-Encoding values --
    so it appears to be client fingerprinting above the header layer. urllib is
    also one fewer dependency in the retrieval path.

    A descriptive User-Agent is always sent: GDELT answers 429 to requests that
    arrive without one, which is easy to misdiagnose as rate limiting.
    """
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    full_url = url + (("?" + urlencode(params)) if params else "")

    if use_cache:
        cached = _CACHE.get(full_url)
        if cached is not None:
            logger.debug("cache hit: %s", url)
            return cached

    headers = {
        "User-Agent": user_agent or config.HTTP_USER_AGENT,
        "Accept": "*/*",
    }

    last_error: Optional[str] = None
    for attempt in range(config.HTTP_MAX_RETRIES + 1):
        try:
            request = Request(full_url, headers=headers)
            with urlopen(request, timeout=config.HTTP_TIMEOUT) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read().decode(charset, errors="replace")
            if use_cache:
                _CACHE.set(full_url, body)
            return body
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 429:
                last_error = "rate limited (HTTP 429)"
                # GDELT throttles aggressively; back off harder than usual.
                time.sleep(1.5 * (attempt + 1))
                continue
            if 400 <= exc.code < 500 and exc.code != 429:
                break  # client errors will not fix themselves on retry
        except (URLError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < config.HTTP_MAX_RETRIES:
            time.sleep(0.6 * (attempt + 1))

    raise RuntimeError(f"GET {url} failed: {last_error}")


def clear_cache() -> None:
    """Drop cached responses (used by tests and the /cache endpoint)."""
    _CACHE.clear()


# --------------------------------------------------------------------------
def parse_datetime(value: str, formats: List[str]) -> Optional[datetime]:
    """Try several timestamp formats, returning None rather than raising."""
    if not value:
        return None
    cleaned = value.strip()
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
        except ValueError:
            continue
    try:  # last resort: ISO 8601
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
