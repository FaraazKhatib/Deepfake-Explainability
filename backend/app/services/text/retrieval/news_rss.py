"""
Google News RSS evidence source.

`news.google.com/rss/search?q=...` is a keyword search over live news that
returns RSS. Free, no API key, no registration. Two things make it valuable
alongside GDELT:

* each item carries a ``<source url="...">`` element naming the real outlet,
  so the domain is reliable even though the link itself is a Google redirect;
* ``<description>`` usually holds a short snippet, giving the NLI model more
  than a bare headline to work with.

Parsing uses the standard library (``xml.etree``) rather than ``feedparser``,
keeping the dependency list short. Feeds are XML, occasionally malformed; a
parse failure raises and the aggregator records it.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List

from .. import config
from ..textutils import strip_html
from .base import Evidence, http_get, normalise_domain, parse_datetime

logger = logging.getLogger(__name__)

SOURCE_NAME = "Google News RSS"

# RFC 822, as used by RSS 2.0 pubDate.
_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %Z",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT",
    "%d %b %Y %H:%M:%S %z",
]


def _split_google_title(title: str) -> tuple:
    """
    Google News titles look like "Real headline - Outlet Name".

    Returns (headline, outlet). The separator also occurs inside legitimate
    headlines, so only the final segment is treated as an outlet, and only when
    it is short enough to plausibly be a masthead.
    """
    if " - " not in title:
        return title, ""
    head, _, tail = title.rpartition(" - ")
    if head and len(tail) <= 40:
        return head.strip(), tail.strip()
    return title, ""


def search(query: str, max_items: int | None = None) -> List[Evidence]:
    """Keyword-search live news via Google News RSS."""
    if not query.strip():
        return []

    max_items = config.GOOGLE_NEWS_MAX_ITEMS if max_items is None else max_items
    params = {
        "q": query,
        "hl": config.GOOGLE_NEWS_LANG,
        "gl": config.GOOGLE_NEWS_COUNTRY,
        "ceid": f"{config.GOOGLE_NEWS_COUNTRY}:{config.GOOGLE_NEWS_LANG.split('-')[0]}",
    }

    body = http_get(config.GOOGLE_NEWS_ENDPOINT, params=params)

    try:
        root = ET.fromstring(body.encode("utf-8"))
    except ET.ParseError as exc:
        raise RuntimeError(f"could not parse Google News RSS: {exc}") from exc

    results: List[Evidence] = []
    for item in root.iter("item"):
        if len(results) >= max_items:
            break

        raw_title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not raw_title or not link:
            continue

        headline, outlet_from_title = _split_google_title(raw_title)

        # <source url="https://www.bbc.com">BBC News</source>
        source_element = item.find("source")
        outlet = outlet_from_title
        domain = ""
        if source_element is not None:
            outlet = (source_element.text or outlet).strip()
            domain = normalise_domain(source_element.get("url", ""))
        if not domain:
            domain = normalise_domain(link)

        snippet = strip_html(item.findtext("description") or "")
        # The description often just repeats the headline plus the outlet.
        if snippet.startswith(headline[:40]):
            snippet = snippet[len(headline) :].strip(" -|")

        results.append(
            Evidence(
                title=headline,
                url=link,
                domain=domain,
                source=SOURCE_NAME,
                snippet=snippet[:500],
                published=parse_datetime(item.findtext("pubDate") or "", _DATE_FORMATS),
                outlet=outlet or domain,
            )
        )

    logger.debug("Google News returned %d items for %r", len(results), query)
    return results
