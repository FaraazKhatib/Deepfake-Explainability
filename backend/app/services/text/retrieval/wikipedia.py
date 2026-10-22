"""
Wikipedia evidence source.

News search answers "is anyone reporting this?". Wikipedia answers a different
and complementary question: "what is the established background here?" -- who
holds an office, when an organisation was founded, what a place is called.
That makes it strong for durable factual claims and weak for breaking events,
where its articles lag or do not exist.

Free, no API key. Two calls: search for candidate pages, then fetch the plain
text introductions of the best matches.
"""

from __future__ import annotations

import json
import logging
from typing import List

from .. import config
from .base import Evidence, http_get

logger = logging.getLogger(__name__)

SOURCE_NAME = "Wikipedia"
_DOMAIN = "en.wikipedia.org"


def _search_titles(query: str, limit: int) -> List[str]:
    body = http_get(
        config.WIKIPEDIA_ENDPOINT,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "format": "json",
        },
        user_agent=config.WIKIPEDIA_USER_AGENT,
    )
    payload = json.loads(body)
    return [hit["title"] for hit in payload.get("query", {}).get("search", [])]


def _fetch_intros(titles: List[str]) -> dict:
    """Return {title: intro_text} for the given page titles."""
    if not titles:
        return {}
    body = http_get(
        config.WIKIPEDIA_ENDPOINT,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": "|".join(titles),
            "format": "json",
            "redirects": "1",
        },
        user_agent=config.WIKIPEDIA_USER_AGENT,
    )
    payload = json.loads(body)
    pages = payload.get("query", {}).get("pages", {}) or {}
    return {
        page.get("title", ""): (page.get("extract") or "").strip()
        for page in pages.values()
        if page.get("extract")
    }


def search(query: str, max_articles: int | None = None) -> List[Evidence]:
    """Return intro paragraphs of the best-matching Wikipedia articles."""
    if not query.strip():
        return []

    max_articles = (
        config.WIKIPEDIA_MAX_ARTICLES if max_articles is None else max_articles
    )

    titles = _search_titles(query, max_articles)
    intros = _fetch_intros(titles)

    results: List[Evidence] = []
    for title in titles:
        intro = intros.get(title, "")
        if not intro:
            continue
        results.append(
            Evidence(
                title=title,
                url=f"https://{_DOMAIN}/wiki/{title.replace(' ', '_')}",
                domain=_DOMAIN,
                source=SOURCE_NAME,
                # Only the first part of the intro: NLI premises are truncated
                # anyway and the opening sentences carry the definitional facts.
                snippet=intro[:800],
                published=None,  # revision date is not fetched; not a news item
                outlet="Wikipedia",
            )
        )

    logger.debug("Wikipedia returned %d articles for %r", len(results), query)
    return results
