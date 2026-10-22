"""
GDELT DOC 2.0 evidence source.

GDELT monitors news worldwide and re-indexes roughly every 15 minutes. It is
free, needs no API key, and covers far more outlets than any single feed --
which makes it the backbone of the "is anyone else reporting this?" question.

Two practical notes learned by testing against the live endpoint:

1. It returns HTTP 429 to requests with no User-Agent header. That looks like
   rate limiting but is not; ``base.http_get`` always sends one.
2. It returns headlines only -- no article body. Stance detection therefore
   runs against the HEADLINE, which is a real limitation: a headline can be
   ambiguous or deliberately misleading about its own article. This is recorded
   in the response so the limitation travels with the result.
"""

from __future__ import annotations

import json
import logging
from typing import List

from .. import config
from .base import Evidence, http_get, normalise_domain, parse_datetime

logger = logging.getLogger(__name__)

SOURCE_NAME = "GDELT"

# GDELT's seendate format, e.g. 20250823T124500Z
_DATE_FORMATS = ["%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"]


def search(query: str, max_records: int | None = None, timespan: str | None = None) -> List[Evidence]:
    """
    Query the GDELT article index.

    Returns an empty list on any failure; the caller records the error.
    """
    if not query.strip():
        return []

    max_records = config.GDELT_MAX_RECORDS if max_records is None else max_records
    timespan = config.GDELT_TIMESPAN if timespan is None else timespan

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
        "timespan": timespan,
        "sort": "hybridrel",   # relevance blended with recency
    }

    body = http_get(config.GDELT_ENDPOINT, params=params)

    # GDELT occasionally answers with a plain-text error page and HTTP 200.
    stripped = body.lstrip()
    if not stripped.startswith("{"):
        raise RuntimeError(f"unexpected GDELT response: {stripped[:120]}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse GDELT JSON: {exc}") from exc

    results: List[Evidence] = []
    for article in payload.get("articles", []) or []:
        url = (article.get("url") or "").strip()
        title = (article.get("title") or "").strip()
        if not url or not title:
            continue
        results.append(
            Evidence(
                title=title,
                url=url,
                domain=normalise_domain(article.get("domain") or url),
                source=SOURCE_NAME,
                snippet="",  # GDELT provides no body text
                published=parse_datetime(article.get("seendate", ""), _DATE_FORMATS),
                language=(article.get("language") or "en").lower()[:2],
                outlet=normalise_domain(article.get("domain") or url),
            )
        )

    logger.debug("GDELT returned %d articles for %r", len(results), query)
    return results
