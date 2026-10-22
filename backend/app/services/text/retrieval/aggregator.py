"""
Run every enabled evidence source concurrently, then deduplicate.

Deduplication matters more here than it looks. Wire copy is syndicated: one
Reuters story appears verbatim on fifty sites. Counting those as fifty
independent corroborations would turn a single source into an overwhelming
consensus, which is exactly the failure mode that makes naive "many outlets
agree" systems worthless. Two defences:

1. near-duplicate titles and identical canonical URLs collapse into one item;
2. a per-domain cap stops any single outlet dominating the pool.

The verdict layer applies the more important defence -- collapsing to one
stance per domain -- in ``scoring/verdict.py``.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Tuple

from .. import config
from ..claims.query_builder import SearchQuery
from .base import Evidence, RetrievalReport, canonical_url
from .relevance import filter_by_language
from . import gdelt, news_rss, wikipedia

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _title_fingerprint(title: str) -> str:
    """
    Order-insensitive bag of significant tokens, for near-duplicate detection.

    Numbers are kept regardless of length: "5 killed in crash" and
    "50 killed in crash" are different stories, and dropping short numeric
    tokens would silently merge them.
    """
    tokens = _WORD_RE.findall(title.lower())
    significant = sorted(set(t for t in tokens if len(t) > 3 or t.isdigit()))
    return " ".join(significant[:12])


def deduplicate(items: List[Evidence], max_per_domain: int) -> List[Evidence]:
    """Collapse duplicates and enforce the per-domain cap, preserving order."""
    seen_urls: set = set()
    seen_titles: set = set()
    per_domain: Dict[str, int] = {}
    kept: List[Evidence] = []

    for item in items:
        url_key = canonical_url(item.url)
        if url_key and url_key in seen_urls:
            continue
        title_key = _title_fingerprint(item.title)
        if title_key and title_key in seen_titles:
            continue
        if per_domain.get(item.domain, 0) >= max_per_domain:
            continue

        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        per_domain[item.domain] = per_domain.get(item.domain, 0) + 1
        kept.append(item)

    return kept


def gather_evidence(query: SearchQuery, max_items: int | None = None) -> RetrievalReport:
    """
    Query all enabled sources in parallel and return a merged, deduplicated pool.

    Sources run concurrently because they are independent network calls; total
    latency is that of the slowest source rather than their sum. A source that
    raises is recorded in ``report.errors`` and contributes nothing.
    """
    max_items = config.MAX_EVIDENCE_ITEMS if max_items is None else max_items
    report = RetrievalReport()

    keyword_query = query.as_keyword_string()
    gdelt_query = query.as_gdelt_query()
    reference_query = query.as_reference_query()

    tasks: List[Tuple[str, Callable[[], List[Evidence]], str]] = []
    if config.ENABLE_GDELT:
        tasks.append((gdelt.SOURCE_NAME, lambda: gdelt.search(gdelt_query), gdelt_query))
    if config.ENABLE_GOOGLE_NEWS:
        tasks.append(
            (news_rss.SOURCE_NAME, lambda: news_rss.search(keyword_query), keyword_query)
        )
    if config.ENABLE_WIKIPEDIA:
        tasks.append(
            (
                wikipedia.SOURCE_NAME,
                lambda: wikipedia.search(reference_query),
                reference_query,
            )
        )

    if not tasks:
        report.errors["configuration"] = "No evidence source is enabled."
        return report

    collected: List[Evidence] = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {}
        for name, function, used_query in tasks:
            report.queries_used[name] = used_query
            futures[pool.submit(function)] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                items = future.result()
                collected.extend(items)
                report.per_source_counts[name] = len(items)
            except Exception as exc:
                report.per_source_counts[name] = 0
                report.errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("Evidence source %s failed: %s", name, exc)

    # Interleave sources before capping so one backend cannot fill the budget.
    by_source: Dict[str, List[Evidence]] = {}
    for item in collected:
        by_source.setdefault(item.source, []).append(item)

    interleaved: List[Evidence] = []
    position = 0
    while any(position < len(items) for items in by_source.values()):
        for items in by_source.values():
            if position < len(items):
                interleaved.append(items[position])
        position += 1

    # Drop non-English evidence before it can be mis-scored by an
    # English-trained NLI model (see retrieval/relevance.py).
    kept, dropped = filter_by_language(interleaved)
    if dropped:
        report.notices.setdefault(
            "language_filter",
            f"{len(dropped)} non-English item(s) were excluded; the NLI model "
            "is English-trained and would score them unreliably.",
        )

    report.evidence = deduplicate(kept, config.MAX_PER_DOMAIN)[:max_items]
    logger.info(
        "Retrieved %d evidence items (%s)",
        len(report.evidence),
        ", ".join(f"{k}={v}" for k, v in report.per_source_counts.items()) or "none",
    )
    return report
