"""
Turn a claim into search queries for the evidence sources.

Retrieval quality dominates the whole system: the NLI model can only judge
evidence that was actually found, so a bad query means an "Unverified" verdict
no matter how good the model is.

Strategy: keep the salient, searchable parts of the claim -- named entities,
quoted phrases, numbers and distinctive content words -- and drop the
grammatical scaffolding. Different backends want different syntax, so each gets
its own formatter over the same extracted terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .. import config
from ..textutils import (
    STOPWORDS,
    numbers_and_dates,
    proper_nouns,
    quoted_phrases,
    tokenize_words,
)

# Common news verbs carry little discriminative power in a search query.
_LOW_VALUE = {
    "said", "says", "reported", "according", "new", "news", "today",
    "yesterday", "week", "year", "people", "time", "way", "thing", "make",
    "made", "get", "got", "also", "after", "before", "more", "most",
}


@dataclass
class SearchQuery:
    """Extracted search terms plus per-backend renderings."""

    terms: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    def as_keyword_string(self) -> str:
        """Plain space-separated query (Google News RSS, Wikipedia)."""
        parts = [f'"{p}"' for p in self.phrases[:1]]
        parts.extend(self.terms)
        return " ".join(parts).strip()

    def as_gdelt_query(self) -> str:
        """
        GDELT DOC 2.0 syntax.

        Bare terms are ANDed. Too many ANDed terms returns nothing at all, so
        the term list is kept deliberately short and multi-word entities are
        quoted to survive as phrases.
        """
        def usable(term: str) -> bool:
            """
            GDELT rejects short or over-common bare keywords outright, failing
            the whole query: "D.C" produced 'One or more of your keywords were
            too short, too long or too common: (d.c)' and returned nothing.
            Quoted phrases are exempt from the length rule.
            """
            return len(re.sub(r"[^A-Za-z0-9]", "", term)) >= 3

        parts: List[str] = []
        for phrase in self.phrases[:1]:
            parts.append(f'"{phrase}"')
        for entity in self.entities[:3]:
            if " " in entity:
                parts.append(f'"{entity}"')
            elif usable(entity):
                parts.append(entity)
        # Hard cap: GDELT ANDs bare terms, so a long query matches nothing.
        limit = config.GDELT_MAX_QUERY_TERMS
        parts = parts[:limit]
        remaining = limit - len(parts)
        if remaining > 0:
            joined = " ".join(parts).lower()
            extras = [
                t for t in self.terms if t.lower() not in joined and usable(t)
            ]
            parts.extend(extras[:remaining])
        return " ".join(parts).strip()

    def as_reference_query(self) -> str:
        """
        Query for encyclopedic sources (Wikipedia).

        Reference search wants the SUBJECT, not the claim. Passing the whole
        claim dilutes it: "World Health Organization United Nations specialised
        agency" returned "List of specialized agencies of the UN" and "UN Office
        at Geneva" but never the WHO article itself, so the one page that would
        have settled the claim was never retrieved. Searching the single most
        specific entity finds the subject page instead.
        """
        if self.entities:
            return max(self.entities, key=len)
        return self.as_keyword_string()

    def to_dict(self) -> dict:
        return {
            "terms": self.terms,
            "phrases": self.phrases,
            "entities": self.entities,
            "keyword_query": self.as_keyword_string(),
            "gdelt_query": self.as_gdelt_query(),
            "reference_query": self.as_reference_query(),
        }


def _merge_adjacent_entities(text: str, entities: List[str]) -> List[str]:
    """Join capitalised runs so 'New York Times' stays one entity, not three."""
    if not entities:
        return []
    merged: List[str] = []
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(e) for e in entities) + r")"
        r"(?:\s+(?:" + "|".join(re.escape(e) for e in entities) + r"))*\b"
    )
    for match in pattern.finditer(text):
        candidate = match.group(0).strip()
        if candidate and candidate not in merged:
            merged.append(candidate)
    return merged or entities


def build_query(claim_text: str, max_terms: int | None = None) -> SearchQuery:
    """Extract the searchable core of a claim."""
    max_terms = config.MAX_QUERY_TERMS if max_terms is None else max_terms

    phrases = quoted_phrases(claim_text)
    entities = _merge_adjacent_entities(claim_text, proper_nouns(claim_text))
    figures = numbers_and_dates(claim_text)

    # Content words: not stopwords, not low-value news filler, length >= 4.
    entity_words = {w.lower() for e in entities for w in tokenize_words(e)}
    content: List[str] = []
    for word in tokenize_words(claim_text):
        lowered = word.lower()
        if lowered in STOPWORDS or lowered in _LOW_VALUE or lowered in entity_words:
            continue
        if len(lowered) < 4 or lowered.isdigit():
            continue
        if lowered not in content:
            content.append(lowered)

    # Priority order: entities, then years/figures, then content words.
    terms: List[str] = []
    terms.extend(entities[:3])
    terms.extend([f for f in figures if len(f) == 4 and f.startswith(("19", "20"))][:1])
    terms.extend(content[: max(0, max_terms - len(terms))])

    return SearchQuery(terms=terms[:max_terms], phrases=phrases, entities=entities)
