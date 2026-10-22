"""
Guards applied to evidence BEFORE it reaches the NLI model.

Both of these were added in response to failures observed against the live
endpoints, not anticipated in advance:

LANGUAGE FILTER
    A query about the Eiffel Tower returned Greek, Turkish and Russian
    Euronews articles. The NLI model is English-trained, so it scored them
    confidently and wrongly, and those scores went on to vote in the verdict.
    Non-English evidence is now dropped rather than mis-scored.

RELEVANCE GATE
    An NLI model given a premise that has nothing to do with the hypothesis
    should answer "neutral". In practice it frequently returns a confident
    contradiction: an article about European night trains "contradicted" the
    claim "The Eiffel Tower is located in Berlin" at 0.98 confidence. Requiring
    lexical overlap between claim and evidence before a stance may count is a
    cheap, transparent guard against that failure mode.

The gate is deliberately lexical rather than semantic. A second neural model
to check relevance would fail in correlated ways with the first; a word-overlap
rule fails independently and is auditable by a human reading the response.
"""

from __future__ import annotations

import logging
import re
from typing import List, Set

from .. import config
from ..textutils import STOPWORDS, tokenize_words
from .base import Evidence

logger = logging.getLogger(__name__)

# Frequent English function words, used as a cheap language signal.
_ENGLISH_MARKERS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "has", "have", "had", "will", "would",
    "that", "this", "it", "as", "at", "by", "from", "after", "says", "said",
    "not", "but", "be", "been", "new", "more", "over", "into", "about",
}


def non_ascii_letter_ratio(text: str) -> float:
    """Share of alphabetic characters outside the ASCII range."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    return non_ascii / len(letters)


def looks_english(text: str) -> bool:
    """
    Cheap language check: script plus function-word density.

    Catches Greek, Cyrillic, Arabic, CJK and similar outright via script.
    Latin-script languages (Turkish, Spanish, Indonesian) are caught by the
    absence of English function words. Not a language identifier -- short
    headlines are genuinely ambiguous -- so it errs toward keeping text.
    """
    if not text or not text.strip():
        return False

    if non_ascii_letter_ratio(text) > config.NON_ASCII_LETTER_LIMIT:
        return False

    words = [w.lower() for w in tokenize_words(text)]
    if len(words) < 4:
        return True  # too short to judge; keep it and let the gate decide

    markers = sum(1 for w in words if w in _ENGLISH_MARKERS)
    return markers / len(words) >= 0.10


def filter_by_language(items: List[Evidence]) -> tuple:
    """Split evidence into (kept, dropped) by language."""
    if not config.FILTER_NON_ENGLISH:
        return list(items), []

    kept, dropped = [], []
    for item in items:
        declared = (item.language or "en").lower()[:2]
        if declared and declared != "en":
            dropped.append(item)
            continue
        (kept if looks_english(item.premise) else dropped).append(item)

    if dropped:
        logger.info("Dropped %d non-English evidence item(s)", len(dropped))
    return kept, dropped


# --------------------------------------------------------------------------
def salient_terms(text: str) -> Set[str]:
    """Content words and numbers that carry the meaning of a claim."""
    terms: Set[str] = set()
    for word in tokenize_words(text):
        lowered = word.lower()
        if lowered in STOPWORDS:
            continue
        if lowered.isdigit() or len(lowered) >= 4:
            terms.add(lowered)
    return terms


def relevance_overlap(claim: str, evidence_text: str) -> float:
    """
    Fraction of the claim's salient terms that appear in the evidence.

    Normalised by a capped term count so that a long claim is not penalised
    for having many terms an accurate snippet would never repeat verbatim.
    """
    claim_terms = salient_terms(claim)
    if not claim_terms:
        return 1.0  # nothing to match on; do not block
    evidence_terms = salient_terms(evidence_text)
    shared = claim_terms & evidence_terms
    denominator = min(len(claim_terms), config.RELEVANCE_TERM_CAP)
    return len(shared) / denominator if denominator else 0.0


def is_relevant(claim: str, evidence_text: str) -> tuple:
    """Return (passes_gate, overlap_score)."""
    overlap = relevance_overlap(claim, evidence_text)
    terms = salient_terms(claim)
    shared = terms & salient_terms(evidence_text)
    # A single generic word (e.g. "moon") cannot establish that a headline
    # addresses a multi-term claim. At the 0.25 overlap threshold it previously
    # admitted Venus/Mars headlines as contradictions about Earth's Moon.
    # Keep the fractional gate for longer claims, and allow genuinely one-term
    # claims through. Paraphrases may abstain; this is a lexical guard, not NER.
    enough_terms = len(shared) >= min(2, len(terms))
    return enough_terms and overlap >= config.MIN_RELEVANCE_OVERLAP, overlap
