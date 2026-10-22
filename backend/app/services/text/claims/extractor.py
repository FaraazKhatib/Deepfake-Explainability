"""
Check-worthy claim extraction.

Given a headline, a paragraph or a full article, decide WHAT to fact-check.
Not every sentence is checkable: opinions, questions, hedged speculation and
transitional filler cannot be verified against evidence, and wasting retrieval
budget on them buries the claims that matter.

Scoring is heuristic and transparent -- each sentence accumulates points for
properties that make a statement verifiable, and loses points for properties
that make it unverifiable. Every reason is recorded so the API can show the
user why a particular sentence was chosen.

This is a regex heuristic, not a trained check-worthiness model (ClaimBuster
and similar exist but need training data we do not have). It is good enough to
pick the factual backbone out of a news paragraph, and it fails visibly rather
than silently: the claims it selected are always returned in the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .. import config
from ..exceptions import InputValidationError, NoClaimFoundError
from ..textutils import (
    clean_text,
    numbers_and_dates,
    proper_nouns,
    quoted_phrases,
    split_sentences,
    tokenize_words,
)

# Verbs that report or assert a fact about the world.
_ASSERTIVE_VERBS = {
    "said", "says", "announced", "confirmed", "reported", "revealed",
    "stated", "declared", "admitted", "denied", "claimed", "warned",
    "found", "showed", "proved", "ruled", "voted", "approved", "rejected",
    "banned", "launched", "signed", "won", "lost", "killed", "died",
    "arrested", "charged", "sentenced", "resigned", "elected", "appointed",
    "acquired", "raised", "fell", "rose", "increased", "decreased",
    "reached", "passed", "failed", "agreed", "refused", "issued", "ordered",
    "is", "was", "are", "were", "has", "have", "had", "will", "became",
}

# Markers of opinion rather than fact.
_OPINION_MARKERS = (
    "i think", "i believe", "in my opinion", "i feel", "we should",
    "you should", "arguably", "it seems to me", "personally", "i reckon",
    "one could argue", "in my view",
)

# Hedges: still checkable, but weaker.
_HEDGE_MARKERS = (
    "may", "might", "could", "possibly", "perhaps", "reportedly",
    "allegedly", "rumoured", "rumored", "speculation", "unconfirmed",
)

_SUPERLATIVES = (
    "first", "largest", "biggest", "smallest", "highest", "lowest", "worst",
    "best", "record", "unprecedented", "never before", "only",
)


@dataclass
class Claim:
    """A single statement selected for verification."""

    text: str
    score: float
    reasons: List[str] = field(default_factory=list)
    position: int = 0

    def to_dict(self) -> dict:
        return {
            "claim": self.text,
            "checkworthiness": round(self.score, 3),
            "selected_because": self.reasons,
            "sentence_index": self.position,
        }


def score_sentence(sentence: str) -> tuple:
    """Return (score, reasons) for one sentence. Higher = more checkable."""
    lowered = sentence.lower()
    words = tokenize_words(sentence)
    score = 0.0
    reasons: List[str] = []

    if len(words) < config.MIN_CLAIM_WORDS:
        return 0.0, ["too short to be a standalone claim"]
    if len(words) > config.MAX_CLAIM_WORDS:
        score -= 0.3
        reasons.append("unusually long sentence")

    if sentence.rstrip().endswith("?"):
        return 0.0, ["a question, not an assertion"]

    for marker in _OPINION_MARKERS:
        if marker in lowered:
            return 0.0, [f"opinion marker: '{marker}'"]

    entities = proper_nouns(sentence)
    if entities:
        score += 0.9 + 0.1 * min(len(entities), 3)
        reasons.append(f"names specific entities ({', '.join(entities[:3])})")

    figures = numbers_and_dates(sentence)
    if figures:
        score += 0.7
        reasons.append(f"contains checkable figures ({', '.join(figures[:3])})")

    verbs = [w for w in (w.lower() for w in words) if w in _ASSERTIVE_VERBS]
    if verbs:
        score += 0.5
        reasons.append(f"asserts an event or state ('{verbs[0]}')")

    if quoted_phrases(sentence):
        score += 0.4
        reasons.append("contains a direct quotation")

    if any(s in lowered for s in _SUPERLATIVES):
        score += 0.3
        reasons.append("makes a superlative or record claim")

    hedges = [h for h in _HEDGE_MARKERS if re.search(rf"\b{re.escape(h)}\b", lowered)]
    if hedges:
        score -= 0.25
        reasons.append(f"hedged language ('{hedges[0]}') weakens verifiability")

    return max(0.0, score), reasons


def extract_claims(
    raw_text: str,
    max_claims: int | None = None,
    treat_as_single_claim: bool = False,
) -> List[Claim]:
    """
    Pick the most check-worthy statements from ``raw_text``.

    Parameters
    ----------
    treat_as_single_claim
        Force the entire input to be treated as one claim. Use this for a
        headline or a user-typed assertion, where splitting is unhelpful.

    Raises
    ------
    InputValidationError
        Empty input, or input beyond the configured size limit.
    NoClaimFoundError
        Nothing verifiable could be found (e.g. pure opinion or questions).
    """
    max_claims = config.MAX_CLAIMS if max_claims is None else max_claims

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise InputValidationError("No text supplied.")
    if len(raw_text) > config.MAX_INPUT_CHARS:
        raise InputValidationError(
            f"Input is too long ({len(raw_text)} characters, limit "
            f"{config.MAX_INPUT_CHARS}).",
            character_count=len(raw_text),
            limit=config.MAX_INPUT_CHARS,
        )

    text = clean_text(raw_text)
    words = tokenize_words(text)
    if len(words) < config.MIN_INPUT_WORDS:
        raise InputValidationError(
            f"Input is too short to check: {len(words)} words "
            f"(minimum {config.MIN_INPUT_WORDS}).",
            word_count=len(words),
            minimum=config.MIN_INPUT_WORDS,
        )

    sentences = split_sentences(text)

    # A single short sentence (a headline, a typed assertion) is the claim.
    if treat_as_single_claim or len(sentences) <= 1:
        score, reasons = score_sentence(text)
        if score <= 0:
            raise NoClaimFoundError("Please enter a factual assertion, rather than a question or opinion.")
        if len(text) > 400:
            raise InputValidationError("A single claim must be at most 400 characters; submit separate claims instead.")
        return [
            Claim(
                text=text,
                score=score,
                reasons=reasons,
                position=0,
            )
        ]

    scored: List[Claim] = []
    for index, sentence in enumerate(sentences):
        score, reasons = score_sentence(sentence)
        if score > 0:
            scored.append(Claim(text=sentence, score=score, reasons=reasons, position=index))

    if not scored:
        raise NoClaimFoundError(
            "No verifiable factual claim was found. The text appears to be "
            "opinion, questions, or too vague to check against evidence.",
            sentences_examined=len(sentences),
        )

    scored.sort(key=lambda c: (-c.score, c.position))
    selected = scored[:max_claims]
    selected.sort(key=lambda c: c.position)  # restore reading order
    return selected
