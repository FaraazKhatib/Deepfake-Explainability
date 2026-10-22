"""
Debunk detection and reported-speech detection.

Both exist because of a limitation found by testing the live system against a
known fake-news example (Pizzagate), not anticipated in the design.

THE PROBLEM: NLI DOES NOT UNDERSTAND DEBUNKING
----------------------------------------------
An NLI model is trained on direct semantic contradiction. Measured against
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` with the claim "Comet Ping Pong
was connected to a child sex trafficking ring":

    premise                                              contradiction
    "Comet Ping Pong was NOT connected to any ring"          0.997  correct
    "There is no evidence... The claim is false."            0.099  missed
    Wikipedia: "Pizzagate ... falsely claiming that ..."     0.006  missed

A source saying "this claim is false" is making a statement *about* the claim
rather than asserting its negation, and NLI treats that as neutral. The
consequence is severe: the system misses exactly the claims that professional
fact-checkers have already debunked -- the highest-value case for the whole
module.

The fix is a transparent lexical detector, deliberately NOT another neural
model, so that it fails independently of the NLI model rather than in
correlated ways. It fires only when BOTH hold:

  * the evidence contains explicit debunking language, AND
  * the evidence overlaps strongly with the claim's subject matter.

THE SECOND PROBLEM: REPORTED SPEECH
-----------------------------------
"In 2016, a story circulated that Comet Ping Pong was connected to a
trafficking ring" is a TRUE statement about a FALSE story. Fact-checking it as
though it asserted the trafficking ring would be wrong.

So reported-speech framing is detected on the claim, and when present the
debunk override is suppressed and the ambiguity is reported instead. The system
checks the claim as written; it should say so rather than quietly checking a
different claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .. import config
from ..retrieval.relevance import relevance_overlap

# Explicit debunking language. Each entry is a regex fragment.
_DEBUNK_PATTERNS = [
    r"\bfalsely (?:claim|claimed|claiming|stated|reported)\b",
    r"\bdebunk(?:ed|ing|s)?\b",
    r"\bno evidence\b",
    r"\bwithout evidence\b",
    r"\blacks? evidence\b",
    r"\bconspiracy theory\b",
    r"\bhoax\b",
    r"\bmisinformation\b",
    r"\bdisinformation\b",
    r"\bbaseless\b",
    r"\bunfounded\b",
    r"\bunsubstantiated\b",
    r"\bfabricated\b",
    r"\bdiscredited\b",
    r"\bdisproven\b",
    r"\bis (?:not true|false|untrue|incorrect)\b",
    r"\bfake news\b",
    r"\bmyth\b",
    r"\bmisleading\b",
    r"\brating:? false\b",
    r"\bpants on fire\b",
]

_COMPILED_DEBUNK = [re.compile(p, re.IGNORECASE) for p in _DEBUNK_PATTERNS]

# Framing that makes a claim a statement ABOUT an assertion, not the assertion.
_REPORTED_SPEECH_PATTERNS = [
    r"\ba (?:story|rumou?r|claim|report|theory|post|article|video) (?:circulated|spread|went viral|emerged|surfaced|claimed)\b",
    r"\b(?:stories|rumou?rs|claims|reports|posts) (?:circulated|spread|went viral|emerged|surfaced)\b",
    r"\bit was (?:claimed|alleged|reported|rumou?red|suggested)\b",
    r"\b(?:people|users|posts|some) (?:claimed|alleged|shared|said)\b",
    r"\baccording to (?:a|the) (?:rumou?r|conspiracy|viral|unverified)\b",
    r"\bconspiracy theory (?:that|claiming|alleging)\b",
    r"\bwas accused of\b",
    r"\ballegedly\b",
    r"\bthere (?:were|was) (?:claims|reports|rumou?rs)\b",
    r"\bwidely (?:shared|circulated|spread) (?:claim|story|post)\b",
]

_COMPILED_REPORTED = [re.compile(p, re.IGNORECASE) for p in _REPORTED_SPEECH_PATTERNS]


@dataclass
class DebunkSignal:
    """A lexical debunk detection against one piece of evidence."""

    detected: bool
    matched_terms: List[str]
    overlap: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "matched_terms": self.matched_terms,
            "topical_overlap": round(self.overlap, 4),
            "confidence": round(self.confidence, 4),
        }


def detect_reported_speech(claim: str) -> Optional[str]:
    """
    Return the framing phrase when a claim reports an assertion rather than
    making one, else None.

    "A story circulated that X" can be perfectly true while X is false, so the
    two must not be conflated.
    """
    for pattern in _COMPILED_REPORTED:
        match = pattern.search(claim or "")
        if match:
            return match.group(0)
    return None


def find_debunk_terms(text: str) -> List[str]:
    """Every distinct debunking expression present in the text."""
    found: List[str] = []
    for pattern in _COMPILED_DEBUNK:
        match = pattern.search(text or "")
        if match:
            term = match.group(0).lower()
            if term not in found:
                found.append(term)
    return found


def assess_debunk(claim: str, evidence_text: str) -> DebunkSignal:
    """
    Decide whether ``evidence_text`` is debunking ``claim``.

    Requires strong topical overlap as well as debunking language: an article
    debunking a DIFFERENT claim about the same people must not be read as
    refuting this one. The bar is deliberately higher than the ordinary
    relevance gate.
    """
    terms = find_debunk_terms(evidence_text)
    overlap = relevance_overlap(claim, evidence_text)

    if not terms or overlap < config.DEBUNK_MIN_OVERLAP:
        return DebunkSignal(False, terms, overlap, 0.0)

    # Confidence grows with how much debunking language is present and how
    # closely the evidence tracks the claim, but is capped: this is a lexical
    # heuristic and must never look as certain as a direct model contradiction.
    strength = min(1.0, len(terms) / 2.0)
    confidence = min(
        config.DEBUNK_MAX_CONFIDENCE,
        config.DEBUNK_BASE_CONFIDENCE + 0.15 * strength + 0.15 * (overlap - config.DEBUNK_MIN_OVERLAP),
    )
    return DebunkSignal(True, terms, overlap, confidence)


@dataclass
class OverrideResult:
    """Outcome of applying the debunk override to one NLI judgement."""

    stance: str
    contradiction: float
    fired: bool
    terms: List[str]
    suppressed_by_framing: bool


def apply_debunk_override(
    stance: str,
    contradiction: float,
    claim: str,
    evidence_text: str,
    reported_framing: Optional[str],
    refutes_label: str = "refutes",
) -> OverrideResult:
    """
    Promote an NLI judgement to "refutes" when the evidence debunks the claim.

    Suppressed entirely when the claim is reported speech: "a story circulated
    that X" is true whenever the story circulated, so debunking X does not
    refute the sentence actually submitted.
    """
    if not config.ENABLE_DEBUNK_DETECTION:
        return OverrideResult(stance, contradiction, False, [], False)

    if reported_framing is not None:
        return OverrideResult(stance, contradiction, False, [], True)

    if stance == refutes_label:
        return OverrideResult(stance, contradiction, False, [], False)
    # A keyword is not a semantic contradiction. In particular, a fact-check
    # confirming a claim must never be inverted just for discussing misinformation.
    if stance == "supports":
        return OverrideResult(stance, contradiction, False, [], False)

    signal = assess_debunk(claim, evidence_text)
    if not signal.detected:
        return OverrideResult(stance, contradiction, False, signal.matched_terms, False)

    # The NLI contradiction probability is low by construction here, so the
    # lexical confidence replaces it for weighting purposes.
    return OverrideResult(
        refutes_label,
        max(contradiction, signal.confidence),
        True,
        signal.matched_terms,
        False,
    )
