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
    "occurred", "happened", "imploded", "suspended", "broke", "crashed", "survived",
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

# Pronouns that indicate a sentence lost its subject. When a sentence contains
# pronouns but no proper nouns, the headline's subject should be injected so
# that the search query actually finds evidence about the right person/entity.
_SUBJECT_PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "its"}

# Vague / editorial words that make a sentence bad for fact-checking queries.
# Headlines like "dark reminder of gangster past" are commentary, not claims.
_EDITORIAL_MARKERS = (
    "reminder", "symbol", "sign of", "wake-up call", "turning point",
    "watershed", "harbinger", "echo of", "shades of", "throwback",
    "reflection of", "cautionary tale", "litmus test", "hallmark",
    "shaken", "in shock", "shocked", "left reeling", "reeling from", "rocked by",
)

# Common web page boilerplate / cookie banners to discard
_BOILERPLATE_PATTERNS = (
    re.compile(r"\b(?:allow|accept|manage)\s+(?:all\s+)?cookies\b", re.I),
    re.compile(r"\b(?:use\s+of\s+cookies|cookie\s+policy|cookie\s+settings)\b", re.I),
    re.compile(r"\bby\s+clicking\s+.*?\bcookies\b", re.I),
    re.compile(r"\bprivacy\s+policy\b", re.I),
    re.compile(r"\bterms\s+of\s+(?:service|use)\b", re.I),
    re.compile(r"\ball\s+rights\s+reserved\b", re.I),
    re.compile(r"\bsubscribe\s+to\s+(?:our\s+)?newsletter\b", re.I),
    re.compile(r"\bsign\s+in\s+to\s+(?:continue|read)\b", re.I),
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


def score_sentence(sentence: str, position: int = -1, subject: str = "") -> tuple:
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

    for bp in _BOILERPLATE_PATTERNS:
        if bp.search(lowered):
            return 0.0, ["website UI boilerplate (cookie / privacy / terms notice)"]

    for marker in _OPINION_MARKERS:
        if marker in lowered:
            return 0.0, [f"opinion marker: '{marker}'"]

    entities = proper_nouns(sentence)
    if entities:
        score += 0.9 + 0.1 * min(len(entities), 3)
        reasons.append(f"names specific entities ({', '.join(entities[:3])})")

    # Central subject match (e.g. if the sentence directly names the headline's subject or surname)
    if subject:
        sub_tokens = [w.lower() for w in tokenize_words(subject)]
        if any(re.search(rf"\b{re.escape(st)}\b", lowered) for st in sub_tokens if len(st) >= 3):
            score += 0.5
            reasons.append(f"names central subject '{subject}'")

    figures = numbers_and_dates(sentence)
    if figures:
        # If the number is an incidental age pattern (e.g. "36-year-old"), award modest score
        if re.search(r"\b\d{1,2}-year-old\b", lowered):
            score += 0.3
            reasons.append("contains age")
        else:
            score += 0.7
            reasons.append(f"contains checkable figures ({', '.join(figures[:3])})")

    verbs = [w for w in (w.lower() for w in words) if w in _ASSERTIVE_VERBS]
    if verbs:
        score += 0.4
        reasons.append(f"asserts an event or state ('{verbs[0]}')")

    if quoted_phrases(sentence):
        score += 0.4
        reasons.append("contains a direct quotation")

    if any(s in lowered for s in _SUPERLATIVES):
        score += 0.3
        reasons.append("makes a superlative or record claim")

    # Penalize editorial / metaphor language. These produce poor search
    # queries because they describe interpretation, not events.
    if any(m in lowered for m in _EDITORIAL_MARKERS):
        score -= 0.8
        reasons.append("editorial or metaphorical language (less verifiable)")

    # Boost sentences that contain high-value action verbs -- these describe
    # concrete events (shot, killed, arrested, survived) rather than filler.
    from .query_builder import _HIGH_VALUE_VERBS
    action_hits = [w for w in (w.lower() for w in words) if w in _HIGH_VALUE_VERBS]
    if action_hits:
        distinct = set(action_hits)
        action_score = 0.6 + 0.3 * min(len(distinct) - 1, 2)
        score += action_score
        reasons.append(f"describes concrete event(s): {', '.join(list(distinct)[:3])}")

    # Penalize sentences lacking any assertive or action verb (e.g. metadata list labels or prepositional fragments)
    if not (verbs or action_hits):
        score -= 0.8
        reasons.append("lacks an assertive or action verb (fragment)")

    # Inverted pyramid lede boost: sentences near top of article carry the core news event
    if 1 <= position <= 4:
        score += 0.5
        reasons.append("lead paragraph lede (core news event)")

    # Penalize sentences dominated by pronouns without proper nouns.
    # These lose their subject when checked in isolation, causing the search
    # query to miss (e.g. "he was shot" instead of "Tiger Shroff was shot").
    word_lowers = {w.lower() for w in words}
    has_pronouns = bool(word_lowers & _SUBJECT_PRONOUNS)
    if has_pronouns and not entities:
        score -= 0.4
        reasons.append("uses pronouns without naming the subject")

    hedges = [h for h in _HEDGE_MARKERS if re.search(rf"\b{re.escape(h)}\b", lowered)]
    if hedges:
        score -= 0.25
        reasons.append(f"hedged language ('{hedges[0]}') weakens verifiability")

    return max(0.0, score), reasons


def clean_headline_text(text: str) -> str:
    """Strip topic category slugs, bullet points, metadata labels, and trailing clickbait rhetorical phrases."""
    cleaned = text.strip()
    # Strip leading bullet markers, dashes, or numbered list indicators (e.g. "• ", "* ", "- ", "1. ")
    cleaned = re.sub(r"^[\s•*–—\-\d.]+\s*", "", cleaned).strip()
    # Strip common list labels (e.g. "Location: ", "Date: ", "Company Status: ", "Status: ")
    cleaned = re.sub(
        r"^(?:Location|Date|Company Status|Status|Victims?|Overview|Key Findings?|Background):\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Strip topic slug prefix (e.g. "Tiger Shroff Car Attack Case: ...")
    cleaned = re.sub(
        r"^[A-Za-z0-9\s]+(?:Case|News|Update|Attack|Firing|Alert|Scandal):\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    # Strip trailing clickbait questions or appendages (e.g. ", what happened next?, police were confused!")
    cleaned = re.sub(
        r",\s*(?:what\s+happened\s+next|who\s+was\s+involved|know\s+details|read\s+here|see\s+details|watch\s+video|details\s+inside)[?!,].*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r",\s*what\s+happened\s+next\?.*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def split_compound_headline(headline: str) -> List[str]:
    """
    Decompose compound headlines joined by semicolons or conjunctions (' as ', ' while ', ' whilst ', ' amidst ', ' when ').
    Also cleans topic slugs, bullet markers, and trailing clickbait appendages.
    """
    cleaned = clean_headline_text(headline)

    # 1. Semicolon decomposition (e.g. "Accused practice in hotel before attack on Tiger Shroff; bullet accidentally discharged")
    if ";" in cleaned:
        parts = [clean_headline_text(p) for p in cleaned.split(";") if len(p.split()) >= 3]
        if len(parts) >= 2:
            return parts

    # 2. Conjunction decomposition (e.g. "Haaland Rescues Man City as Arsenal Hold On in Champions League",
    # "Jet may have broken apart before Florida crash, officials say, as family named among victims",
    # or "The OceanGate Titan submersible disaster occurred on June 18, 2023, when the vessel catastrophically imploded...")
    pattern = r"^(.*?\b[A-Za-z0-9]+)[,\s]+(?:as|while|whilst|amidst|when)\s+([a-zA-Z0-9].*)$"
    m = re.match(pattern, cleaned.strip().rstrip("."))
    if not m:
        return [cleaned]
    c1, c2 = m.group(1).strip(), m.group(2).strip()
    if len(c1.split()) < 3 or len(c2.split()) < 3:
        return [cleaned]
    prep_match = re.search(r"\s+(in|at|during|for|on)\s+([A-Z][A-Za-z0-9\s]+)$", c2)
    if prep_match:
        prep_phrase = prep_match.group(0)
        prep_words = prep_phrase.lower().split()
        if not any(w in c1.lower().split() for w in prep_words[1:]):
            c1_with_prep = f"{c1}{prep_phrase}"
        else:
            c1_with_prep = c1
    else:
        c1_with_prep = c1

    # If c2 refers to the primary subject with a generic descriptor ("the experimental vessel", "the car", etc.),
    # resolve it back to the subject named in c1 so c2 is a standalone factual assertion.
    from ..textutils import proper_nouns
    c1_subject = _get_primary_subject(c1)
    if not c1_subject:
        c1_nouns = [n for n in proper_nouns(c1)]
        c1_subject = c1_nouns[0] if c1_nouns else ""
    if c1_subject:
        resolved_subject = (
            f"The {c1_subject} submersible"
            if ("titan" in c1_subject.lower() and "submersible" not in c1_subject.lower())
            else f"The {c1_subject}"
        )
        c2 = re.sub(
            r"^the\s+(?:experimental\s+)?(?:carbon-fiber\s+)?(?:vessel|submersible|submarine|craft|vehicle|car|plane|jet|flight)\b",
            resolved_subject,
            c2,
            flags=re.IGNORECASE,
        )

    # If c2 has no named entities/locations but c1 established the event/location context,
    # transfer that context so c2 can be verified independently if selected.
    if not proper_nouns(c2):
        event_match = re.search(
            r"\b(?:in|at|before|after|during)\s+([A-Z][A-Za-z0-9\s]+(?:\bcrash|\battack|\bblast|\bincident|\baccident|\bfire|\bcase|\bshooting|\bplane)?)\b",
            c1,
        )
        if event_match:
            c2_with_prep = f"{c2} in {event_match.group(1).strip()}"
        else:
            c2_with_prep = c2
    else:
        c2_with_prep = c2

    return [c1_with_prep, c2_with_prep]


def _get_primary_subject(headline: str) -> str:
    """Extract the central named subject from the headline."""
    cleaned_hl = clean_headline_text(headline)
    from .query_builder import _merge_adjacent_entities, _HIGH_VALUE_VERBS
    all_verbs = set(_ASSERTIVE_VERBS) | set(_HIGH_VALUE_VERBS) | {
        "rescues", "rescue", "hold", "holds", "held", "beats", "beat", "cruises", "cruise",
        "scores", "scored", "scoring", "urges", "urged", "tells", "told", "settles", "settled",
        "survives", "survive", "clinch", "clinches", "clinched", "stays", "stayed"
    }
    raw_nouns = [n for n in proper_nouns(cleaned_hl) if n.lower() not in all_verbs]
    entities = _merge_adjacent_entities(cleaned_hl, raw_nouns)
    cleaned = [
        e for e in entities
        if not (e.lower() in _HIGH_VALUE_VERBS and cleaned_hl.strip().startswith(e))
    ]
    # Multi-word proper nouns (e.g. 'Tiger Shroff', 'Joe Biden') take precedence
    multi = [e for e in cleaned if " " in e]
    if multi:
        return multi[0]
    return cleaned[0] if cleaned else ""


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

    raw_sentences = split_sentences(text)
    if not raw_sentences:
        return []

    # Decompose compound headline if conjoining two distinct clauses
    sentences: List[str] = []
    headline_parts = split_compound_headline(raw_sentences[0])
    sentences.extend(headline_parts)
    for s in raw_sentences[1:]:
        cleaned_s = clean_headline_text(s)
        if cleaned_s:
            sentences.append(cleaned_s)

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

    headline = raw_sentences[0]
    subject = _get_primary_subject(headline)

    scored: List[Claim] = []
    for index, sentence in enumerate(sentences):
        score, reasons = score_sentence(sentence, position=index, subject=subject)
        if score > 0:
            scored.append(Claim(text=sentence, score=score, reasons=reasons, position=index))

    if not scored:
        raise NoClaimFoundError(
            "No verifiable factual claim was found. The text appears to be "
            "opinion, questions, or too vague to check against evidence.",
            sentences_examined=len(sentences),
        )

    # If the headline (position 0) is a factual assertion without editorial commentary,
    # give it priority. If it contains editorial markers (e.g. "dark reminder..."),
    # do NOT force it to displace concrete event claims from the body.
    first_claim = next((c for c in scored if c.position == 0), None)
    is_editorial_headline = first_claim and any("editorial" in r for r in first_claim.reasons)

    if first_claim is not None and not is_editorial_headline and first_claim.score >= 1.0:
        other_claims = [c for c in scored if c.position != 0]
        other_claims.sort(key=lambda c: (-c.score, c.position))
        selected = [first_claim] + other_claims[: max(0, max_claims - 1)]
    else:
        # Rank all claims strictly by checkability score
        scored.sort(key=lambda c: (-c.score, c.position))
        selected = scored[:max_claims]

    selected.sort(key=lambda c: c.position)  # restore reading order

    # ── Non-restrictive clause cleanup & context resolution:
    # 1. Strip parenthetical relative clauses (e.g. ", whose films usually include...")
    #    which contain journalistic trivia that NLI models fail to entail from wire news.
    # 2. Inject or expand the primary headline subject so downstream search and NLI
    #    have clean, standalone assertions with the full entity name.
    for claim in selected:
        # Strip parenthetical relative clauses to isolate the factual core
        simplified = re.sub(r',\s*(?:who|whose|which)\s+[^,]{8,150}\s*,', '', claim.text)
        if simplified != claim.text and len(simplified.strip()) >= 15:
            claim.text = simplified.strip()
            claim.reasons.append("simplified parenthetical clause for verification")

        if subject and claim.position != 0:
            claim_lowered = claim.text.lower()
            subject_lowered = subject.lower()
            tokens = tokenize_words(subject)
            surname = tokens[-1].lower() if tokens else ""

            # Resolve anaphora like "in this case" -> "in the {subject} case"
            if "in this case" in claim_lowered and subject_lowered not in claim_lowered:
                claim.text = re.sub(r"\bin this case\b", f"in the {subject} case", claim.text, flags=re.I)
                claim_lowered = claim.text.lower()
                claim.reasons.append(f"resolved 'in this case' to '{subject} case'")
            elif "in the case" in claim_lowered and subject_lowered not in claim_lowered:
                claim.text = re.sub(r"\bin the case\b", f"in the {subject} case", claim.text, flags=re.I)
                claim_lowered = claim.text.lower()
                claim.reasons.append(f"resolved 'in the case' to '{subject} case'")

            # Simplify agency-reported cumulative arrest counts to standalone factual assertions
            if re.search(
                r"^(?:(?:mumbai\s+|delhi\s+)?(?:crime\s+branch|special\s+cell|police|authorities|investigators)\s+(?:has|have)\s+(?:so\s+far\s+)?(?:arrested|detained))\s+(\d+)\s+accused\b",
                claim.text,
                flags=re.I,
            ):
                claim.text = re.sub(
                    r"^(?:(?:mumbai\s+|delhi\s+)?(?:crime\s+branch|special\s+cell|police|authorities|investigators)\s+(?:has|have)\s+(?:so\s+far\s+)?(?:arrested|detained))\s+(\d+)\s+accused\b",
                    r"A total of \1 accused were arrested",
                    claim.text,
                    flags=re.I,
                )
                claim_lowered = claim.text.lower()
                claim.reasons.append("formulated core event count assertion")

            # Only inject if the sentence actually lacks a named subject or starts with a pronoun
            existing_nouns = [
                n for n in proper_nouns(claim.text)
                if n.lower() not in {
                    "police", "officials", "authorities", "investigators", "report", "news",
                    "sources", "people", "state", "city", "department", "champions", "league",
                    "crime", "branch", "mumbai", "delhi", "special", "cell", "court",
                }
            ]
            first_word = tokenize_words(claim.text)[0].lower() if tokenize_words(claim.text) else ""
            lacks_subject = not existing_nouns or first_word in _SUBJECT_PRONOUNS

            # If the claim starts with the surname (e.g. "Shroff had just survived..."),
            # expand to full name: "Tiger Shroff had just survived..."
            if tokens and claim.text.startswith(tokens[-1]):
                claim.text = subject + claim.text[len(tokens[-1]):]
                claim.reasons.append(f"expanded surname to '{subject}'")
            elif lacks_subject and subject_lowered not in claim_lowered and (not surname or surname not in claim_lowered):
                # Neither full subject nor surname is in the claim:
                # Prepend the subject to provide context for downstream query & NLI
                claim.text = f"{subject}: {claim.text}"
                claim.reasons.append(
                    f"subject '{subject}' injected from headline (context resolution)"
                )

    return selected

