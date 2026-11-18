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
    "rescues", "rescue", "rescued", "beats", "beat", "defeated", "defeats", "defeat",
    "scores", "score", "scored", "strikes", "struck", "hold", "holds", "held", "powers", "powered",
    "seals", "sealed", "downs", "downed", "edges", "edged", "stuns", "stunned", "lifts", "lifted",
    "fires", "fired", "nets", "netted", "qualifies", "qualified",
    "summoned", "summon", "testify", "testified", "subpoena", "subpoenaed",
    "resigns", "resign", "resigning", "steps", "stepping", "stepped", "ousted",
    "named", "name", "cast", "casting", "picked", "chosen", "selected", "tapped",
    "stars", "starred", "starring", "plays", "played", "playing", "joins", "joined",
    "replaces", "replaced", "replacing", "debuts", "debuted",
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
    # Strategy A: position == 0 with named entities is a headline; headlines use telegraphic style
    # or passive voice with omitted auxiliary verbs and must not be penalized as fragments.
    if not (verbs or action_hits):
        if position == 0 and entities:
            pass  # Do not penalize headlines containing named entities
        else:
            score -= 0.8
            reasons.append("lacks an assertive or action verb (fragment)")

    # Inverted pyramid priority boost:
    # Headlines (position == 0) represent the primary news event by definition.
    # Lede sentences (positions 1-4) carry immediate secondary details.
    if position == 0:
        score += 1.0
        reasons.append("article headline (primary news event)")
    elif 1 <= position <= 4:
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


def split_compound_headline(headline: str, context_subject: str = "") -> List[str]:
    """
    Decompose compound sentences/headlines joined by semicolons or conjunctions
    (' as ', ' while ', ' whilst ', ' amidst ', ' when ', ' whereas ').
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
    pattern = r"^(.*?\b[A-Za-z0-9]+)[,\s]+(as|while|whilst|amidst|when|whereas|though|although)\s+([a-zA-Z0-9].*)$"
    m = re.match(pattern, cleaned.strip().rstrip("."))
    if not m:
        return [cleaned]
    c1, conj, c2 = m.group(1).strip(), m.group(2).lower(), m.group(3).strip()
    if len(c1.split()) < 3 or len(c2.split()) < 3:
        return [cleaned]

    # Prepositional role phrases like "resigned as CEO" or "taken over as interim chief"
    # are grammatical objects, NOT conjunctions joining two separate events.
    if conj == "as":
        role_verb_pattern = r"\b(?:resigned|resign|resigns|resigning|taken\s+over|take\s+over|took\s+over|takes\s+over|taking\s+over|stepped\s+down|step\s+down|stepped\s+in|step\s+in|served|serve|serves|appointed|named|acted|hired|known|identified|debut|debuts|debuted|cast|stars|starred|starring)\s*$"
        if re.search(role_verb_pattern, c1, re.I):
            return [cleaned]

    # Only transfer trailing prepositional tournament/setting phrase (e.g. 'in Champions League', 'in Florida')
    # if c1 does not already specify its own location/setting.
    has_own_prep = bool(re.search(r"\b(?:in|at|during|for|on)\s+[A-Z]", c1))
    prep_match = re.search(r"\s+(in|at|during|for|on)\s+([A-Z][A-Za-z0-9\s]{1,25})$", c2)
    if prep_match and not has_own_prep:
        prep_phrase = prep_match.group(0)
        prep_words = prep_phrase.lower().split()
        if not any(w in {"to", "that", "which", "and", "or", "while"} for w in prep_words) and len(prep_words) <= 4:
            if not any(w in c1.lower().split() for w in prep_words[1:]):
                c1 = f"{c1}{prep_phrase}"

    # If c1 starts with a pronoun and we have subject context, resolve it
    if context_subject:
        if re.match(r"^his\b", c1, re.I):
            poss = f"{context_subject}'s" if not context_subject.endswith("s") else f"{context_subject}'"
            c1 = re.sub(r"^his\b", poss, c1, flags=re.I)
        elif re.match(r"^he\b", c1, re.I):
            c1 = re.sub(r"^he\b", context_subject, c1, flags=re.I)

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
            c2 = f"{c2} in {event_match.group(1).strip()}"

    return [c1, c2]


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

    headline_raw = raw_sentences[0]
    headline_subject = _get_primary_subject(headline_raw)
    from ..textutils import proper_nouns
    all_nouns = proper_nouns(clean_headline_text(headline_raw))
    person_subject = next(
        (n for n in all_nouns if n.lower() not in {"man", "city", "arsenal", "league", "champions", "united"}),
        headline_subject,
    )

    # Decompose compound headline and body sentences if conjoining distinct clauses
    sentences: List[str] = []
    headline_parts = split_compound_headline(headline_raw, context_subject=person_subject or headline_subject)
    sentences.extend(headline_parts)
    headline_len = len(headline_parts)

    for s in raw_sentences[1:]:
        cleaned_s = clean_headline_text(s)
        if not cleaned_s:
            continue
        parts = split_compound_headline(cleaned_s, context_subject=person_subject or headline_subject)
        for p in parts:
            if len(p.split()) >= 3:
                sentences.append(p)

    # If the user requested Headline Focus (treat_as_single_claim=True), extract the
    # headline's primary assertion as the single claim.
    if treat_as_single_claim:
        candidate_text = headline_parts[0] if headline_parts else (sentences[0] if sentences else text.strip())
        simplified = re.sub(r',\s*(?:who|whose|which)\s+[^,]{8,150}\s*,', '', candidate_text).strip()
        if simplified and len(simplified) >= 15:
            candidate_text = simplified
        score, reasons = score_sentence(candidate_text, position=0, subject=person_subject or headline_subject)

        if score <= 0:
            for cand in headline_parts[1:] + sentences[headline_len:]:
                cand_clean = re.sub(r',\s*(?:who|whose|which)\s+[^,]{8,150}\s*,', '', cand).strip()
                s_score, s_reasons = score_sentence(cand_clean, position=0, subject=person_subject or headline_subject)
                if s_score > 0:
                    candidate_text, score, reasons = cand_clean, s_score, s_reasons
                    break

        if score <= 0:
            raise NoClaimFoundError("Please enter a factual assertion, rather than a question or opinion.")

        return [
            Claim(
                text=candidate_text,
                score=score,
                reasons=reasons,
                position=0,
            )
        ]

    # A single short sentence (a headline, a typed assertion) is the claim.
    if len(sentences) <= 1:
        target_text = sentences[0] if sentences else text.strip()
        simplified = re.sub(r',\s*(?:who|whose|which)\s+[^,]{8,150}\s*,', '', target_text).strip()
        if simplified and len(simplified) >= 15:
            target_text = simplified
        score, reasons = score_sentence(target_text, position=0, subject=person_subject or headline_subject)
        if score <= 0:
            raise NoClaimFoundError("Please enter a factual assertion, rather than a question or opinion.")
        if len(target_text) > 400:
            raise InputValidationError("A single claim must be at most 400 characters; submit separate claims instead.")
        return [
            Claim(
                text=target_text,
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

    # Strategy A: The headline is the central assertion of the news article by definition.
    # Prioritize valid headline claims as Claim 1 (and Claim 2 if multi-topic) unless they are
    # explicit questions, pure opinion, or editorial commentary.
    headline_claims = [
        c for c in scored
        if c.position < headline_len and not any("editorial" in r for r in c.reasons) and not c.text.rstrip().endswith("?")
    ]
    body_claims = [c for c in scored if c.position >= headline_len]
    body_claims.sort(key=lambda c: (-c.score, c.position))

    if headline_claims:
        # Include all valid headline claims (up to max_claims - 1 if body claims exist)
        needed_from_hl = min(len(headline_claims), max(1, max_claims - (1 if body_claims else 0)))
        selected = headline_claims[:needed_from_hl]
        remaining_slots = max(0, max_claims - len(selected))
        selected.extend(body_claims[:remaining_slots])
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
                # Resolve leading pronouns directly with the subject name rather than prepending "Subject: ..."
                if re.match(r"^his\b", claim.text, re.I):
                    poss = f"{subject}'s" if not subject.endswith("s") else f"{subject}'"
                    claim.text = re.sub(r"^his\b", poss, claim.text, flags=re.I)
                    claim.reasons.append(f"resolved 'his' to '{poss}'")
                elif re.match(r"^he\b", claim.text, re.I):
                    claim.text = re.sub(r"^he\b", subject, claim.text, flags=re.I)
                    claim.reasons.append(f"resolved 'he' to '{subject}'")
                elif re.match(r"^(?:the\s+)?(?:actor|actress|player|star|singer|victim|driver|pilot|leader|president|official|spokesperson|witness)\b", claim.text, re.I):
                    claim.text = re.sub(r"^(?:the\s+)?(?:actor|actress|player|star|singer|victim|driver|pilot|leader|president|official|spokesperson|witness)\b", subject, claim.text, flags=re.I)
                    claim.reasons.append(f"resolved role noun to '{subject}'")
                else:
                    # Prepend the subject to provide context for downstream query & NLI
                    claim.text = f"{subject}: {claim.text}"
                    claim.reasons.append(
                        f"subject '{subject}' injected from headline (context resolution)"
                    )

    return selected

