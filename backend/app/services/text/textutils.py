"""
Small self-contained text helpers.

The sibling ``text_detector`` module has a fuller preprocessing layer. This
module deliberately keeps its own compact copy rather than importing across
package boundaries, so ``fact_checker`` stays independently deployable.

As before: no NLTK/spaCy, so there is no runtime data download.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_ODD_SPACES = re.compile(r"[   -   　]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
_BOUNDARY_RE = re.compile(r"([.!?…]+[\"'”’)\]]*)(\s+)")
_LAST_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z.']*)[.!?…]+[\"'”’)\]]*$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "gen", "sen",
    "rep", "gov", "pres", "vs", "etc", "e.g", "i.e", "cf", "al", "inc", "ltd",
    "co", "corp", "dept", "est", "approx", "no", "vol", "pp", "u.s", "u.k",
    "e.u", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
}

# Words that are capitalised for reasons other than being a proper noun.
_COMMON_CAPITALISED = {
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "you", "i", "there", "here", "but", "and", "or", "if",
    "when", "while", "after", "before", "however", "meanwhile", "according",
    "in", "on", "at", "for", "to", "of", "by", "with", "from", "as", "his",
    "her", "its", "their", "our", "my", "no", "not", "one", "two",
}

# Ordinary words that routinely OPEN a place or organisation name.
# Kept only when immediately followed by another capitalised word, so
# "New Delhi" survives while "New rules were announced" does not.
_NAME_PREFIXES = {
    "new", "north", "south", "east", "west", "united", "great", "saint",
    "san", "santa", "los", "las", "fort", "port", "cape", "mount", "lake",
    "old", "upper", "lower", "central", "national", "royal", "federal",
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "out", "over", "own", "s", "same",
    "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your",
}


def clean_text(raw: str) -> str:
    """Conservative cleanup: strip noise, keep punctuation and casing."""
    text = unicodedata.normalize("NFC", raw)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _ODD_SPACES.sub(" ", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(raw: str) -> str:
    """Remove tags and decode the handful of entities RSS feeds actually use."""
    text = _HTML_TAG_RE.sub(" ", raw or "")
    replacements = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'", "&hellip;": "...",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _is_false_boundary(candidate: str, paragraph: str, match: re.Match) -> bool:
    token_match = _LAST_TOKEN_RE.search(candidate)
    if token_match:
        token = token_match.group(1)
        stripped = token.lower().rstrip(".")
        if stripped in _ABBREVIATIONS:
            return True
        if len(stripped) == 1 and token[0].isupper():
            return True
    following = paragraph[match.end() : match.end() + 1]
    return bool(following and following.islower())


def split_sentences(text: str) -> List[str]:
    """Abbreviation-aware regex sentence segmentation."""
    text = (text or "").strip()
    if not text:
        return []
    sentences: List[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        start = 0
        for match in _BOUNDARY_RE.finditer(line):
            candidate = line[start : match.end(1)].strip()
            if not candidate or _is_false_boundary(candidate, line, match):
                continue
            sentences.append(candidate)
            start = match.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if s]


def proper_nouns(text: str) -> List[str]:
    """
    Capitalised tokens that are probably named entities.

    A regex heuristic, not a named-entity recogniser: it misses lowercase
    entities and over-fires on title-case headlines. It is good enough to build
    a search query, which is all it is used for.
    """
    found: List[str] = []
    for sentence in split_sentences(text) or [text]:
        matches = list(re.finditer(r"\b[A-Z][A-Za-z0-9.'-]*\b", sentence))
        for index, match in enumerate(matches):
            token = match.group(0)
            lowered = token.lower().strip(".")
            if len(lowered) < 2 or lowered in _COMMON_CAPITALISED:
                continue

            if lowered in _NAME_PREFIXES:
                # Keep only when it opens a compound name: the next capitalised
                # token must follow immediately and not itself be a filler word.
                following = matches[index + 1] if index + 1 < len(matches) else None
                if following is None:
                    continue
                gap = sentence[match.end() : following.start()]
                next_word = following.group(0).lower().strip(".")
                if gap.strip() or next_word in _COMMON_CAPITALISED:
                    continue

            # Sentence-initial capitals are otherwise kept. News sentences lead
            # with names constantly ("Narendra Modi met...", "Apple announced..."),
            # so requiring the name to appear twice loses the main entity of the
            # claim. The cost is a few ordinary words entering the query, which
            # is harmless -- terms are capped and ranked -- whereas a missing
            # name breaks retrieval outright.
            found.append(token.strip(".,"))
    # De-duplicate, preserving order.
    seen, unique = set(), []
    for token in found:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            unique.append(token)
    return unique


def quoted_phrases(text: str) -> List[str]:
    """Phrases inside quotation marks: usually the crux of a claim."""
    phrases: List[str] = []
    for pattern in (r'"([^"]{4,80})"', r"“([^”]{4,80})”"):
        phrases.extend(m.group(1).strip() for m in re.finditer(pattern, text or ""))
    return [p for p in phrases if p]


def numbers_and_dates(text: str) -> List[str]:
    """Numeric tokens, percentages, years and money amounts."""
    return re.findall(
        r"\b(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?%?|\$\d[\d,.]*|\d{4})\b", text or ""
    )
