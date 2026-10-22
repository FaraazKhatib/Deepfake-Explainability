"""
Central configuration for the real-time news fact-checking module.

Every tunable lives here and every value can be overridden with an environment
variable (prefix ``FC_``). Nothing downstream hardcodes a threshold or a weight.

As with the text-detector module, the defaults are reasoned starting points,
not calibrated values. Tune them against your own labelled claim set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ==========================================================================
# NLI / STANCE MODEL  (the core of the system)
# ==========================================================================
# Trained on MNLI + FEVER + ANLI. FEVER is a fact-verification corpus, which
# is exactly the task shape here: given evidence, does it support the claim?
NLI_MODEL = _env_str("FC_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
# Optional larger candidate, selectable with FC_NLI_MODEL. Keep the CPU baseline
# until a deployment-specific claim/evidence evaluation justifies promotion.
NLI_LARGE_CANDIDATE = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_FALLBACK_MODELS: List[str] = [
    m.strip()
    for m in _env_str(
        "FC_NLI_FALLBACKS",
        "cross-encoder/nli-deberta-v3-base,facebook/bart-large-mnli",
    ).split(",")
    if m.strip()
]
NLI_MAX_LENGTH = _env_int("FC_NLI_MAX_LENGTH", 256)
NLI_BATCH_SIZE = _env_int("FC_NLI_BATCH_SIZE", 8)

# Minimum probability before a snippet counts as taking a side at all.
STANCE_MIN_CONFIDENCE = _env_float("FC_STANCE_MIN_CONFIDENCE", 0.55)

DEVICE = _env_str("FC_DEVICE", "auto")
HF_CACHE_DIR = os.getenv("FC_HF_CACHE_DIR") or None
HF_TOKEN = (
    os.getenv("FC_HF_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGING_FACE_HUB_TOKEN")
    or None
)
OFFLINE_ONLY = _env_bool("FC_OFFLINE_ONLY", False)


# ==========================================================================
# STYLE CLASSIFIER  (deliberately NOT in the causal path to a verdict)
# ==========================================================================
ENABLE_STYLE_CLASSIFIER = _env_bool("FC_ENABLE_STYLE", True)
STYLE_MODEL = _env_str("FC_STYLE_MODEL", "hamzab/roberta-fake-news-classification")

# Some fake-news checkpoints were trained on a specific input template.
# Placeholders: {title} and {text}.
STYLE_INPUT_TEMPLATE = _env_str("FC_STYLE_TEMPLATE", "{title} {text}")

# SAFETY RAIL. When False (the default) the style classifier and the
# sensationalism heuristics can never move the veracity verdict; they are
# reported separately as `manipulation_risk`.
#
# Why this is the default: a fake-news style classifier learns the register of
# its training corpus -- datelines, punctuation, sensational vocabulary -- not
# truth. Letting it vote on veracity means accurate reporting written in a
# tabloid register, or by a non-native speaker, gets pushed toward "false",
# while a well-formatted fabrication gets pushed toward "true". That is a
# discrimination risk, not just an accuracy problem.
#
# Set FC_STYLE_AFFECTS_VERDICT=true to fold it in anyway (not recommended);
# STYLE_VERDICT_WEIGHT then applies.
STYLE_AFFECTS_VERDICT = _env_bool("FC_STYLE_AFFECTS_VERDICT", False)
STYLE_VERDICT_WEIGHT = _env_float("FC_STYLE_VERDICT_WEIGHT", 0.15)


# ==========================================================================
# EVIDENCE RETRIEVAL
# ==========================================================================
ENABLE_GDELT = _env_bool("FC_ENABLE_GDELT", True)
ENABLE_GOOGLE_NEWS = _env_bool("FC_ENABLE_GOOGLE_NEWS", True)
ENABLE_WIKIPEDIA = _env_bool("FC_ENABLE_WIKIPEDIA", True)

# GDELT indexes worldwide news every 15 minutes. Free, no API key.
# It DOES require a User-Agent header; without one it answers 429.
GDELT_ENDPOINT = _env_str(
    "FC_GDELT_ENDPOINT", "https://api.gdeltproject.org/api/v2/doc/doc"
)
GDELT_TIMESPAN = _env_str("FC_GDELT_TIMESPAN", "14d")
GDELT_MAX_RECORDS = _env_int("FC_GDELT_MAX_RECORDS", 25)
# GDELT ANDs bare terms, so a long query matches nothing at all. Observed
# during testing: a 9-term query returned 0 articles, a 4-term one returned 9.
GDELT_MAX_QUERY_TERMS = _env_int("FC_GDELT_MAX_QUERY_TERMS", 4)

# Google News RSS search. Free, no API key, effectively a live news search.
GOOGLE_NEWS_ENDPOINT = _env_str(
    "FC_GOOGLE_NEWS_ENDPOINT", "https://news.google.com/rss/search"
)
GOOGLE_NEWS_LANG = _env_str("FC_GOOGLE_NEWS_LANG", "en-US")
GOOGLE_NEWS_COUNTRY = _env_str("FC_GOOGLE_NEWS_COUNTRY", "US")
GOOGLE_NEWS_MAX_ITEMS = _env_int("FC_GOOGLE_NEWS_MAX_ITEMS", 25)

WIKIPEDIA_ENDPOINT = _env_str(
    "FC_WIKIPEDIA_ENDPOINT", "https://en.wikipedia.org/w/api.php"
)
WIKIPEDIA_MAX_ARTICLES = _env_int("FC_WIKIPEDIA_MAX_ARTICLES", 3)

HTTP_TIMEOUT = _env_float("FC_HTTP_TIMEOUT", 20.0)
HTTP_USER_AGENT = _env_str(
    "FC_USER_AGENT",
    "Mozilla/5.0 (compatible; fact-checker-fyp/1.0; academic research prototype)",
)
HTTP_MAX_RETRIES = _env_int("FC_HTTP_MAX_RETRIES", 2)

# Wikimedia's User-Agent policy rejects generic browser-like strings: sending
# "Mozilla/5.0 ..." to the Wikipedia API returns 403 Forbidden. It wants a
# descriptive agent identifying the tool. Put a contact URL or address here if
# you deploy this anywhere real.
WIKIPEDIA_USER_AGENT = _env_str(
    "FC_WIKIPEDIA_USER_AGENT",
    "fact-checker-fyp/1.0 (academic project; evidence-retrieval prototype)",
)

# Language filtering. The NLI model is English-trained; asking it to judge a
# Greek or Turkish headline produces confident nonsense, and that nonsense was
# observed voting in real verdicts during testing. Non-English evidence is
# therefore dropped before the stance stage rather than scored badly.
FILTER_NON_ENGLISH = _env_bool("FC_FILTER_NON_ENGLISH", True)
NON_ASCII_LETTER_LIMIT = _env_float("FC_NON_ASCII_LIMIT", 0.25)

# Relevance gate. An NLI model given an unrelated premise should answer
# "neutral", but in practice it often returns a confident contradiction --
# during testing an article about night trains "contradicted" a claim about the
# Eiffel Tower at 0.98. Requiring lexical overlap with the claim before a
# stance is allowed to count blocks that failure mode cheaply.
MIN_RELEVANCE_OVERLAP = _env_float("FC_MIN_RELEVANCE_OVERLAP", 0.25)
RELEVANCE_TERM_CAP = _env_int("FC_RELEVANCE_TERM_CAP", 6)

# Debunk detection. NLI treats "this claim is false" as NEUTRAL rather than as
# a contradiction (measured: 0.099 contradiction on explicit debunk phrasing,
# 0.006 on Wikipedia's "falsely claiming ..."), so debunked conspiracy theories
# were being missed entirely. A lexical detector covers the gap; it is kept
# lexical so that it fails independently of the NLI model.
ENABLE_DEBUNK_DETECTION = _env_bool("FC_ENABLE_DEBUNK", True)
# Higher bar than the ordinary relevance gate: an article debunking a DIFFERENT
# claim about the same people must not be read as refuting this one.
DEBUNK_MIN_OVERLAP = _env_float("FC_DEBUNK_MIN_OVERLAP", 0.50)
DEBUNK_BASE_CONFIDENCE = _env_float("FC_DEBUNK_BASE_CONFIDENCE", 0.60)
# Capped below 1.0: a lexical heuristic must never look as certain as a direct
# model contradiction.
DEBUNK_MAX_CONFIDENCE = _env_float("FC_DEBUNK_MAX_CONFIDENCE", 0.85)

# Total evidence snippets carried into the (comparatively expensive) NLI stage.
MAX_EVIDENCE_ITEMS = _env_int("FC_MAX_EVIDENCE", 30)
# Cap per domain BEFORE scoring, so one prolific outlet cannot flood the pool.
MAX_PER_DOMAIN = _env_int("FC_MAX_PER_DOMAIN", 3)

# Cache identical retrievals briefly: re-checking the same claim while
# developing should not hammer the free endpoints.
CACHE_TTL_SECONDS = _env_int("FC_CACHE_TTL", 300)
CACHE_MAX_ENTRIES = _env_int("FC_CACHE_MAX_ENTRIES", 128)


# ==========================================================================
# CLAIM EXTRACTION
# ==========================================================================
MAX_CLAIMS = _env_int("FC_MAX_CLAIMS", 3)      # claims checked per request
MIN_CLAIM_WORDS = _env_int("FC_MIN_CLAIM_WORDS", 5)
MAX_CLAIM_WORDS = _env_int("FC_MAX_CLAIM_WORDS", 60)
MIN_INPUT_WORDS = _env_int("FC_MIN_INPUT_WORDS", 4)
MAX_INPUT_CHARS = _env_int("FC_MAX_INPUT_CHARS", 50_000)
MAX_QUERY_TERMS = _env_int("FC_MAX_QUERY_TERMS", 6)


# ==========================================================================
# VERDICT AGGREGATION
# ==========================================================================
# Minimum number of INDEPENDENT domains taking a side before that side can
# carry a verdict. Syndication means article count is not source count.
MIN_INDEPENDENT_SOURCES = _env_int("FC_MIN_INDEPENDENT_SOURCES", 2)

# Single-source exception. MIN_INDEPENDENT_SOURCES exists to defeat news
# SYNDICATION -- fifty copies of one wire story are not fifty sources. That
# rationale does not apply to an encyclopedia entry, an official body speaking
# about itself, a peer-reviewed journal, or an IFCN fact-checker publishing a
# rating: none of those are syndicated newsroom copy. Without this exception,
# durable factual claims ("the WHO is a UN agency") stay permanently
# Unverified because news search does not cover settled facts.
#
# Verdicts reached this way are capped at Medium confidence and always carry a
# note saying they rest on one source.
SINGLE_SOURCE_TIERS = set(
    _env_str(
        "FC_SINGLE_SOURCE_TIERS",
        # NOTE: wire services are deliberately EXCLUDED. A wire service is
        # precisely the syndicated newsroom this rule guards against -- letting
        # one carry a verdict alone reopens the hole that MIN_INDEPENDENT_SOURCES
        # was added to close. (Caught by test_syndicated_single_source_cannot_
        # reach_a_verdict.)
        "reference,official_body,peer_reviewed,ifcn_factchecker",
    ).split(",")
)
# Gated on two SEPARATE conditions rather than on their product. The product
# form conflated "is this source authoritative" with "is this stance
# confident", so a highly credible source with a merely good stance score fell
# below the bar for the wrong reason (Wikipedia at 0.75 credibility x 0.75
# stance = 0.5625, under a 0.60 floor, despite both halves being fine).
SINGLE_SOURCE_MIN_CREDIBILITY = _env_float("FC_SINGLE_SOURCE_MIN_CREDIBILITY", 0.70)
SINGLE_SOURCE_MIN_STANCE = _env_float("FC_SINGLE_SOURCE_MIN_STANCE", 0.70)

# Weighted-evidence thresholds. Weight per domain =
#   credibility_weight * stance_confidence * recency_factor
SUPPORT_THRESHOLD = _env_float("FC_SUPPORT_THRESHOLD", 1.2)
REFUTE_THRESHOLD = _env_float("FC_REFUTE_THRESHOLD", 1.0)
# When both sides clear their thresholds and neither dominates by this ratio,
# the verdict is "Disputed" rather than a coin flip.
DISPUTE_RATIO = _env_float("FC_DISPUTE_RATIO", 2.0)

# Evidence older than this contributes less. Breaking news should not be
# refuted by a five-year-old article about a different event.
RECENCY_HALFLIFE_DAYS = _env_float("FC_RECENCY_HALFLIFE_DAYS", 30.0)
RECENCY_MIN_FACTOR = _env_float("FC_RECENCY_MIN_FACTOR", 0.35)

# Credibility weight given to a domain that is not in the ratings file.
# Deliberately mid-low: unknown is not the same as untrustworthy, but an
# unrated domain should not outvote a wire service.
UNRATED_SOURCE_WEIGHT = _env_float("FC_UNRATED_WEIGHT", 0.40)
CREDIBILITY_FILE = _env_str("FC_CREDIBILITY_FILE", "data/source_credibility.csv")


@dataclass(frozen=True)
class Settings:
    """Snapshot for logging and GET /health."""

    nli_model: str = NLI_MODEL
    style_model: str = STYLE_MODEL
    style_affects_verdict: bool = STYLE_AFFECTS_VERDICT
    device: str = DEVICE
    max_evidence: int = MAX_EVIDENCE_ITEMS
    max_claims: int = MAX_CLAIMS
    min_independent_sources: int = MIN_INDEPENDENT_SOURCES


SETTINGS = Settings()


def resolve_device(preference: str = DEVICE) -> str:
    """Return the torch device string, honouring 'auto'."""
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"
