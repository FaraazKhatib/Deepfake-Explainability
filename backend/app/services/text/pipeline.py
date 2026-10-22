"""
End-to-end orchestration for a fact check.

    input text (headline / paragraph / article)
      -> extract check-worthy claims        (claims/extractor.py)
      -> build a search query per claim     (claims/query_builder.py)
      -> retrieve live evidence in parallel (retrieval/aggregator.py)
      -> NLI stance per evidence item       (stance/nli.py)
      -> collapse per domain, weight, decide(scoring/verdict.py)
      -> style assessment, reported apart   (style/classifier.py)
      -> narrative explanation              (scoring/explain.py)

Like the text-detector module, this deliberately does not import FastAPI, so
the API, the tests and any batch script all drive the same object.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from . import config
from .claims.extractor import Claim, extract_claims
from .claims.query_builder import build_query
from .credibility.sources import CredibilityIndex
from .exceptions import ModelLoadError
from .retrieval.aggregator import gather_evidence
from .retrieval.base import Evidence, RetrievalReport
from .scoring.explain import explain_claim, explain_overall, short_summary
from .scoring.verdict import ClaimVerdict, combine_claim_verdicts, decide
from .stance.debunk import detect_reported_speech
from .stance.nli import StanceDetector
from .style.classifier import StyleAssessment, StyleClassifier
from .textutils import clean_text, split_sentences

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0.0"


class FactCheckPipeline:
    """Owns the loaded models and the credibility index. One per process."""

    def __init__(
        self,
        stance_detector: Optional[StanceDetector] = None,
        style_classifier: Optional[StyleClassifier] = None,
        credibility: Optional[CredibilityIndex] = None,
        enable_style: Optional[bool] = None,
    ):
        self.stance_detector = stance_detector or StanceDetector()
        self.enable_style = (
            config.ENABLE_STYLE_CLASSIFIER if enable_style is None else enable_style
        )
        self.style_classifier = style_classifier or (
            StyleClassifier() if self.enable_style else None
        )
        self.credibility = credibility or CredibilityIndex()
        self.loaded_at: Optional[float] = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """
        Load models and data ONCE, at application startup.

        The NLI model is essential; failure to load it is fatal. The style
        classifier is optional and degrades to heuristics only.
        """
        started = time.perf_counter()
        self.credibility.load()
        self.stance_detector.load()  # raises ModelLoadError if all candidates fail

        if self.style_classifier is not None:
            if not self.style_classifier.load():
                pass  # heuristics still work; the classifier reports itself unloaded

        self.loaded_at = time.time()
        logger.info("Fact-check pipeline ready in %.1fs", time.perf_counter() - started)

    @property
    def is_ready(self) -> bool:
        return self.stance_detector.is_loaded

    # ------------------------------------------------------------------
    def check(
        self,
        text: str,
        max_claims: Optional[int] = None,
        treat_as_single_claim: bool = False,
        include_evidence: bool = True,
    ) -> dict:
        """
        Run a full fact check.

        Raises InputValidationError / NoClaimFoundError for bad input,
        ModelLoadError when not ready, InferenceError when the model fails.
        """
        started = time.perf_counter()

        if not self.is_ready:
            raise ModelLoadError(
                "The stance model is not loaded; the service is not ready."
            )

        cleaned = clean_text(text)
        claims: List[Claim] = extract_claims(
            cleaned,
            max_claims=max_claims,
            treat_as_single_claim=treat_as_single_claim,
        )

        # Style is assessed on the whole input, once, and kept out of the
        # verdict path (see style/classifier.py for why).
        style: Optional[StyleAssessment] = None
        if self.style_classifier is not None:
            sentences = split_sentences(cleaned)
            headline = sentences[0] if sentences else ""
            style = self.style_classifier.assess(cleaned, title=headline)

        claim_payloads = []
        verdicts: List[ClaimVerdict] = []
        merged_evidence: List[Evidence] = []
        merged_report = RetrievalReport()

        for claim in claims:
            query = build_query(claim.text)
            report = gather_evidence(query)

            # Merge retrieval diagnostics across claims.
            for name, count in report.per_source_counts.items():
                merged_report.per_source_counts[name] = (
                    merged_report.per_source_counts.get(name, 0) + count
                )
            merged_report.errors.update(report.errors)
            merged_report.notices.update(report.notices)
            merged_report.queries_used.update(
                {f"{name} (claim {claim.position})": q for name, q in report.queries_used.items()}
            )
            merged_evidence.extend(report.evidence)

            stances = self.stance_detector.classify(claim.text, report.evidence)
            verdict = decide(claim.text, stances, self.credibility)
            verdicts.append(verdict)

            # A claim phrased as reported speech is a statement ABOUT an
            # assertion, not the assertion. "A story circulated that X" is true
            # whenever the story circulated, however false X is. The system
            # checks the claim as written, so it must say so plainly rather
            # than appear to have checked the underlying statement.
            framing = detect_reported_speech(claim.text)
            if framing:
                verdict.notes.append(
                    f"This claim reports on what someone said (\"{framing}\"). "
                    "We only checked if they said it, not if their statement is true. "
                    "To check the actual story, remove the quotes and framing."
                )

            payload = verdict.to_dict(include_evidence=include_evidence)
            payload["reported_speech_framing"] = framing
            payload["query"] = query.to_dict()
            payload["checkworthiness"] = round(claim.score, 3)
            payload["selected_because"] = claim.reasons
            payload["explanation"] = explain_claim(verdict)
            claim_payloads.append(payload)

        merged_report.evidence = merged_evidence
        overall = combine_claim_verdicts(verdicts)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return {
            # --- headline fields -------------------------------------------
            "verdict": overall["overall"],
            "rationale": overall["rationale"],
            "claim_count": overall["claim_count"],
            "supported_claims": overall.get("supported", 0),
            "refuted_claims": overall.get("refuted", 0),
            "disputed_claims": overall.get("disputed", 0),
            "unverified_claims": overall.get("unverified", 0),
            "explanation": explain_overall(verdicts, merged_report, style),
            "summary": short_summary(overall, verdicts),
            # --- detail ------------------------------------------------------
            "claims": claim_payloads,
            "manipulation_risk": style.to_dict() if style else None,
            "retrieval": merged_report.to_dict(),
            "meta": {
                "pipeline_version": PIPELINE_VERSION,
                "stance_model": self.stance_detector.model_name,
                "device": self.stance_detector.device,
                "style_affects_verdict": config.STYLE_AFFECTS_VERDICT,
                "processing_time_ms": round(elapsed_ms, 1),
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

    # ------------------------------------------------------------------
    def info(self) -> dict:
        """State report for GET /health."""
        return {
            "ready": self.is_ready,
            "pipeline_version": PIPELINE_VERSION,
            "stance": self.stance_detector.info(),
            "style": (
                self.style_classifier.info()
                if self.style_classifier is not None
                else {"enabled": False, "loaded": False}
            ),
            "credibility": self.credibility.info(),
            "retrieval": {
                "gdelt": config.ENABLE_GDELT,
                "google_news": config.ENABLE_GOOGLE_NEWS,
                "wikipedia": config.ENABLE_WIKIPEDIA,
                "max_evidence": config.MAX_EVIDENCE_ITEMS,
                "cache_ttl_seconds": config.CACHE_TTL_SECONDS,
            },
            "thresholds": {
                "min_independent_sources": config.MIN_INDEPENDENT_SOURCES,
                "support_threshold": config.SUPPORT_THRESHOLD,
                "refute_threshold": config.REFUTE_THRESHOLD,
                "stance_min_confidence": config.STANCE_MIN_CONFIDENCE,
                "max_claims": config.MAX_CLAIMS,
            },
        }
