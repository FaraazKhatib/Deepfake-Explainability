"""
Presentation-style signal -- reported, but NOT part of the veracity verdict.

Two things are measured here:

1. a pretrained fake-news classifier (``hamzab/roberta-fake-news-classification``);
2. transparent sensationalism heuristics (shouting caps, exclamation pile-ups,
   clickbait formulas, absolutist language, missing attribution).

WHY THIS CANNOT DECIDE TRUTH
----------------------------
A fake-news classifier trained on a corpus like ISOT or FakeNewsNet learns the
*register* of that corpus. In practice that means it learns "Reuters-style
dateline and byline = true, emotive tabloid phrasing = false". That is a
stylistic discriminator, not a truth detector, and it fails in both directions:

* Accurate reporting written in a tabloid register, in a second language, or
  by a local outlet without a house style guide gets pushed toward "false".
* A careful, well-formatted fabrication gets pushed toward "true" -- which is
  exactly what a competent disinformation operation produces.

The first failure mode is a fairness problem, not merely an accuracy one: it
penalises writers for how they write rather than for what they assert.

So by default (``STYLE_AFFECTS_VERDICT = False``) this score is reported
alongside the verdict as ``manipulation_risk`` and cannot change it. It answers
"is this presented in a manipulative style?", which is a genuinely useful and
genuinely different question from "is this true?".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config
from ..textutils import split_sentences, tokenize_words

logger = logging.getLogger(__name__)

_CLICKBAIT_PATTERNS = [
    r"you won'?t believe", r"what happened next", r"will shock you",
    r"doctors hate", r"this one (weird )?trick", r"the truth about",
    r"they don'?t want you to know", r"number \d+ will", r"here'?s why",
    r"goes viral", r"breaking\s*:", r"must (see|watch|read)",
    r"\bwake up\b", r"mainstream media", r"they'?re hiding",
]

_ABSOLUTIST_TERMS = {
    "always", "never", "everyone", "nobody", "all", "none", "every",
    "totally", "completely", "absolutely", "undeniable", "irrefutable",
    "proven", "guaranteed", "definitely", "obviously", "clearly",
}

_ATTRIBUTION_MARKERS = (
    "according to", "said", "told", "reported by", "spokesperson",
    "statement", "study", "research", "data from", "survey", "cited",
)


@dataclass
class StyleAssessment:
    """Presentation-style measurements. Says nothing about truth."""

    manipulation_risk: float                       # 0-1, higher = more manipulative style
    classifier_fake_style: Optional[float] = None  # model probability, if loaded
    heuristic_score: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    model_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "manipulation_risk": round(self.manipulation_risk, 4),
            "classifier_fake_style_probability": (
                round(self.classifier_fake_style, 4)
                if self.classifier_fake_style is not None
                else None
            ),
            "heuristic_score": round(self.heuristic_score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "observations": self.observations,
            "model": self.model_name,
            "interpretation": (
                "Measures HOW the text is presented, not whether it is true. "
                "A high score means sensational or manipulative presentation; "
                "it is not evidence that the claim is false, and a low score is "
                "not evidence that it is true."
            ),
        }


def sensationalism_heuristics(text: str) -> tuple:
    """Transparent style measurements. Returns (score, signals, observations)."""
    signals: Dict[str, float] = {}
    observations: List[str] = []

    words = tokenize_words(text)
    if not words:
        return 0.0, signals, observations

    shouting = [w for w in words if len(w) > 2 and w.isupper()]
    caps_ratio = len(shouting) / len(words)
    signals["all_caps_word_ratio"] = caps_ratio
    if caps_ratio > 0.05:
        observations.append(
            f"{len(shouting)} of {len(words)} words are in all caps "
            f"({caps_ratio * 100:.0f}%)"
        )

    exclamations = text.count("!")
    signals["exclamation_marks"] = float(exclamations)
    if exclamations >= 2:
        observations.append(f"{exclamations} exclamation marks")

    repeated_punct = len(re.findall(r"[!?]{2,}", text))
    signals["repeated_punctuation"] = float(repeated_punct)
    if repeated_punct:
        observations.append(f"{repeated_punct} run(s) of repeated punctuation")

    lowered = text.lower()
    clickbait_hits = [p for p in _CLICKBAIT_PATTERNS if re.search(p, lowered)]
    signals["clickbait_phrases"] = float(len(clickbait_hits))
    if clickbait_hits:
        observations.append(f"{len(clickbait_hits)} clickbait-style phrase(s)")

    absolutist = [w for w in (w.lower() for w in words) if w in _ABSOLUTIST_TERMS]
    absolutist_ratio = len(absolutist) / len(words)
    signals["absolutist_ratio"] = absolutist_ratio
    if absolutist_ratio > 0.03:
        observations.append(
            f"frequent absolutist language ({len(absolutist)} instances)"
        )

    has_attribution = any(marker in lowered for marker in _ATTRIBUTION_MARKERS)
    signals["has_attribution"] = 1.0 if has_attribution else 0.0
    if not has_attribution and len(split_sentences(text)) >= 3:
        observations.append("no attribution to a named source or study")

    # Weighted combination, each component clamped to [0, 1].
    score = (
        min(1.0, caps_ratio / 0.15) * 0.20
        + min(1.0, exclamations / 4.0) * 0.15
        + min(1.0, repeated_punct / 2.0) * 0.10
        + min(1.0, len(clickbait_hits) / 2.0) * 0.30
        + min(1.0, absolutist_ratio / 0.08) * 0.15
        + (0.10 if not has_attribution else 0.0)
    )
    return min(1.0, score), signals, observations


class StyleClassifier:
    """Optional pretrained fake-news style classifier plus the heuristics."""

    def __init__(
        self,
        model_name: str = config.STYLE_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or config.resolve_device()
        self.tokenizer = None
        self.model = None
        self.fake_index: int = 0
        self.label_resolution = "not-loaded"
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> bool:
        """Load the classifier. Returns True on success; never raises."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            kwargs = {}
            if config.HF_CACHE_DIR:
                kwargs["cache_dir"] = config.HF_CACHE_DIR
            if config.OFFLINE_ONLY:
                kwargs["local_files_only"] = True
            if config.HF_TOKEN:
                kwargs["token"] = config.HF_TOKEN

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            )
            self.model.eval()
            self.model.to(self.device)
            self._resolve_label_index()
            logger.info("Style classifier ready: %s", self.model_name)
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.tokenizer = None
            self.model = None
            logger.warning(
                "Style classifier unavailable (%s); using heuristics only.",
                self.load_error,
            )
            return False

    def _resolve_label_index(self) -> None:
        """Find which index means 'fake', reading the config rather than guessing."""
        id2label = getattr(self.model.config, "id2label", {}) or {}
        mapping = {int(k): str(v).lower() for k, v in id2label.items()}
        for index, label in mapping.items():
            if any(token in label for token in ("fake", "false", "unreliable")):
                self.fake_index = index
                self.label_resolution = f"matched-label:{mapping[index]}"
                return
        for index, label in mapping.items():
            if any(token in label for token in ("true", "real", "reliable")):
                self.fake_index = 1 - index if len(mapping) == 2 else 0
                self.label_resolution = f"inverted-from:{mapping[index]}"
                return
        self.fake_index = 0
        self.label_resolution = (
            f"assumed-index-0; labels were {mapping}. Verify before relying on this."
        )
        logger.warning(
            "Style classifier labels %s are uninformative; assuming index 0 = fake.",
            mapping,
        )

    def _classify(self, title: str, text: str) -> Optional[float]:
        if not self.is_loaded:
            return None
        import torch

        try:
            prompt = config.STYLE_INPUT_TEMPLATE.format(title=title or "", text=text or "")
            with torch.no_grad():
                encoded = self.tokenizer(
                    prompt, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)[0]
            index = min(self.fake_index, probabilities.shape[-1] - 1)
            return float(probabilities[index].item())
        except Exception as exc:
            logger.warning("Style classification failed: %s", exc)
            return None

    def assess(self, text: str, title: str = "") -> StyleAssessment:
        """Combine the classifier (when available) with the heuristics."""
        heuristic_score, signals, observations = sensationalism_heuristics(
            f"{title} {text}".strip()
        )
        classifier_score = self._classify(title, text)

        if classifier_score is None:
            risk = heuristic_score
        else:
            # The heuristics are transparent and auditable; the classifier is a
            # black box trained on a dated corpus. They are weighted evenly and
            # neither is allowed to dominate.
            risk = 0.5 * classifier_score + 0.5 * heuristic_score

        return StyleAssessment(
            manipulation_risk=min(1.0, max(0.0, risk)),
            classifier_fake_style=classifier_score,
            heuristic_score=heuristic_score,
            signals=signals,
            observations=observations,
            model_name=self.model_name if classifier_score is not None else None,
        )

    def info(self) -> dict:
        return {
            "enabled": config.ENABLE_STYLE_CLASSIFIER,
            "loaded": self.is_loaded,
            "model": self.model_name,
            "fake_label_index": self.fake_index,
            "label_resolution": self.label_resolution,
            "load_error": self.load_error,
            "affects_verdict": config.STYLE_AFFECTS_VERDICT,
        }
