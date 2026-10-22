"""
Natural Language Inference stance detection -- the core of the system.

For every retrieved snippet the model is asked: taking this snippet as the
premise, does the claim follow (entailment), is it contradicted
(contradiction), or is the snippet simply not about it (neutral)?

Model: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``. FEVER is a fact
verification corpus built around exactly this "claim vs retrieved evidence"
shape, which is why it is preferred over a plain MNLI checkpoint.

WHAT ENTAILMENT DOES AND DOES NOT MEAN
--------------------------------------
Entailment means "this source asserts something consistent with the claim". It
does NOT mean the claim is true. If a false story is reported by many outlets,
this model will find entailment in all of them -- as would a human reading the
same pages. The system's protection against that is source diversity and
credibility weighting downstream, not the NLI model, and even that protection
is partial. This limitation is stated in every response.

LABEL ORDER IS RESOLVED, NOT ASSUMED
------------------------------------
The three supported checkpoints order their labels differently:

    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli -> {0: entailment, 1: neutral, 2: contradiction}
    facebook/bart-large-mnli                     -> {0: contradiction, 1: neutral, 2: entailment}
    cross-encoder/nli-deberta-v3-base            -> {0: contradiction, 1: entailment, 2: neutral}

Hardcoding an index would silently invert supports and refutes on two of the
three. Indices are therefore read from ``model.config.id2label``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config
from ..exceptions import InferenceError, ModelLoadError
from ..retrieval.base import Evidence
from ..retrieval.relevance import is_relevant
from .debunk import apply_debunk_override, detect_reported_speech

logger = logging.getLogger(__name__)

STANCE_SUPPORTS = "supports"
STANCE_REFUTES = "refutes"
STANCE_UNRELATED = "unrelated"


@dataclass
class StanceResult:
    """One evidence item judged against one claim."""

    evidence: Evidence
    stance: str
    entailment: float
    neutral: float
    contradiction: float
    relevance: float = 1.0
    gated: bool = False        # True when the relevance gate overrode the model
    debunk_detected: bool = False   # lexical debunk override fired
    debunk_terms: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Strength of the side taken (or of the 'unrelated' judgement)."""
        if self.stance == STANCE_SUPPORTS:
            return self.entailment
        if self.stance == STANCE_REFUTES:
            return self.contradiction
        return self.neutral

    def to_dict(self) -> dict:
        payload = self.evidence.to_dict()
        payload.update(
            {
                "stance": self.stance,
                "confidence": round(self.confidence, 4),
                "probabilities": {
                    "entailment": round(self.entailment, 4),
                    "neutral": round(self.neutral, 4),
                    "contradiction": round(self.contradiction, 4),
                },
                "relevance_overlap": round(self.relevance, 4),
                "relevance_gated": self.gated,
                "debunk_detected": self.debunk_detected,
                "debunk_terms": self.debunk_terms,
            }
        )
        return payload


class StanceDetector:
    """Loads the NLI model once and scores (claim, evidence) pairs in batches."""

    def __init__(
        self,
        model_name: str = config.NLI_MODEL,
        fallback_model_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        max_length: int = config.NLI_MAX_LENGTH,
        batch_size: int = config.NLI_BATCH_SIZE,
    ):
        self.requested_model_name = model_name
        self.fallback_model_names = (
            list(config.NLI_FALLBACK_MODELS)
            if fallback_model_names is None
            else list(fallback_model_names)
        )
        self.model_name: Optional[str] = None
        self.used_fallback = False
        self.device = device or config.resolve_device()
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = None
        self.model = None
        self.label_indices = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.label_resolution = "not-loaded"
        self.load_errors: List[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the NLI model, walking the fallback list if necessary."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError(
                "PyTorch and transformers must be installed: "
                "pip install -r requirements.txt"
            ) from exc

        candidates = [self.requested_model_name] + [
            m for m in self.fallback_model_names if m != self.requested_model_name
        ]

        for position, name in enumerate(candidates):
            try:
                logger.info("Loading NLI model '%s' ...", name)
                kwargs = {}
                if config.HF_CACHE_DIR:
                    kwargs["cache_dir"] = config.HF_CACHE_DIR
                if config.OFFLINE_ONLY:
                    kwargs["local_files_only"] = True
                if config.HF_TOKEN:
                    kwargs["token"] = config.HF_TOKEN

                tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
                model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
            except Exception as exc:
                self.load_errors.append(f"{name}: {type(exc).__name__}: {exc}")
                logger.warning("Could not load NLI model '%s': %s", name, exc)
                continue

            model.eval()
            model.to(self.device)
            self.tokenizer = tokenizer
            self.model = model
            self.model_name = name
            self.used_fallback = position > 0
            try:
                self._resolve_label_indices()
            except ModelLoadError as exc:
                self.load_errors.append(f"{name}: {exc}")
                self.model = self.tokenizer = None
                self.model_name = None
                continue
            logger.info(
                "Stance detector ready: model=%s device=%s indices=%s",
                name, self.device, self.label_indices,
            )
            return

        raise ModelLoadError(
            "Could not load any NLI model. Attempts: " + " | ".join(self.load_errors),
            attempted=candidates,
        )

    def _resolve_label_indices(self) -> None:
        """Read entailment/neutral/contradiction positions from the model config."""
        id2label = getattr(self.model.config, "id2label", {}) or {}
        mapping = {int(k): str(v).lower() for k, v in id2label.items()}

        resolved = {}
        for index, label in mapping.items():
            if "entail" in label:
                resolved["entailment"] = index
            elif "contradict" in label:
                resolved["contradiction"] = index
            elif "neutral" in label:
                resolved["neutral"] = index

        if len(resolved) == 3:
            self.label_indices = resolved
            self.label_resolution = f"resolved-from-config:{mapping}"
        else:
            raise ModelLoadError(f"Cannot resolve NLI label order safely: {mapping}")

    # ------------------------------------------------------------------
    def classify(self, claim: str, evidence: List[Evidence]) -> List[StanceResult]:
        """Judge each evidence item against the claim."""
        if not self.is_loaded:
            raise ModelLoadError("Stance model is not loaded. Call load() first.")
        if not evidence:
            return []

        import torch

        # A claim phrased as reported speech ("a story circulated that X") can
        # be TRUE while X is false. The debunk override is suppressed for such
        # claims so the system does not silently check a different statement.
        reported_framing = detect_reported_speech(claim)

        # Premise = the retrieved snippet, hypothesis = the claim under test.
        premises = [item.premise for item in evidence]
        hypotheses = [claim] * len(premises)

        results: List[StanceResult] = []
        try:
            with torch.no_grad():
                for start in range(0, len(premises), self.batch_size):
                    batch_premises = premises[start : start + self.batch_size]
                    batch_hypotheses = hypotheses[start : start + self.batch_size]

                    encoded = self.tokenizer(
                        batch_premises,
                        batch_hypotheses,
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    logits = self.model(**encoded).logits
                    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

                    for offset, probs in enumerate(probabilities):
                        item = evidence[start + offset]
                        entail = float(probs[self.label_indices["entailment"]])
                        neutral = float(probs[self.label_indices["neutral"]])
                        contra = float(probs[self.label_indices["contradiction"]])
                        stance = self._decide(entail, contra)

                        # Relevance gate: an NLI model handed an unrelated
                        # premise often returns a confident contradiction.
                        # Without shared vocabulary the judgement is not
                        # trustworthy, so it is demoted to "unrelated" and the
                        # override is recorded rather than hidden.
                        relevant, overlap = is_relevant(claim, item.premise)
                        gated = False
                        if stance != STANCE_UNRELATED and not relevant:
                            stance = STANCE_UNRELATED
                            gated = True

                        # Debunk override: NLI scores explicit debunking as
                        # neutral, so a source that says the claim is false
                        # would otherwise contribute nothing at all.
                        override = apply_debunk_override(
                            stance, contra, claim, item.premise,
                            reported_framing, refutes_label=STANCE_REFUTES,
                        )
                        stance = override.stance
                        contra = override.contradiction
                        debunk_terms = override.terms
                        debunk_fired = override.fired
                        if debunk_fired:
                            gated = False

                        results.append(
                            StanceResult(
                                evidence=item,
                                stance=stance,
                                entailment=entail,
                                neutral=neutral,
                                contradiction=contra,
                                relevance=overlap,
                                gated=gated,
                                debunk_detected=debunk_fired,
                                debunk_terms=debunk_terms,
                            )
                        )
        except Exception as exc:
            raise InferenceError(f"Stance inference failed: {exc}") from exc

        return results

    @staticmethod
    def _decide(entailment: float, contradiction: float) -> str:
        """
        Turn probabilities into a stance.

        A snippet must clear STANCE_MIN_CONFIDENCE before it counts as taking
        a side at all. Most retrieved items are genuinely unrelated -- they
        matched the keywords without addressing the claim -- and letting weak
        signals vote is how these systems manufacture false consensus.
        """
        threshold = config.STANCE_MIN_CONFIDENCE
        if entailment >= threshold and entailment > contradiction:
            return STANCE_SUPPORTS
        if contradiction >= threshold and contradiction > entailment:
            return STANCE_REFUTES
        return STANCE_UNRELATED

    def info(self) -> dict:
        return {
            "requested_model": self.requested_model_name,
            "loaded_model": self.model_name,
            "used_fallback": self.used_fallback,
            "device": self.device,
            "label_indices": self.label_indices,
            "label_resolution": self.label_resolution,
            "min_stance_confidence": config.STANCE_MIN_CONFIDENCE,
            "min_relevance_overlap": config.MIN_RELEVANCE_OVERLAP,
            "loaded": self.is_loaded,
            "load_errors": self.load_errors,
        }
