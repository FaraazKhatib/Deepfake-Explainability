"""
Turn per-snippet stances into a verdict about a claim.

The central problem this solves is that ARTICLE COUNT IS NOT SOURCE COUNT.
Wire copy is syndicated verbatim across dozens of sites; aggregators republish;
outlets rewrite each other. A system that counts articles will report
overwhelming consensus from what is really one newsroom. So:

    stances are collapsed to ONE per domain (the most confident), and the
    verdict requires MIN_INDEPENDENT_SOURCES distinct domains on a side.

Each domain then contributes:

    weight = credibility_weight * stance_confidence * recency_factor

Verdict space is four-valued, and "Unverified" is a real answer rather than a
failure. For genuinely breaking news, no corroboration existing yet is the
correct output; forcing a true/false call there is how these systems cause harm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config
from ..credibility.sources import CredibilityIndex, SourceRating
from ..stance.nli import STANCE_REFUTES, STANCE_SUPPORTS, StanceResult

VERDICT_SUPPORTED = "Supported by available evidence"
VERDICT_REFUTED = "Contradicted by available evidence"
VERDICT_DISPUTED = "Disputed by available evidence"
VERDICT_UNVERIFIED = "Unverified"


@dataclass
class DomainStance:
    """One domain's single strongest position on the claim."""

    domain: str
    outlet: str
    stance: str
    confidence: float
    rating: SourceRating
    recency_factor: float
    weight: float
    url: str
    title: str
    age_days: Optional[float]
    # Carried through from the stance stage so a verdict reached via the
    # lexical debunk override is visibly labelled as such.
    debunk_detected: bool = False
    debunk_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "outlet": self.outlet,
            "stance": self.stance,
            "stance_confidence": round(self.confidence, 4),
            "credibility": self.rating.to_dict(),
            "recency_factor": round(self.recency_factor, 4),
            "contributed_weight": round(self.weight, 4),
            "title": self.title,
            "url": self.url,
            "age_days": round(self.age_days, 2) if self.age_days is not None else None,
            "debunk_detected": self.debunk_detected,
            "debunk_terms": self.debunk_terms,
        }


@dataclass
class ClaimVerdict:
    """The outcome for a single claim."""

    claim: str
    verdict: str
    confidence: str
    support_weight: float
    refute_weight: float
    supporting_domains: List[DomainStance] = field(default_factory=list)
    refuting_domains: List[DomainStance] = field(default_factory=list)
    unrelated_count: int = 0
    evidence_examined: int = 0
    independent_domains: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self, include_evidence: bool = True) -> dict:
        payload = {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "support_weight": round(self.support_weight, 4),
            "refute_weight": round(self.refute_weight, 4),
            "supporting_source_count": len(self.supporting_domains),
            "refuting_source_count": len(self.refuting_domains),
            "unrelated_evidence_count": self.unrelated_count,
            "evidence_examined": self.evidence_examined,
            "independent_domains": self.independent_domains,
            "notes": self.notes,
        }
        if include_evidence:
            payload["supporting_sources"] = [d.to_dict() for d in self.supporting_domains]
            payload["refuting_sources"] = [d.to_dict() for d in self.refuting_domains]
        return payload


def recency_factor(age_days: Optional[float]) -> float:
    """
    Exponential decay by age, floored at RECENCY_MIN_FACTOR.

    Old coverage still counts -- background facts do not expire -- but a
    five-year-old article should not settle a claim about last week. Undated
    evidence (e.g. Wikipedia) is treated as neutral rather than penalised.
    """
    if age_days is None:
        return 1.0
    halflife = max(0.1, config.RECENCY_HALFLIFE_DAYS)
    decayed = math.pow(0.5, age_days / halflife)
    return max(config.RECENCY_MIN_FACTOR, decayed)


def collapse_to_domains(
    stances: List[StanceResult], credibility: CredibilityIndex
) -> Dict[str, DomainStance]:
    """
    Keep one position per domain: the one the model was most confident about.

    This is the anti-syndication defence. Fifty copies of the same wire story
    become one vote from that wire service.
    """
    best: Dict[str, DomainStance] = {}

    for result in stances:
        if result.stance not in (STANCE_SUPPORTS, STANCE_REFUTES):
            continue

        evidence = result.evidence
        domain = evidence.domain or "unknown"
        rating = credibility.lookup(domain)
        recency = recency_factor(evidence.age_days)
        weight = rating.weight * result.confidence * recency

        candidate = DomainStance(
            domain=domain,
            outlet=evidence.outlet or domain,
            stance=result.stance,
            confidence=result.confidence,
            rating=rating,
            recency_factor=recency,
            weight=weight,
            url=evidence.url,
            title=evidence.title,
            age_days=evidence.age_days,
            debunk_detected=getattr(result, "debunk_detected", False),
            debunk_terms=list(getattr(result, "debunk_terms", [])),
        )

        existing = best.get(domain)
        if existing is None or candidate.confidence > existing.confidence:
            best[domain] = candidate

    return best


def single_authoritative_source(sides: List[DomainStance]) -> Optional[DomainStance]:
    """
    Find a source strong enough to carry a verdict on its own.

    Only tiers whose evidence is not syndicated newsroom copy qualify, and only
    above a weight floor. See SINGLE_SOURCE_TIERS in config.py for the reasoning.
    """
    for stance in sides:
        if (
            stance.rating.tier in config.SINGLE_SOURCE_TIERS
            and stance.rating.weight >= config.SINGLE_SOURCE_MIN_CREDIBILITY
            and stance.confidence >= config.SINGLE_SOURCE_MIN_STANCE
        ):
            return stance
    return None


def _confidence_band(
    winning_weight: float, losing_weight: float, source_count: int
) -> str:
    """Confidence reflects evidence volume and one-sidedness, not the model alone."""
    if source_count == 0:
        return "None"
    margin = winning_weight - losing_weight
    if source_count >= 4 and margin >= 2.0:
        return "High"
    if source_count >= 2 and margin >= 0.8:
        return "Medium"
    return "Low"


def decide(
    claim: str,
    stances: List[StanceResult],
    credibility: CredibilityIndex,
) -> ClaimVerdict:
    """Aggregate stances into a verdict for one claim."""
    domain_stances = collapse_to_domains(stances, credibility)

    supporting = sorted(
        (d for d in domain_stances.values() if d.stance == STANCE_SUPPORTS),
        key=lambda d: -d.weight,
    )
    refuting = sorted(
        (d for d in domain_stances.values() if d.stance == STANCE_REFUTES),
        key=lambda d: -d.weight,
    )

    support_weight = sum(d.weight for d in supporting)
    refute_weight = sum(d.weight for d in refuting)
    unrelated = sum(
        1 for s in stances if s.stance not in (STANCE_SUPPORTS, STANCE_REFUTES)
    )

    notes: List[str] = []
    minimum = config.MIN_INDEPENDENT_SOURCES

    support_normal = (
        support_weight >= config.SUPPORT_THRESHOLD and len(supporting) >= minimum
    )
    refute_normal = (
        refute_weight >= config.REFUTE_THRESHOLD and len(refuting) >= minimum
    )

    # Narrow exception for non-syndicated authoritative sources.
    support_authority = None if support_normal else single_authoritative_source(supporting)
    refute_authority = None if refute_normal else single_authoritative_source(refuting)

    support_qualifies = support_normal or support_authority is not None
    refute_qualifies = refute_normal or refute_authority is not None
    via_single_source = (support_authority is not None and support_qualifies) or (
        refute_authority is not None and refute_qualifies
    )

    if support_qualifies and refute_qualifies:
        ratio = support_weight / max(refute_weight, 1e-6)
        if ratio >= config.DISPUTE_RATIO:
            verdict = VERDICT_SUPPORTED
        elif ratio <= 1 / config.DISPUTE_RATIO:
            verdict = VERDICT_REFUTED
        else:
            verdict = VERDICT_DISPUTED
            notes.append(
                "Sources disagree. Please read the evidence below."
            )
    elif support_qualifies:
        verdict = VERDICT_SUPPORTED
    elif refute_qualifies:
        verdict = VERDICT_REFUTED
    else:
        verdict = VERDICT_UNVERIFIED
        if not supporting and not refuting:
            notes.append(
                "No sources found for this claim. It might be too new, local, "
                "or phrased differently."
            )
        else:
            found = len(supporting) + len(refuting)
            notes.append(
                f"Not enough sources found ({found}) to make a verdict. "
                f"We need at least {minimum}."
            )

    if verdict == VERDICT_SUPPORTED:
        band = _confidence_band(support_weight, refute_weight, len(supporting))
    elif verdict == VERDICT_REFUTED:
        band = _confidence_band(refute_weight, support_weight, len(refuting))
    elif verdict == VERDICT_DISPUTED:
        band = "Low"
    else:
        band = "None"

    # A verdict resting on one source is never High confidence, and the
    # dependence is stated rather than buried.
    if via_single_source and verdict != VERDICT_UNVERIFIED:
        band = "Medium" if band == "High" else band
        only = support_authority or refute_authority
        notes.append(
            f"Based on only one source ({only.outlet}). Treat this with caution."
        )

    # Corroboration is not truth. Say so whenever we return "Supported".
    if verdict == VERDICT_SUPPORTED:
        notes.append(
            "Remember: Even widely reported stories can sometimes be wrong."
        )

    unrated = [d for d in (supporting + refuting) if not d.rating.is_rated]
    if unrated and len(unrated) == len(supporting) + len(refuting):
        notes.append(
            "All sources used are unrated. Our confidence in this verdict is lower."
        )

    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        confidence=band,
        support_weight=support_weight,
        refute_weight=refute_weight,
        supporting_domains=supporting,
        refuting_domains=refuting,
        unrelated_count=unrelated,
        evidence_examined=len(stances),
        independent_domains=len(domain_stances),
        notes=notes,
    )


def combine_claim_verdicts(verdicts: List[ClaimVerdict]) -> dict:
    """
    Roll several claim verdicts into one document-level summary.

    Deliberately conservative: a single well-evidenced refutation matters more
    than several unverified claims, because a document containing one
    demonstrably false central claim is a problem regardless of the rest.
    """
    if not verdicts:
        return {
            "overall": VERDICT_UNVERIFIED,
            "rationale": "No claims were checked.",
            "claim_count": 0,
        }

    refuted = [v for v in verdicts if v.verdict == VERDICT_REFUTED]
    disputed = [v for v in verdicts if v.verdict == VERDICT_DISPUTED]
    supported = [v for v in verdicts if v.verdict == VERDICT_SUPPORTED]
    unverified = [v for v in verdicts if v.verdict == VERDICT_UNVERIFIED]

    if refuted:
        overall = VERDICT_REFUTED
        rationale = (
            f"{len(refuted)} of {len(verdicts)} checked claim(s) were "
            "contradicted by independent sources."
        )
    elif disputed:
        overall = VERDICT_DISPUTED
        rationale = f"{len(disputed)} of {len(verdicts)} checked claim(s) are disputed."
    elif supported and not unverified:
        overall = VERDICT_SUPPORTED
        rationale = "Every checked claim was confirmed by independent sources."
    elif supported:
        # If the lead claim (headline) is supported or at least half the claims are confirmed,
        # and none are refuted or disputed, the main story is confirmed.
        if (verdicts and verdicts[0].verdict == VERDICT_SUPPORTED) or len(supported) >= len(unverified):
            overall = VERDICT_SUPPORTED
            rationale = (
                f"The main story is confirmed by independent sources "
                f"({len(supported)} claim(s) confirmed, {len(unverified)} secondary detail(s) unverified)."
            )
        else:
            overall = VERDICT_UNVERIFIED
            rationale = (
                f"{len(supported)} claim(s) confirmed, but {len(unverified)} could "
                "not be verified by independent sources."
            )
    else:
        overall = VERDICT_UNVERIFIED
        rationale = "None of the checked claims could be confirmed or refuted by current news sources."

    return {
        "overall": overall,
        "rationale": rationale,
        "claim_count": len(verdicts),
        "supported": len(supported),
        "refuted": len(refuted),
        "disputed": len(disputed),
        "unverified": len(unverified),
    }
