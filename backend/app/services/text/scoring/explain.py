"""
Human-readable explanation of a fact-check result.

Same rule as the text-detector module: every sentence must be tied to
something actually retrieved or computed in this request. No template fires
without its evidence, and the explanation always names real outlets and counts
rather than asserting a conclusion the pipeline did not establish.
"""

from __future__ import annotations

from typing import List, Optional

from ..retrieval.base import RetrievalReport
from ..style.classifier import StyleAssessment
from .verdict import (
    VERDICT_DISPUTED,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    VERDICT_UNVERIFIED,
    ClaimVerdict,
)

_DISCLAIMER = (
    "This is an automated evidence check, not a determination of truth. It "
    "reports what a set of searchable sources say right now; sources can be "
    "wrong, can repeat each other, and may not yet cover a developing story."
)


def _outlet_list(stances, limit: int = 4) -> str:
    names = [d.outlet or d.domain for d in stances[:limit]]
    if not names:
        return ""
    if len(stances) > limit:
        return ", ".join(names) + f" and {len(stances) - limit} more"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def explain_claim(verdict: ClaimVerdict) -> str:
    """Narrative for one claim, grounded in the sources actually found."""
    parts: List[str] = []

    if verdict.verdict == VERDICT_SUPPORTED:
        parts.append(
            f"{len(verdict.supporting_domains)} independent source(s) "
            f"({_outlet_list(verdict.supporting_domains)}) published material "
            "consistent with this claim."
        )
        if verdict.refuting_domains:
            parts.append(
                f"{len(verdict.refuting_domains)} source(s) "
                f"({_outlet_list(verdict.refuting_domains)}) appeared to "
                "contradict it, but carried less weight."
            )
    elif verdict.verdict == VERDICT_REFUTED:
        parts.append(
            f"{len(verdict.refuting_domains)} independent source(s) "
            f"({_outlet_list(verdict.refuting_domains)}) published material that "
            "contradicts this claim."
        )
        if verdict.supporting_domains:
            parts.append(
                f"{len(verdict.supporting_domains)} source(s) appeared to support "
                "it, so read both sets below."
            )
    elif verdict.verdict == VERDICT_DISPUTED:
        parts.append(
            f"Sources are split: {len(verdict.supporting_domains)} "
            f"({_outlet_list(verdict.supporting_domains)}) are consistent with "
            f"the claim and {len(verdict.refuting_domains)} "
            f"({_outlet_list(verdict.refuting_domains)}) contradict it."
        )
    else:
        parts.append(
            f"Of {verdict.evidence_examined} retrieved item(s) across "
            f"{verdict.independent_domains} domain(s), none clearly took a "
            "position on this claim."
            if not verdict.supporting_domains and not verdict.refuting_domains
            else (
                f"Only {len(verdict.supporting_domains) + len(verdict.refuting_domains)} "
                "source(s) addressed the claim, which is below the threshold "
                "needed to report a verdict."
            )
        )

    if verdict.unrelated_count:
        parts.append(
            f"{verdict.unrelated_count} retrieved item(s) matched the search "
            "terms but did not address the claim itself."
        )

    parts.extend(verdict.notes)
    return " ".join(p for p in parts if p)


def explain_overall(
    verdicts: List[ClaimVerdict],
    retrieval: RetrievalReport,
    style: Optional[StyleAssessment] = None,
) -> str:
    """Document-level narrative, including retrieval coverage and style."""
    parts: List[str] = []

    sources_used = ", ".join(
        f"{name} ({count})" for name, count in retrieval.per_source_counts.items()
    )
    parts.append(
        f"Checked {len(verdicts)} claim(s) against {len(retrieval.evidence)} "
        f"evidence item(s) retrieved from: {sources_used or 'no source'}."
    )

    if retrieval.errors:
        failed = ", ".join(retrieval.errors.keys())
        parts.append(
            f"Evidence coverage was reduced: {failed} did not respond, so this "
            "check saw less of the record than usual."
        )
    # Notices are working behaviour, not failures, and are worded as such.
    for notice in retrieval.notices.values():
        parts.append(notice)

    counts = {}
    for verdict in verdicts:
        counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
    if counts:
        summary = "; ".join(f"{count} {label.lower()}" for label, count in counts.items())
        parts.append(f"Per-claim outcome: {summary}.")

    if style is not None:
        if style.manipulation_risk >= 0.6:
            detail = f" ({'; '.join(style.observations[:3])})" if style.observations else ""
            parts.append(
                f"Separately, the writing style scores {style.manipulation_risk:.2f} "
                f"on presentation-manipulation markers{detail}. This describes how "
                "the text is written, not whether it is true, and it did not "
                "affect the verdict above."
            )
        elif style.observations:
            parts.append(
                f"Presentation-manipulation score is low ({style.manipulation_risk:.2f}); "
                "note that a calm, professional style is not evidence of accuracy."
            )

    parts.append(_DISCLAIMER)
    return " ".join(p for p in parts if p)


def short_summary(overall: dict, verdicts: List[ClaimVerdict]) -> str:
    """One-line summary for compact UIs."""
    total_sources = sum(
        len(v.supporting_domains) + len(v.refuting_domains) for v in verdicts
    )
    return (
        f"{overall['overall']} across {overall['claim_count']} claim(s), "
        f"based on {total_sources} independent source(s). "
        "Automated evidence check, not a determination of truth."
    )
