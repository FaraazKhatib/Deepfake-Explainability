"""
Human-readable, simplified explanation of a fact-check result.
Provides clear, plain-English explanations that everyday users can easily understand.
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
    "Note: This is an automated check of currently indexed news and may not reflect "
    "breaking or developing events."
)


def _outlet_list(stances, limit: int = 3) -> str:
    """Format a readable list of news outlets."""
    names = [d.outlet or d.domain for d in stances[:limit]]
    if not names:
        return ""
    if len(stances) > limit:
        return ", ".join(names) + f" and {len(stances) - limit} others"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def explain_claim(verdict: ClaimVerdict) -> str:
    """Narrative for one claim in simple, clear English."""
    parts: List[str] = []

    if verdict.verdict == VERDICT_SUPPORTED:
        outlets = _outlet_list(verdict.supporting_domains)
        source_str = f" ({outlets})" if outlets else ""
        if verdict.refuting_domains:
            parts.append(
                f"Supported: Verified by {len(verdict.supporting_domains)} source(s){source_str}, "
                "though a few sources raised counter-points."
            )
        else:
            parts.append(
                f"Supported: Confirmed by {len(verdict.supporting_domains)} credible source(s){source_str}."
            )

    elif verdict.verdict == VERDICT_REFUTED:
        outlets = _outlet_list(verdict.refuting_domains)
        source_str = f" ({outlets})" if outlets else ""
        if verdict.supporting_domains:
            parts.append(
                f"Contradicted: Debunked or refuted by {len(verdict.refuting_domains)} source(s){source_str}, "
                "though some conflicting reports exist."
            )
        else:
            parts.append(
                f"Contradicted: Debunked or contradicted by {len(verdict.refuting_domains)} credible source(s){source_str}."
            )

    elif verdict.verdict == VERDICT_DISPUTED:
        parts.append(
            f"Disputed: Sources disagree on this claim. {len(verdict.supporting_domains)} source(s) support it "
            f"while {len(verdict.refuting_domains)} source(s) contradict it."
        )

    else:
        # Unverified
        if not verdict.supporting_domains and not verdict.refuting_domains:
            parts.append(
                "Unverified: No credible news sources were found confirming or denying this claim."
            )
        else:
            parts.append(
                "Unverified: Found limited mentions, but not enough verified reporting to confirm or deny this claim."
            )

    if verdict.notes:
        parts.extend(verdict.notes)

    return " ".join(p for p in parts if p)


def explain_overall(
    verdicts: List[ClaimVerdict],
    retrieval: RetrievalReport,
    style: Optional[StyleAssessment] = None,
) -> str:
    """Document-level narrative written in plain English for regular readers."""
    parts: List[str] = []

    n_claims = len(verdicts)
    refuted = sum(1 for v in verdicts if v.verdict == VERDICT_REFUTED)
    supported = sum(1 for v in verdicts if v.verdict == VERDICT_SUPPORTED)
    disputed = sum(1 for v in verdicts if v.verdict == VERDICT_DISPUTED)
    unverified = sum(1 for v in verdicts if v.verdict == VERDICT_UNVERIFIED)

    # 1. Main finding summary
    if refuted > 0:
        if refuted == n_claims:
            parts.append(
                f"We checked {n_claims} key statement(s) against current public news sources. "
                "All checked statements were contradicted or debunked by credible reporting."
            )
        else:
            parts.append(
                f"We checked {n_claims} key statement(s) against current public news sources. "
                f"{refuted} statement(s) were contradicted by credible reporting."
            )
    elif disputed > 0:
        parts.append(
            f"We checked {n_claims} key statement(s) against current public news sources. "
            "Credible news outlets report conflicting information on this topic."
        )
    elif supported > 0 and unverified == 0:
        parts.append(
            f"We checked {n_claims} key statement(s) against current public news sources. "
            "Independent reporting confirms the main claims."
        )
    elif supported > 0 and unverified > 0:
        parts.append(
            f"We checked {n_claims} key statement(s) against current public news sources. "
            f"{supported} statement(s) were confirmed by sources, while {unverified} statement(s) "
            "could not be verified."
        )
    else:
        # All unverified
        parts.append(
            f"We checked {n_claims} key statement(s) against current public news and web sources. "
            "None of the statements could be confirmed or debunked by credible reporting. "
            "This usually happens when a story is satire, fictional, very new, or simply "
            "not covered by established news outlets."
        )

    # 2. Writing style note if notable
    if style is not None:
        if style.manipulation_risk >= 0.6:
            parts.append(
                "Writing style note: The text uses emotional or sensational wording often found in clickbait."
            )
        elif style.manipulation_risk <= 0.2 and style.observations:
            parts.append(
                "Writing style note: The text uses neutral, straightforward language."
            )

    # 3. Simple disclaimer
    parts.append(_DISCLAIMER)

    return "\n\n".join(parts)


def short_summary(overall: dict, verdicts: List[ClaimVerdict]) -> str:
    """One-line summary for compact cards and extension UI."""
    total_sources = sum(
        len(v.supporting_domains) + len(v.refuting_domains) for v in verdicts
    )
    status = overall.get("overall", VERDICT_UNVERIFIED)
    n_claims = overall.get("claim_count", len(verdicts))

    if status == VERDICT_SUPPORTED:
        return f"Verified: Key claims are supported by {total_sources} independent news source(s)."
    elif status == VERDICT_REFUTED:
        return "Likely False: Key claims are contradicted or debunked by independent news reports."
    elif status == VERDICT_DISPUTED:
        return "Disputed: News sources report conflicting details on these claims."
    else:
        if total_sources > 0:
            return f"Unverified: Checked {n_claims} claim(s), but there is not enough confirmed reporting to verify this story."
        return "Unverified: No credible news reports were found covering these claims."
