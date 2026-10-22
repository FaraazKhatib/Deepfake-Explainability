"""Pydantic v2 request/response models -- the public contract of the service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .. import config


class CheckNewsRequest(BaseModel):
    """Body of POST /check-news."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The World Health Organization declared a new global health emergency yesterday.",
                "single_claim": True,
            }
        }
    )

    text: str = Field(
        ...,
        description="A headline, claim, paragraph or full article to check. "
        f"Minimum {config.MIN_INPUT_WORDS} words.",
    )
    single_claim: bool = Field(
        False,
        description="Treat the whole input as one claim instead of extracting "
        "several. Use for headlines and typed assertions.",
    )
    max_claims: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description=f"How many claims to check (default {config.MAX_CLAIMS}).",
    )
    include_evidence: bool = Field(
        True, description="Include the full list of cited sources per claim."
    )


class ClaimResult(BaseModel):
    """Outcome for one claim."""

    model_config = ConfigDict(extra="allow")

    claim: str
    verdict: str
    confidence: str
    supporting_source_count: int
    refuting_source_count: int
    evidence_examined: int
    explanation: str


class CheckNewsResponse(BaseModel):
    """Body of a successful POST /check-news."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "verdict": "Supported by available evidence",
                "rationale": "Every checked claim was corroborated by independent sources.",
                "claim_count": 1,
                "supported_claims": 1,
                "refuted_claims": 0,
                "summary": "Supported by available evidence across 1 claim(s), based on 5 independent source(s).",
            }
        },
    )

    verdict: str = Field(
        ...,
        description="'Supported by available evidence', 'Contradicted by "
        "available evidence', 'Disputed by available evidence', or 'Unverified'.",
    )
    rationale: str
    claim_count: int
    supported_claims: int
    refuted_claims: int
    disputed_claims: int
    unverified_claims: int
    explanation: str
    summary: str
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    manipulation_risk: Optional[Dict[str, Any]] = Field(
        None,
        description="Presentation-style assessment. Reported separately and, by "
        "default, unable to affect the verdict.",
    )
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Body of GET /health."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(..., description="'ok', 'degraded' or 'loading'.")
    ready: bool
    pipeline_version: str
    stance: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    credibility: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, Any] = Field(default_factory=dict)


class SourceRatingResponse(BaseModel):
    """Body of GET /sources/{domain}."""

    model_config = ConfigDict(extra="allow")

    domain: str
    tier: str
    weight: float
    rationale: str
    explicitly_rated: bool


class ErrorResponse(BaseModel):
    """Uniform error body for every non-2xx response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "no_checkable_claim",
                "message": "No verifiable factual claim was found.",
                "details": {"sentences_examined": 3},
            }
        }
    )

    error: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
