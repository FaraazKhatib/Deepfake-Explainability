"""
HTTP routes.

As in the sibling module, routes never load a model. The single, already-loaded
pipeline is pulled off ``request.app.state`` through a dependency.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from .. import config
from ..exceptions import ModelLoadError
from ..pipeline import FactCheckPipeline
from ..retrieval.base import clear_cache
from ..schemas.response import (
    CheckNewsRequest,
    CheckNewsResponse,
    ErrorResponse,
    HealthResponse,
    SourceRatingResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request) -> FactCheckPipeline:
    """Return the process-wide pipeline, or fail with a clean 503."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None or not pipeline.is_ready:
        raise ModelLoadError(
            "The fact-check models are still loading or failed to load. "
            "Check GET /health for details."
        )
    return pipeline


def _error_example(code: str, message: str, details: dict) -> dict:
    """Per-status example, so /docs does not show one body for every code."""
    return {
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {"error": code, "message": message, "details": details}
            }
        },
    }


_ERROR_RESPONSES = {
    400: {
        "description": "Input missing, too short, or over the size limit",
        **_error_example(
            "invalid_input",
            "Input is too short to check: 2 words (minimum 4).",
            {"word_count": 2, "minimum": 4},
        ),
    },
    422: {
        "description": "Nothing verifiable in the text, or malformed request body",
        **_error_example(
            "no_checkable_claim",
            "No verifiable factual claim was found. The text appears to be "
            "opinion, questions, or too vague to check against evidence.",
            {"sentences_examined": 3},
        ),
    },
    503: {
        "description": "Models still loading or failed to load - see GET /health",
        **_error_example(
            "model_unavailable",
            "The fact-check models are still loading or failed to load. "
            "Check GET /health for details.",
            {},
        ),
    },
}


@router.post(
    "/check-news",
    response_model=CheckNewsResponse,
    responses=_ERROR_RESPONSES,
    summary="Check a claim or article against live news and reference sources",
)
def check_news(
    payload: CheckNewsRequest,
    pipeline: FactCheckPipeline = Depends(get_pipeline),
):
    """
    Retrieve current evidence for the claim(s) in the text and report what it says.

    This performs live network retrieval, so latency depends on the upstream
    sources (typically 2-8 seconds).

    The result is an evidence report, not a determination of truth. "Unverified"
    is a legitimate outcome and is the correct answer for a story too new or too
    local to have been corroborated yet.
    """
    return pipeline.check(
        payload.text,
        max_claims=payload.max_claims,
        treat_as_single_claim=payload.single_claim,
        include_evidence=payload.include_evidence,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and model-state probe",
)
def health(request: Request):
    """Always 200, so 'alive but degraded' is distinguishable from 'dead'."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return {
            "status": "loading",
            "ready": False,
            "pipeline_version": "unknown",
            "startup_error": getattr(request.app.state, "startup_error", None),
        }

    info = pipeline.info()
    degraded = (
        not info.get("ready")
        or info.get("stance", {}).get("used_fallback")
        or not info.get("style", {}).get("loaded", False)
    )
    info["status"] = "ok" if not degraded else ("degraded" if info["ready"] else "loading")
    info["startup_error"] = getattr(request.app.state, "startup_error", None)
    return info


@router.get(
    "/sources/{domain}",
    response_model=SourceRatingResponse,
    summary="Show how a domain is rated and why",
)
def source_rating(domain: str, pipeline: FactCheckPipeline = Depends(get_pipeline)):
    """
    Inspect the credibility rating applied to a domain.

    Exposed deliberately: source weighting changes verdicts, so it should be
    auditable rather than buried. Unlisted domains return the unrated default,
    which is not a judgement that they are unreliable.
    """
    return pipeline.credibility.lookup(domain).to_dict()


@router.post("/cache/clear", summary="Drop cached upstream responses")
def clear_retrieval_cache():
    """Force the next check to hit the live sources again."""
    clear_cache()
    return {"cleared": True, "cache_ttl_seconds": config.CACHE_TTL_SECONDS}


@router.get("/", include_in_schema=False)
def index():
    return {
        "service": "Real-time news fact-checking module",
        "docs": "/docs",
        "endpoints": [
            "POST /check-news",
            "GET /health",
            "GET /sources/{domain}",
            "POST /cache/clear",
        ],
        "method": "evidence retrieval (GDELT, Google News RSS, Wikipedia) "
        "+ NLI stance detection",
        "disclaimer": (
            "Reports what sources say; does not determine truth. "
            "'Unverified' is a legitimate result."
        ),
    }
