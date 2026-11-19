"""
HTTP routes.

As in the sibling module, routes never load a model. The single, already-loaded
pipeline is pulled off ``request.app.state`` through a dependency.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import config
from ..feedback import VerdictFeedback, save_feedback, get_all_feedback, review_report
from ..exceptions import ModelLoadError
from ..pipeline import FactCheckPipeline
from ..retrieval.base import clear_cache
from ..translation import translate_to_english
from ..schemas.response import (
    CheckNewsRequest,
    TranslateRequest,
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
        source_language=payload.source_language,
        strict_mode=payload.strict_mode,
        page_url=payload.page_url,
    )





@router.post("/feedback", summary="Save a verdict correction for review")
def submit_feedback(report: VerdictFeedback):
    try:
        save_feedback(report)
    except (OSError, sqlite3.Error) as exc:
        logger.exception("Could not save verdict feedback")
        raise HTTPException(503, "Feedback storage is unavailable. Please retry.") from exc
    return {"status": "saved", "scan_id": str(report.scan_id)}


class ReviewActionRequest(BaseModel):
    scan_id: str
    action: Literal["approve", "revoke", "dismiss"]
    source: str = ""
    reason: str = ""


def _require_collaborator_key(request: Request) -> None:
    provided = request.headers.get("X-Admin-Key") or request.query_params.get("admin_key")
    if not provided or provided.strip() != config.ADMIN_SECRET_KEY:
        raise HTTPException(
            status_code=403,
            detail="Collaborator access denied. Invalid or missing administrator secret key."
        )


@router.post("/feedback/verify-key", summary="Verify collaborator admin key")
def verify_admin_key(request: Request):
    _require_collaborator_key(request)
    return {"status": "ok", "message": "Collaborator key verified"}


@router.get("/feedback/list", summary="List all feedback submissions for collaborator review")
def list_feedback(request: Request):
    _require_collaborator_key(request)
    try:
        items = get_all_feedback()
        return {"items": items, "total": len(items)}
    except Exception as exc:
        logger.exception("Failed to query feedback database")
        raise HTTPException(500, f"Error retrieving feedback list: {exc}")


@router.post("/feedback/review", summary="Collaborator review action (approve, revoke, dismiss)")
def process_feedback_review(payload: ReviewActionRequest, request: Request):
    _require_collaborator_key(request)
    try:
        review_report(
            scan_id=payload.scan_id,
            action=payload.action,
            source=payload.source,
            reason=payload.reason,
        )
        return {
            "status": "ok",
            "scan_id": payload.scan_id,
            "action": payload.action,
            "message": f"Successfully performed action '{payload.action}' on report {payload.scan_id}."
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Failed to process feedback review")
        raise HTTPException(500, f"Internal error during feedback review: {exc}")


@router.post(
    "/translate",
    summary="Translate text to English with automatic language detection",
)
def translate(payload: TranslateRequest):
    """Detect language and translate to English."""
    return translate_to_english(payload.text, source_hint=payload.source_language or "")


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
