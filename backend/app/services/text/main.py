"""
FastAPI application factory for the real-time news fact-checking module.

Run locally with:

    uvicorn app.main:app --reload --port 8001

Port 8001 by convention, so it can run alongside the text_detector module on
8000 within the same multimodal system.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .api.routes import router
from .exceptions import FactCheckError
from .pipeline import FactCheckPipeline

logging.basicConfig(
    level=os.getenv("FC_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("fact_checker")

DESCRIPTION = """
Real-time news fact-checking module for a multimodal detection system.

Given a claim, headline or article, this service:

1. extracts the check-worthy claims;
2. searches **live** sources for evidence (GDELT, Google News RSS, Wikipedia);
3. uses an NLI model to judge whether each retrieved snippet **supports** or
   **refutes** the claim;
4. collapses evidence to one position per domain, weights it by source
   credibility and recency, and reports a verdict with every source cited.

**This reports what sources say; it does not determine truth.** Outlets repeat
each other, a widely reported story can still be wrong, and a story too new or
too local to be covered will correctly come back as *Unverified*.

A separate `manipulation_risk` score describes writing style. By design it
**cannot change the verdict** -- judging truth by writing style penalises
people for how they write rather than for what they assert.
"""


def create_app(pipeline: Optional[FactCheckPipeline] = None) -> FastAPI:
    """Build the app. Passing a pipeline skips model loading (used by tests)."""
    injected = pipeline is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        if injected:
            app.state.pipeline = pipeline
            logger.info("Using an injected pipeline; skipping model loading.")
        else:
            app.state.pipeline = None
            try:
                built = FactCheckPipeline()
                logger.info(
                    "Loading models (NLI='%s', device='%s'). First run downloads "
                    "weights and may take several minutes.",
                    config.NLI_MODEL,
                    config.resolve_device(),
                )
                built.load()
                app.state.pipeline = built
            except Exception as exc:
                app.state.startup_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Model loading failed: %s. The API will start but /check-news "
                    "will return 503.", exc,
                )
        yield
        app.state.pipeline = None

    app = FastAPI(
        title="Real-Time News Fact-Checking Module",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    allowed = os.getenv("FC_CORS_ORIGINS", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allowed == "*" else [o.strip() for o in allowed.split(",")],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(FactCheckError)
    async def fact_check_error_handler(request: Request, exc: FactCheckError):
        if exc.http_status >= 500:
            logger.exception("Fact-check failure on %s", request.url.path)
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        safe_errors = [
            {
                "location": ".".join(str(p) for p in err.get("loc", ())),
                "message": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
            for err in exc.errors()[:10]
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "message": "The request body could not be parsed. Send JSON of the "
                "form {\"text\": \"...\"} with Content-Type: application/json.",
                "details": {"errors": safe_errors},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred while processing the request.",
                "details": {"type": type(exc).__name__},
            },
        )

    app.include_router(router)
    return app


app = create_app()
