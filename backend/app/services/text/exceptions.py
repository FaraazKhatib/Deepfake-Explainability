"""Typed exceptions mapped onto HTTP status codes by the API layer."""

from __future__ import annotations


class FactCheckError(Exception):
    """Base class for every error raised by this module."""

    code = "fact_check_error"
    http_status = 500

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "details": self.details}


class InputValidationError(FactCheckError):
    """Claim/article text missing, too short or too long -> 400."""

    code = "invalid_input"
    http_status = 400


class NoClaimFoundError(FactCheckError):
    """Input parsed, but nothing check-worthy could be extracted -> 422."""

    code = "no_checkable_claim"
    http_status = 422


class RetrievalError(FactCheckError):
    """Every evidence source failed (network down, all endpoints erroring)."""

    code = "retrieval_failed"
    http_status = 503


class ModelLoadError(FactCheckError):
    """The NLI or style model could not be loaded -> 503."""

    code = "model_unavailable"
    http_status = 503


class InferenceError(FactCheckError):
    """A model loaded but failed while scoring -> 500."""

    code = "inference_failed"
    http_status = 500
