from __future__ import annotations

import asyncio
import base64
import logging
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.services.image_audio import __version__
from app.services.image_audio.config import settings
from app.services.image_audio.engine import Engine
from app.services.image_audio.preprocessing import ImageRejected
from app.services.image_audio.audio import AudioRejected, sniff_audio_mime
from app.services.image_audio.schemas import AnalyzeResponse, Base64Request, HealthResponse

log = logging.getLogger("veritrust")

router = APIRouter(prefix="/api/v1/image_audio")

# Create a module-level engine
engine = Engine(settings)
inference_slots = asyncio.Semaphore(settings.max_concurrency)

UPLOAD_CEILING = max(settings.max_upload_bytes, settings.max_audio_bytes)

def _too_large(size: int | None) -> bool:
    return size is not None and size > UPLOAD_CEILING

async def run_analysis(data: bytes, want_heatmap: bool) -> dict:
    async with inference_slots:
        try:
            analysis = await run_in_threadpool(engine.analyze, data, want_heatmap)
        except (ImageRejected, AudioRejected) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analysis.as_dict()

def setup_image_audio_lifespan():
    log.info("Loading image/audio detectors on device=%s", engine.registry.device)
    engine.load()
    status = engine.status()
    log.info(
        "Image/Audio Ensemble ready: %s model(s), %s failure(s), face backend=%s",
        status["ensemble_size"],
        len(status["failures"]),
        status["face_backend"],
    )
    for failure in status["failures"]:
        log.warning("Detector %s unavailable: %s", failure["key"], failure["error"])

@router.get("/health", response_model=HealthResponse)
async def health():
    status = engine.status()
    return {
        "status": "ok" if status["ensemble_size"] else "degraded",
        "version": __version__,
        "models_loaded": status["ensemble_size"],
        "device": status["device"],
    }

@router.get("/models")
async def models():
    return engine.status()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), heatmap: bool = True, modality: Literal["image", "audio"] | None = None):
    if _too_large(getattr(file, "size", None)):
        raise HTTPException(status_code=413, detail="File too large.")
    data = await file.read(UPLOAD_CEILING + 1)
    if _too_large(len(data)):
        raise HTTPException(status_code=413, detail="File too large.")
    if modality is not None:
        actual = "audio" if sniff_audio_mime(data) is not None else "image"
        if actual != modality:
            raise HTTPException(status_code=400, detail=f"Please upload an {modality} file on this page.")
    return await run_analysis(data, heatmap)

@router.post("/analyze-base64", response_model=AnalyzeResponse)
async def analyze_base64(payload: Base64Request):
    raw = payload.image
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    if len(raw) > 4 * ((UPLOAD_CEILING + 2) // 3):
        raise HTTPException(status_code=413, detail="File too large.")
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Malformed base64 payload.") from exc
    if _too_large(len(data)):
        raise HTTPException(status_code=413, detail="File too large.")
    return await run_analysis(data, payload.heatmap)
