"""
DeepGuard AI — FastAPI Routes

Endpoints:
  GET  /api/health      — Health check + model status
  POST /api/analyze     — Video deepfake analysis
"""
from __future__ import annotations

import logging
import asyncio
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.services.video.analyzer import analyze_video
from app.services.video.model_service import get_model_service
from app.utils.file_utils import (
    make_temp_path,
    safe_delete,
    validate_video_extension,
    validate_video_mime,
)

logger = logging.getLogger(__name__)
router = APIRouter()
inference_slots = asyncio.Semaphore(1)

# Resolved directory paths (set at startup via lifespan in main.py)
_UPLOAD_DIR: Path | None = None
_RESULTS_DIR: Path | None = None


def set_dirs(upload_dir: Path, results_dir: Path) -> None:
    global _UPLOAD_DIR, _RESULTS_DIR
    _UPLOAD_DIR = upload_dir
    _RESULTS_DIR = results_dir


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check() -> JSONResponse:
    """Return API health and model status."""
    try:
        svc = get_model_service()
        model_ready = svc.is_loaded
        model_error = svc.load_error
        model_name = svc.model_name
    except RuntimeError:
        model_ready = False
        model_error = "Model service not initialized."
        model_name = settings.model_name

    return JSONResponse({
        "status": "ok",
        "model_ready": model_ready,
        "model_name": model_name,
        "model_error": model_error,
        "device": settings.model_device,
        "num_frames": settings.num_frames,
        "demo_mode": settings.demo_mode,
    })


# ── Analysis ───────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a video file and analyze it for deepfake manipulation.

    Returns a JSON object with classification, probabilities, suspicious frames, etc.
    """
    if _UPLOAD_DIR is None:
        raise HTTPException(status_code=500, detail="Server not initialized.")

    # ── 1. Validate model ─────────────────────────────────────────────────────
    svc = get_model_service()
    if not svc.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {svc.load_error}",
        )

    # ── 2. Validate file type ─────────────────────────────────────────────────
    original_filename = file.filename or "upload.mp4"
    try:
        ext = validate_video_extension(original_filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    try:
        validate_video_mime(file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    # ── 3. Validate file size ─────────────────────────────────────────────────
    # Read file into memory to check size, then write to disk
    content = await file.read(settings.max_video_size_bytes + 1)
    if len(content) > settings.max_video_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(content) / 1_048_576:.1f} MB. "
                f"Maximum allowed: {settings.max_video_size_mb} MB."
            ),
        )

    # ── 4. Save to temp file ──────────────────────────────────────────────────
    temp_path = make_temp_path(_UPLOAD_DIR, suffix=ext)
    try:
        async with aiofiles.open(str(temp_path), "wb") as f:
            await f.write(content)
        logger.info("Saved upload to %s (%d bytes)", temp_path.name, len(content))
    except Exception as exc:
        safe_delete(temp_path)
        logger.error("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # ── 5. Validate video duration ────────────────────────────────────────────
    from app.preprocessing.frame_extractor import get_video_metadata
    meta = get_video_metadata(str(temp_path))
    if "error" in meta:
        safe_delete(temp_path)
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is not a valid or readable video.",
        )

    duration = meta.get("duration", 0)
    if duration > settings.max_video_duration:
        safe_delete(temp_path)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Video too long: {duration:.0f}s. "
                f"Maximum allowed: {settings.max_video_duration}s."
            ),
        )

    if duration < 0.1:
        safe_delete(temp_path)
        raise HTTPException(
            status_code=422,
            detail="Video is empty or too short to analyze.",
        )

    # ── 6. Demo mode shortcut ─────────────────────────────────────────────────
    if settings.demo_mode:
        safe_delete(temp_path)
        return JSONResponse(_demo_result(original_filename))

    # ── 7. Run analysis ───────────────────────────────────────────────────────
    try:
        async with inference_slots:
            result = await run_in_threadpool(
                analyze_video,
                video_path=str(temp_path),
                num_frames=settings.num_frames,
                fake_threshold=settings.fake_threshold,
                suspicious_threshold=settings.suspicious_frame_threshold,
            )
    except ValueError as exc:
        safe_delete(temp_path)
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        safe_delete(temp_path)
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(exc)[:200]}",
        )
    finally:
        safe_delete(temp_path)

    # ── 8. Serialize response ─────────────────────────────────────────────────
    suspicious_list = [
        {
            "frame": fr.frame_index,
            "timestamp": fr.timestamp,
            "fake_probability": fr.fake_probability,
            "thumbnail": fr.thumbnail_b64,
        }
        for fr in result.suspicious_frames
    ]

    all_frames_list = [
        {
            "frame": fr.frame_index,
            "timestamp": fr.timestamp,
            "fake_probability": fr.fake_probability,
            "face_detected": fr.face_detected,
            "thumbnail": fr.thumbnail_b64,
        }
        for fr in result.all_frames
    ]

    return JSONResponse({
        "classification": result.classification,
        "fake_probability": result.fake_probability,
        "real_probability": result.real_probability,
        "confidence": result.confidence,
        "frames_analyzed": result.frames_analyzed,
        "processing_time": result.processing_time,
        "suspicious_frames": suspicious_list,
        "all_frames": all_frames_list,
        "faces_detected": result.faces_detected,
        "video_duration": result.video_duration,
        "model_name": result.model_name,
        "warning": result.warning,
        "frames_failed": result.frames_failed,
        "calibrated": result.calibrated,
        "confidence_meaning": result.confidence_meaning,
        "disclaimer": (
            "This is an AI-based assessment and should not be treated as conclusive proof. "
            "Results may vary with video quality, compression, and subject matter."
        ),
    })


def _demo_result(filename: str) -> dict:
    """Return a precomputed demo result for UI testing."""
    return {
        "classification": "FAKE",
        "fake_probability": 0.923,
        "real_probability": 0.077,
        "confidence": 0.923,
        "frames_analyzed": 16,
        "processing_time": 3.2,
        "suspicious_frames": [
            {"frame": 45, "timestamp": 1.8, "fake_probability": 0.97, "thumbnail": ""},
            {"frame": 112, "timestamp": 4.5, "fake_probability": 0.96, "thumbnail": ""},
            {"frame": 180, "timestamp": 7.2, "fake_probability": 0.89, "thumbnail": ""},
            {"frame": 225, "timestamp": 9.0, "fake_probability": 0.87, "thumbnail": ""},
        ],
        "all_frames": [],
        "faces_detected": 16,
        "video_duration": 12.5,
        "model_name": "DEMO_MODE",
        "warning": "",
        "disclaimer": (
            "DEMO MODE: These results are precomputed for UI demonstration. "
            "Real analysis is disabled."
        ),
    }
