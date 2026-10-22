"""
DeepGuard AI — Analysis Orchestrator

Ties together:
  1. Frame extraction
  2. Face detection / cropping
  3. Model inference
  4. Aggregation + suspicious frame detection
"""
from __future__ import annotations

import base64
import io
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from app.preprocessing.face_detector import detect_and_crop_faces
from app.preprocessing.frame_extractor import FrameRecord, extract_frames, get_video_metadata
from app.services.video.model_service import get_model_service

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    fake_probability: float
    face_detected: bool
    thumbnail_b64: str = ""   # Base64-encoded JPEG thumbnail (optional)


@dataclass
class AnalysisResult:
    classification: str          # "FAKE" | "REAL"
    fake_probability: float
    real_probability: float
    confidence: float
    frames_analyzed: int
    processing_time: float
    suspicious_frames: list[FrameResult] = field(default_factory=list)
    all_frames: list[FrameResult] = field(default_factory=list)
    faces_detected: int = 0
    video_duration: float = 0.0
    model_name: str = ""
    warning: str = ""
    frames_failed: int = 0
    calibrated: bool = False
    confidence_meaning: str = "Distance from an uncertain score; not measured accuracy."


def _image_to_b64_thumbnail(image: Image.Image, max_size: int = 200) -> str:
    """Encode a PIL image as a small base64 JPEG string."""
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_video(
    video_path: str,
    num_frames: int = 16,
    fake_threshold: float = 0.5,
    suspicious_threshold: float = 0.6,
    progress_callback: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """
    Full end-to-end video deepfake analysis pipeline.

    Args:
        video_path: Absolute path to the video file.
        num_frames: Number of frames to sample.
        fake_threshold: Classification threshold (>= means FAKE).
        suspicious_threshold: Per-frame threshold for "suspicious" flag.
        progress_callback: Optional callback(stage_name) for progress updates.

    Returns:
        AnalysisResult with all predictions and metadata.
    """
    t_start = time.time()
    svc = get_model_service()

    def _emit(stage: str) -> None:
        if progress_callback:
            progress_callback(stage)
        logger.info("[pipeline] %s", stage)

    # ── 1. Video metadata ────────────────────────────────────────────────────
    metadata = get_video_metadata(video_path)
    video_duration = metadata.get("duration", 0.0)
    _emit("extracting_frames")

    # ── 2. Frame extraction ──────────────────────────────────────────────────
    frames: list[FrameRecord] = extract_frames(video_path, num_frames=num_frames)
    logger.info("Extracted %d frames (requested %d)", len(frames), num_frames)
    _emit("detecting_faces")

    # ── 3. Face detection + inference ────────────────────────────────────────
    frame_results: list[FrameResult] = []
    faces_detected = 0
    frames_failed = 0

    for rec in frames:
        try:
            crops = detect_and_crop_faces(rec.image)
            face_found = bool(crops)
            if face_found:
                faces_detected += len(crops)
        except Exception as exc:
            logger.debug("Face detection failed for frame %d: %s", rec.index, exc)
            crops = []
            face_found = False

        try:
            scores = [svc.predict_frame(crop) for crop in (crops or [rec.image])]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
                raise ValueError("Invalid face score")
            # One manipulated face can matter; this maximum is a review score,
            # not a calibrated probability. Frames are averaged equally below.
            fake_prob = max(scores)
            if not math.isfinite(fake_prob) or not 0 <= fake_prob <= 1:
                raise ValueError("Invalid frame score")
        except Exception as exc:
            logger.error("Inference failed for frame %d: %s", rec.index, exc)
            frames_failed += 1
            continue

        thumbnail = _image_to_b64_thumbnail(rec.image)

        frame_results.append(FrameResult(
            frame_index=rec.index,
            timestamp=round(rec.timestamp, 2),
            fake_probability=round(fake_prob, 4),
            face_detected=face_found,
            thumbnail_b64=thumbnail,
        ))

    _emit("running_ai")

    # ── 4. Aggregation ───────────────────────────────────────────────────────
    if not frame_results:
        raise RuntimeError("No frames could be scored. The detector is unavailable.")

    fake_probs = [fr.fake_probability for fr in frame_results]
    avg_fake_prob = sum(fake_probs) / len(fake_probs)

    # Equal weighting avoids a built-in bias toward fake frames. Localized suspicious
    # frames remain visible separately and mixed evidence abstains below.
    final_fake_prob = avg_fake_prob
    final_fake_prob = round(max(0.0, min(1.0, final_fake_prob)), 4)
    final_real_prob = round(1.0 - final_fake_prob, 4)

    lower, upper = max(0.0, fake_threshold - 0.15), min(1.0, fake_threshold + 0.15)
    classification = "FAKE" if final_fake_prob >= upper else "REAL" if final_fake_prob <= lower else "UNCERTAIN"
    mixed = min(fake_probs) <= lower and max(fake_probs) >= upper
    if mixed or frames_failed or faces_detected == 0:
        classification = "UNCERTAIN"
    confidence = 0.0 if classification == "UNCERTAIN" else abs(2 * final_fake_prob - 1)

    # ── 5. Suspicious frames ──────────────────────────────────────────────────
    suspicious = [
        fr for fr in frame_results
        if fr.fake_probability >= suspicious_threshold
    ]
    suspicious.sort(key=lambda fr: fr.fake_probability, reverse=True)

    _emit("generating_report")
    processing_time = round(time.time() - t_start, 2)

    warning = ""
    if mixed:
        warning += "Different parts of the video gave conflicting results. Review the flagged frames. "
    if frames_failed:
        warning += f"Could not score {frames_failed} sampled frame(s). The result is incomplete. "
    if faces_detected == 0 and len(frames) > 0:
        warning += (
            "No faces detected in sampled frames. "
            "The model was applied to full frames. Results may be less accurate."
        )

    return AnalysisResult(
        classification=classification,
        fake_probability=final_fake_prob,
        real_probability=final_real_prob,
        confidence=round(confidence, 4),
        frames_analyzed=len(frame_results),
        processing_time=processing_time,
        suspicious_frames=suspicious,
        all_frames=frame_results,
        faces_detected=faces_detected,
        video_duration=round(video_duration, 2),
        model_name=svc.model_name,
        warning=warning,
        frames_failed=frames_failed,
    )
