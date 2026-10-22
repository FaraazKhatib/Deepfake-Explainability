"""
Frame extractor — samples N frames uniformly from a video file using OpenCV.
Returns a list of (frame_index, timestamp_seconds, PIL.Image) tuples.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FrameRecord(NamedTuple):
    index: int          # Original frame index in the video
    timestamp: float    # Seconds from video start
    image: Image.Image  # PIL Image (RGB)


def extract_frames(
    video_path: str | Path,
    num_frames: int = 16,
) -> list[FrameRecord]:
    """
    Uniformly sample `num_frames` frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        List of FrameRecord(index, timestamp, PIL Image RGB).

    Raises:
        ValueError: If the video cannot be opened or has no readable frames.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Safety: cannot sample more frames than exist
    num_frames = min(num_frames, total_frames)
    fps = fps if fps > 0 else 25.0

    # Uniformly spaced indices across the video
    if num_frames == 1:
        sample_indices = [total_frames // 2]
    else:
        step = total_frames / num_frames
        sample_indices = [int(i * step + step / 2) for i in range(num_frames)]
        # Ensure indices are within bounds
        sample_indices = [min(idx, total_frames - 1) for idx in sample_indices]

    records: list[FrameRecord] = []
    for target_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %d, skipping.", target_idx)
            continue

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        timestamp = target_idx / fps

        records.append(FrameRecord(index=target_idx, timestamp=timestamp, image=pil_image))

    cap.release()

    if not records:
        raise ValueError("No frames could be extracted from the video.")

    logger.info("Extracted %d frames from %s (total=%d, fps=%.1f)",
                len(records), video_path, total_frames, fps)
    return records


def get_video_metadata(video_path: str | Path) -> dict:
    """
    Return basic video metadata without extracting frames.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }
