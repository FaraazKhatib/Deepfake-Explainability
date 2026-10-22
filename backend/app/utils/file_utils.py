"""
File utilities for safe temporary file handling.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "video/mpeg",
}


def make_temp_path(upload_dir: Path, suffix: str = ".mp4") -> Path:
    """
    Generate a secure random temporary file path inside upload_dir.
    Uses UUID to avoid filename collisions and prevent path traversal.
    """
    filename = f"{uuid.uuid4().hex}{suffix}"
    return upload_dir / filename


def safe_delete(path: Path | str) -> None:
    """Delete a file safely, ignoring errors if the file does not exist."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def validate_video_extension(filename: str) -> str:
    """
    Validate the video file extension.

    Returns:
        The lowercase extension (e.g. ".mp4").

    Raises:
        ValueError: If the extension is not in ALLOWED_EXTENSIONS.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def validate_video_mime(content_type: str | None) -> None:
    """
    Validate that the MIME type is a recognized video type.

    Raises:
        ValueError: If content_type is not a recognized video MIME.
    """
    if content_type is None:
        return  # Cannot validate — allow (extension check is primary)
    # Normalize content type (strip parameters like "; charset=utf-8")
    base_type = content_type.split(";")[0].strip().lower()
    if base_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported content type: '{base_type}'. "
            f"Expected a video file."
        )
