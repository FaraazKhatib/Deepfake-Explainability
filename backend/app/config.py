"""
DeepGuard AI — Application Configuration
Loads settings from environment variables / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "dima806/deepfake_vs_real_image_detection"
    model_device: str = "auto"          # "auto" | "cuda" | "mps" | "cpu"

    # ── Inference ──────────────────────────────────────────────────────────────
    num_frames: int = 16
    fake_threshold: float = 0.5
    suspicious_frame_threshold: float = 0.6

    # ── Video constraints ─────────────────────────────────────────────────────
    max_video_size_mb: int = 200
    max_video_duration: int = 120       # seconds

    # ── App ───────────────────────────────────────────────────────────────────
    demo_mode: bool = False
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir: str = "../uploads"
    results_dir: str = "../results"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def max_video_size_bytes(self) -> int:
        return self.max_video_size_mb * 1024 * 1024

    def resolve_upload_dir(self, base: Path) -> Path:
        p = Path(self.upload_dir)
        if not p.is_absolute():
            p = (base / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve_results_dir(self, base: Path) -> Path:
        p = Path(self.results_dir)
        if not p.is_absolute():
            p = (base / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton
settings = Settings()
