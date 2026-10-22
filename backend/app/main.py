from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Video imports
from app.api.video_routes import router as video_router, set_dirs as set_video_dirs
from app.config import settings as video_settings
from app.services.video.model_service import init_model_service as init_video_model

# Image/Audio imports
from app.api.image_audio_routes import router as image_audio_router, setup_image_audio_lifespan

# Text imports
from app.services.text.api.routes import router as text_router
from app.services.text.pipeline import FactCheckPipeline
from app.services.text.exceptions import FactCheckError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Unified Deepfake Platform starting up ===")

    # 1. Video Startup
    base = Path(__file__).parent.parent
    upload_dir = video_settings.resolve_upload_dir(base)
    results_dir = video_settings.resolve_results_dir(base)
    set_video_dirs(upload_dir, results_dir)
    if not video_settings.demo_mode:
        init_video_model(video_settings.model_name, video_settings.model_device)
    else:
        from app.services.video.model_service import ModelService
        import app.services.video.model_service as ms_module
        svc = ModelService(model_name="DEMO", device_pref="cpu")
        svc._model_loaded = True
        svc._model_type = "demo"
        ms_module._service = svc

    # 2. Image/Audio Startup
    setup_image_audio_lifespan()

    # 3. Text Startup
    app.state.startup_error = None
    app.state.pipeline = None
    
    def load_text_pipeline():
        try:
            text_pipeline = FactCheckPipeline()
            text_pipeline.load()
            app.state.pipeline = text_pipeline
        except Exception as exc:
            app.state.startup_error = f"{type(exc).__name__}: {exc}"
            logger.error("Text model loading failed: %s", exc)

    import threading
    threading.Thread(target=load_text_pipeline, daemon=True).start()

    logger.info("=== Unified Deepfake Platform ready ===")
    yield

    app.state.pipeline = None
    logger.info("=== Unified Deepfake Platform shutting down ===")

app = FastAPI(
    title="Unified Deepfake Detection Platform",
    description="Analyzes text, image, audio, and video.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=video_settings.allowed_origins_list,
    allow_origin_regex=r"^chrome-extension://.*$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(FactCheckError)
async def fact_check_error_handler(request: Request, exc: FactCheckError):
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

app.include_router(video_router, prefix="/api/v1/video", tags=["Video"])
app.include_router(image_audio_router, tags=["Image/Audio"])
app.include_router(text_router, prefix="/api/v1/text", tags=["Text"])

@app.get("/")
async def root():
    return {
        "name": "Unified Deepfake Detection API",
        "version": "1.0.0"
    }
