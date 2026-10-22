"""
DeepGuard AI — Deepfake Detection Model Service

Primary model: dima806/deepfake_vs_real_image_detection (ViT-Base, 343 MB)
  - Labels: {0: 'Real', 1: 'Fake'} — dynamically resolved to avoid inversion bugs

Fast CPU alternative: Xicor9/efficientnet-b0-ffpp-c23 (EfficientNet-B0, 21 MB)
  - Trained on FaceForensics++ C23, ~15-30ms per frame on CPU

Research notes:
  - Face cropping is ESSENTIAL for all these models (trained on face crops)
  - Label indices differ per model — always use id2label to find fake index
  - Wvolf/ViT_Deepfake_Detection has INVERTED labels (0=Fake, 1=Real) — avoid
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Ordered list of models to try (most preferred first)
FALLBACK_MODEL_CHAIN = [
    "dima806/deepfake_vs_real_image_detection",   # ViT-Base, 343MB, most reliable
    "prithivMLmods/Deep-Fake-Detector-v2-Model",  # ViT-Base, 343MB, alternative
    "prithivMLmods/deepfake-detector-model-v1",   # ViT-Base, older version
]


class ModelService:
    """Singleton service that loads the model once and exposes frame-level inference."""

    def __init__(self, model_name: str, device_pref: str = "auto") -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device_pref)
        self._processor: Any = None
        self._model: Any = None
        self._fake_index: int = 1   # Default: label 1 = Fake
        self._model_loaded = False
        self._load_error: str | None = None
        self._model_type: str = "unknown"

    @staticmethod
    def _resolve_device(pref: str) -> str:
        if pref == "auto":
            if torch.cuda.is_available():
                logger.info("CUDA GPU detected.")
                return "cuda"
            if torch.backends.mps.is_available():
                logger.info("Apple MPS detected.")
                return "mps"
            logger.info("No GPU detected, using CPU.")
            return "cpu"
        return pref

    # ─────────────────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the model. Called once at startup."""
        t0 = time.time()

        # Build full chain starting with the user's preferred model
        chain = [self.model_name]
        for m in FALLBACK_MODEL_CHAIN:
            if m != self.model_name:
                chain.append(m)

        for model_id in chain:
            if self._try_load_hf_model(model_id):
                elapsed = time.time() - t0
                logger.info("✓ Model '%s' loaded in %.2fs on %s", model_id, elapsed, self.device)
                return

        # Last resort: fast EfficientNet-B0 from HuggingFace (no deepfake weights warning)
        if self._try_load_efficientnet_ffpp():
            elapsed = time.time() - t0
            logger.info("✓ EfficientNet-B0 (FFPP) loaded in %.2fs", elapsed)
            return

        err = (
            "All model loading attempts failed. "
            "Please check your internet connection and install transformers>=4.46."
        )
        logger.error(err)
        self._load_error = err

    def _try_load_hf_model(self, model_id: str) -> bool:
        """Try to load a HuggingFace image classification model."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            logger.info("Trying HuggingFace model: %s ...", model_id)
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)

            # Determine device index for model placement
            if self.device == "cuda":
                model = model.to("cuda")
            elif self.device == "mps":
                model = model.to("mps")
            # CPU: leave as-is
            model.eval()

            # ── Dynamically resolve 'fake' label index ────────────────────────
            fake_index = self._find_fake_label_index(model)
            logger.info("Label map: %s → fake_index=%d", model.config.id2label, fake_index)

            self._processor = processor
            self._model = model
            self._fake_index = fake_index
            self._model_loaded = True
            self._model_type = "hf_auto"
            self.model_name = model_id
            return True

        except Exception as exc:
            logger.warning("Failed to load '%s': %s", model_id, str(exc)[:200])
            return False

    def _try_load_efficientnet_ffpp(self) -> bool:
        """Load Xicor9/efficientnet-b0-ffpp-c23 as a CPU-optimized fallback."""
        try:
            import torch.nn as nn
            from torchvision import models, transforms

            logger.info("Trying EfficientNet-B0 (FaceForensics++ C23) fallback...")
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

            url = (
                "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23"
                "/resolve/main/efficientnet_b0_ffpp_c23.pth"
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url, map_location=self.device, progress=True, weights_only=True
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            self._model = {"model": model, "transform": transform}
            self._fake_index = 1   # Class 1 = Fake for this model
            self._model_loaded = True
            self._model_type = "efficientnet_ffpp"
            self.model_name = "Xicor9/efficientnet-b0-ffpp-c23"
            return True

        except Exception as exc:
            logger.warning("EfficientNet-B0 fallback also failed: %s", str(exc)[:200])
            return False

    @staticmethod
    def _find_fake_label_index(model: Any) -> int:
        """
        Dynamically resolve which label index corresponds to 'fake'.
        This prevents silent prediction inversions caused by inconsistent
        label orderings across HuggingFace models.
        """
        from app.services.image_audio.detectors.hf_audio import resolve_fake_indices
        indices, _ = resolve_fake_indices({int(k): v for k, v in model.config.id2label.items()})
        if len(indices) != 1 or len(model.config.id2label) != 2:
            raise ValueError("Video classifier must have one fake and one real class.")
        return indices[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict_frame(self, image: Image.Image) -> float:
        """
        Run inference on a single (ideally face-cropped) frame.

        Args:
            image: PIL Image (RGB).

        Returns:
            fake_probability: float in [0, 1].

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError(f"Model not loaded: {self._load_error}")

        if self._model_type == "hf_auto":
            return self._predict_hf(image)
        elif self._model_type == "efficientnet_ffpp":
            return self._predict_efficientnet(image)
        else:
            raise RuntimeError(f"Unknown model type: {self._model_type}")

    def _predict_hf(self, image: Image.Image) -> float:
        """Inference via HuggingFace AutoImageProcessor + AutoModelForImageClassification."""
        inputs = self._processor(images=image, return_tensors="pt")

        # Move to appropriate device
        if self.device in ("cuda", "mps"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        fake_prob = float(probs[0, self._fake_index].item())
        if not math.isfinite(fake_prob):
            raise ValueError("Model returned a non-finite score")
        return max(0.0, min(1.0, fake_prob))

    def _predict_efficientnet(self, image: Image.Image) -> float:
        """Inference via EfficientNet-B0 loaded from HuggingFace."""
        model = self._model["model"]
        transform = self._model["transform"]

        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1)

        fake_prob = float(probs[0, self._fake_index].item())
        if not math.isfinite(fake_prob):
            raise ValueError("Model returned a non-finite score")
        return max(0.0, min(1.0, fake_prob))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Return the global ModelService singleton."""
    if _service is None:
        raise RuntimeError("Model service not initialized. Call init_model_service() first.")
    return _service


def init_model_service(model_name: str, device: str = "auto") -> ModelService:
    """Initialize and load the global model service. Call once at startup."""
    global _service
    _service = ModelService(model_name=model_name, device_pref=device)
    _service.load()
    return _service
