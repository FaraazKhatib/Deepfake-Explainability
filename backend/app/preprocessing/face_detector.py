"""
Face detector — uses MTCNN from facenet-pytorch to detect and crop faces.
Falls back to a center crop if no face is detected or if facenet is unavailable.
"""
from __future__ import annotations

import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import MTCNN; gracefully fall back if unavailable
try:
    from facenet_pytorch import MTCNN
    import torch

    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False
    logger.warning(
        "facenet-pytorch not available. Face detection disabled; "
        "will use full frames instead."
    )

# Singleton MTCNN instance (loaded once)
_mtcnn: "MTCNN | None" = None


def _get_mtcnn() -> "MTCNN | None":
    global _mtcnn
    if not _MTCNN_AVAILABLE:
        return None
    if _mtcnn is None:
        try:
            device = "cpu"  # MTCNN is lightweight; CPU is fine
            _mtcnn = MTCNN(
                image_size=224,
                margin=20,
                keep_all=False,       # Only the most prominent face
                post_process=False,   # Return as PIL Image
                device=device,
                select_largest=True,  # Pick the largest face
            )
            logger.info("MTCNN face detector initialized on %s.", device)
        except Exception as exc:
            logger.error("Failed to initialize MTCNN: %s", exc)
            _mtcnn = None
    return _mtcnn


def detect_and_crop_face(
    image: Image.Image,
    fallback_size: int = 224,
) -> tuple[Image.Image, bool]:
    """
    Detect the largest face in `image` and return a cropped/aligned PIL Image.

    Args:
        image: Input PIL Image (RGB).
        fallback_size: Size of the center crop if no face is found.

    Returns:
        (cropped_image, face_found) where:
        - cropped_image is a PIL Image of shape (fallback_size, fallback_size)
        - face_found is True if a face was detected, False if we used the fallback
    """
    mtcnn = _get_mtcnn()

    if mtcnn is not None:
        try:
            # detect() returns (boxes, probs)
            boxes, probs = mtcnn.detect(image)
            if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.9:
                # Use MTCNN's built-in aligned crop
                face = mtcnn(image)
                if face is not None:
                    import torch
                    import torchvision.transforms.functional as F
                    # face is a Tensor [C, H, W] in range [0, 255]
                    face_uint8 = face.permute(1, 2, 0).byte().numpy()
                    from PIL import Image as PILImage
                    cropped = PILImage.fromarray(face_uint8, mode="RGB")
                    return cropped, True
        except Exception as exc:
            logger.debug("MTCNN failed on frame: %s", exc)

    # The OpenCV cascade ships with the existing dependency, unlike optional MTCNN.
    from app.services.image_audio.faces import FaceDetector
    global _fallback_detector
    if _fallback_detector is None:
        _fallback_detector = FaceDetector()
        _fallback_detector.load()
    detection = _fallback_detector.detect(image)
    if detection.faces:
        from app.services.image_audio.preprocessing import crop_with_margin
        return crop_with_margin(image, detection.faces[0].box, 0.15), True
    # Preserve all pixels; the checkpoint's processor owns resizing.
    return image, False


_fallback_detector = None


def detect_and_crop_faces(image: Image.Image, max_faces: int = 8) -> list[Image.Image]:
    """Check each detected face, so a smaller swapped face is not ignored."""
    from app.services.image_audio.preprocessing import crop_with_margin
    boxes = []
    mtcnn = _get_mtcnn()
    if mtcnn is not None:
        try:
            detected, probabilities = mtcnn.detect(image)
            if detected is not None:
                boxes = [tuple(int(x) for x in box) for box, p in zip(detected, probabilities) if p is not None and p > 0.9]
        except Exception as exc:
            logger.debug("MTCNN failed: %s", exc)
    if not boxes:
        from app.services.image_audio.faces import FaceDetector
        global _fallback_detector
        if _fallback_detector is None:
            _fallback_detector = FaceDetector()
            _fallback_detector.load()
        boxes = [f.box for f in _fallback_detector.detect(image).faces]
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return [crop_with_margin(image, box, 0.15) for box in boxes[:max_faces]]


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    """Return a square center crop of `image` resized to `size`x`size`."""
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = image.crop((left, top, left + min_dim, top + min_dim))
    return cropped.resize((size, size), Image.LANCZOS)
