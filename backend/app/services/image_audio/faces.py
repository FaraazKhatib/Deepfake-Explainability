from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Face:
    box: tuple[int, int, int, int]

    def as_dict(self) -> dict[str, Any]:
        left, top, right, bottom = self.box
        # Haar does not provide a calibrated detection confidence.
        return {"x": left, "y": top, "w": right-left, "h": bottom-top, "score": 0.0}


@dataclass
class Detection:
    faces: list[Face]
    available: bool
    backend: str
    note: str


class FaceDetector:
    def __init__(self, settings: Any | None = None):
        self.settings = settings
        self._cascade = None
        self.backend = "unavailable"
        self.note = "Face detection is unavailable."

    def load(self) -> None:
        if self.settings is not None and not self.settings.enable_faces:
            self.note = "Face detection is disabled."
            return
        try:
            import cv2
            self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self._cascade.empty():
                raise ValueError("OpenCV face cascade is unavailable")
            self.backend = "opencv-haar"
            self.note = "Basic frontal-face detection; side profiles and small faces may be missed."
        except Exception as exc:
            self._cascade = None
            self.note = f"Face detection is unavailable: {exc}"

    def detect(self, image: Any) -> Detection:
        if self._cascade is None:
            return Detection([], False, self.backend, self.note)
        import cv2
        import numpy as np
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        size = getattr(self.settings, "face_min_size", 32)
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(size, size))
        boxes = sorted(boxes, key=lambda b: int(b[2])*int(b[3]), reverse=True)
        faces = [Face((int(x), int(y), int(x+w), int(y+h))) for x,y,w,h in boxes[:getattr(self.settings, "max_faces", 8)]]
        return Detection(faces, True, self.backend, self.note)
