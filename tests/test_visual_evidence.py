import io
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from app.services.image_audio.detectors.base import DetectorResult
from app.services.image_audio.engine import Engine
from app.services.image_audio.faces import Detection, Face


class PortraitEvidenceTests(TestCase):
    def analyze(self, scene_score, portrait_score):
        engine = Engine()
        engine.registry.synthetic = [SimpleNamespace(
            key="detectra_openfake", spec=SimpleNamespace(weight=1.0),
            predict=lambda _: DetectorResult("detectra_openfake", "synthetic", scene_score),
        )]
        image = Image.new("RGB", (80, 80))
        image.paste("white", (0, 0, 40, 80))
        data = io.BytesIO()
        image.save(data, format="PNG")
        detection = Detection([Face((10, 10, 60, 60))], True, "test", "")
        service = SimpleNamespace(is_loaded=True, model_name="dima806/deepfake_vs_real_image_detection",
                                  predict_frame=lambda _: portrait_score)
        with (
            patch.object(engine.face_detector, "detect", return_value=detection),
            patch("app.services.video.model_service.get_model_service", return_value=service),
        ):
            return engine.analyze(data.getvalue())

    def test_strong_face_score_recovers_local_manipulation(self):
        result = self.analyze(0.01, 0.995)
        self.assertEqual(result.verdict, "likely_ai_generated")
        self.assertEqual(result.overridden_by, "shared_portrait_detector")

    def test_borderline_face_score_does_not_flag_photo(self):
        self.assertEqual(self.analyze(0.01, 0.98999).verdict, "likely_authentic")

    def test_image_verdict_cutoff(self):
        for score, expected in [(0.299, "likely_authentic"), (0.30, "uncertain"),
                                (0.301, "uncertain"), (0.90, "likely_ai_generated")]:
            with self.subTest(score=score):
                self.assertEqual(self.analyze(score, 0.01).verdict, expected)

    def test_low_face_score_preserves_scene_detection(self):
        self.assertEqual(self.analyze(0.93, 0.01).score_ai, 0.93)
