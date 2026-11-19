import io
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from app.preprocessing.frame_extractor import FrameRecord
from app.services.video import analyzer
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
        self.assertEqual(self.analyze(0.01, 0.98999).verdict, "uncertain")

    def test_low_face_score_preserves_scene_detection(self):
        self.assertEqual(self.analyze(0.93, 0.01).score_ai, 0.93)


class SceneEvidenceTests(TestCase):
    def analyze(self, scores):
        image = Image.new("RGB", (64, 64))
        frames = [FrameRecord(i, float(i), image) for i in range(len(scores))]
        predictions = iter(scores)
        scene = SimpleNamespace(
            key="detectra_openfake",
            predict=lambda _: DetectorResult("scene", "synthetic", next(predictions)),
        )
        portrait = SimpleNamespace(model_name="portrait", predict_frame=lambda _: 0.05)
        with (
            patch.object(analyzer, "get_model_service", return_value=portrait),
            patch.object(analyzer, "get_video_metadata", return_value={"total_frames": len(frames)}),
            patch.object(analyzer, "extract_frames", return_value=frames),
            patch.object(analyzer, "detect_and_crop_faces", return_value=[]),
        ):
            return analyzer.analyze_video("test", scene_detector=scene)

    def test_sustained_scene_evidence_without_faces(self):
        result = self.analyze([0.05, 0.85, 0.9, 0.95])
        self.assertEqual(result.classification, "FAKE")
        self.assertAlmostEqual(result.fake_probability, 0.875)
        self.assertEqual(len(result.suspicious_frames), 3)

    def test_isolated_or_incomplete_evidence_abstains(self):
        for scores in ([0.99, 0.05, 0.05, 0.05], [0.99, 0.99, 0.99, None], [0.99]):
            with self.subTest(scores=scores):
                self.assertEqual(self.analyze(scores).classification, "UNCERTAIN")
