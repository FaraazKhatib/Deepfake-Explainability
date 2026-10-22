"""Regression tests exercise real decoders/routes; model stubs isolate scoring bugs."""
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.mark.parametrize('claim,premise,expected', [
    ('The Moon is a natural satellite of Earth.', 'Venus ate its moon', False),
    ('The Moon is a natural satellite of Earth.', "Japan's mission samples Mars's moon Phobos", False),
    ('The Moon is a natural satellite of Earth.', 'The Moon orbits Earth.', True),
    ('Paris is the capital of Germany.', 'Paris is the capital of France.', True),
    ('Acme doubled revenue.', 'Acme doubled revenue according to its annual report.', True),
    ('Acme doubled revenue.', 'Acme launches a new product.', False),
])
def test_relevance_requires_more_than_generic_single_word(claim, premise, expected):
    from app.services.text.retrieval.relevance import is_relevant
    assert is_relevant(claim, premise)[0] is expected

from app.services.image_audio.audio import decode_audio, make_windows, AudioRejected
from app.services.image_audio.config import Settings, FusionConfig, Thresholds
from app.services.image_audio.detectors.base import LoadError, DetectorResult
from app.services.image_audio.detectors.hf_audio import resolve_fake_indices
from app.services.image_audio.engine import Engine
from app.services.image_audio.fusion import Signal, fuse
from app.services.image_audio.preprocessing import decode_image, ImageRejected


def image_bytes(mode="RGB", fmt="PNG", size=(80, 60)):
    out = io.BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0,256,(size[1],size[0],3),dtype=np.uint8)).convert(mode).save(out, format=fmt)
    return out.getvalue()


def wav_bytes(seconds=1, rate=16000, channels=1, silent=False):
    values = np.zeros(int(seconds*rate)) if silent else np.sin(np.arange(int(seconds*rate))*2*np.pi*220/rate)*0.2
    pcm = np.repeat((values*32767).astype('<i2')[:, None], channels, axis=1)
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(channels); f.setsampwidth(2); f.setframerate(rate); f.writeframes(pcm.tobytes())
    return out.getvalue()


@pytest.mark.parametrize('mode,fmt', [('RGB','JPEG'),('RGBA','PNG'),('L','PNG'),('RGB','WEBP'),('RGB','BMP'),('RGB','TIFF')])
def test_image_formats(mode, fmt):
    result = decode_image(image_bytes(mode, fmt), Settings())
    assert result.image.mode == 'RGB' and result.width == 80


@pytest.mark.parametrize('data', [b'', b'not an image', b'\x89PNG\r\n\x1a\ntruncated'])
def test_bad_images(data):
    with pytest.raises(ImageRejected):
        decode_image(data, Settings())


def test_image_limits():
    with pytest.raises(ImageRejected):
        decode_image(image_bytes(), Settings(max_pixels=100))
    with pytest.raises(ImageRejected):
        decode_image(image_bytes(), Settings(max_upload_bytes=10))
    assert decode_image(image_bytes(size=(800,400)), Settings(max_edge=200)).image.size == (200,100)


def test_blank_image_rejected():
    data = io.BytesIO(); Image.new('RGB',(100,100),'gray').save(data,format='PNG')
    with pytest.raises(ImageRejected): decode_image(data.getvalue(),Settings())


def test_face_backend_available():
    from app.services.image_audio.faces import FaceDetector
    detector = FaceDetector(Settings())
    detector.load()
    assert detector.backend == 'opencv-haar', detector.note


def test_audio_duration_bounded():
    with pytest.raises(AudioRejected): decode_audio(wav_bytes(seconds=2),Settings(max_audio_seconds=1))


def test_strong_disagreement_abstains():
    result = fuse([Signal('a',0.999,1,'audio'),Signal('b',0.001,1,'audio'),Signal('c',0.999,1,'audio')],Thresholds(),FusionConfig())
    assert result.verdict == 'uncertain' and result.confidence == 0


@pytest.mark.parametrize('scores,expected', [([.70,.99],'likely_ai_generated'),([.30,.01],'likely_authentic')])
def test_score_magnitude_difference_does_not_erase_agreement(scores,expected):
    result = fuse([Signal(str(i),s,1,'audio') for i,s in enumerate(scores)],Thresholds(),FusionConfig())
    assert result.spread_exceeds_limit
    assert result.verdict == expected


def test_uncertain_member_is_not_treated_as_agreement():
    result = fuse([Signal('a',.99,1,'audio'),Signal('b',.45,1,'audio')],Thresholds(),FusionConfig())
    assert result.verdict == 'uncertain'


@pytest.mark.parametrize('logit,expected', [(-4.,0.017986),(4.,0.982014)])
def test_single_logit_image_uses_sigmoid(logit,expected):
    import torch
    from app.services.image_audio.config import ModelSpec
    from app.services.image_audio.detectors.hf_image import HFImageClassifier
    detector = HFImageClassifier(ModelSpec('test','test','synthetic',single_logit_fake=True),'cpu')
    detector._processor = lambda **kwargs: {'pixel_values':torch.zeros(1,3,32,32)}
    detector._model = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[logit]]))
    prediction = detector.predict(Image.new('RGB',(32,32)))
    assert prediction.usable and prediction.p_fake == pytest.approx(expected,abs=1e-6)


def test_old_community_forensics_config_refused():
    from app.services.image_audio.detectors.hf_image import HFImageClassifier
    with pytest.raises(LoadError):
        HFImageClassifier._validate_community_forensics(SimpleNamespace(hidden_size=384,num_attention_heads=12,intermediate_size=3072),None)


@pytest.mark.parametrize('rate,channels', [(16000,1),(44100,2),(48000,1),(8000,1)])
def test_audio_resampling(rate, channels):
    result = decode_audio(wav_bytes(rate=rate, channels=channels), Settings())
    assert result.sample_rate == 16000
    assert abs(len(result.samples)-16000) < 2
    assert np.max(np.abs(result.samples)) < 0.3


def test_silence_abstains():
    engine = Engine(Settings(enable_faces=False))
    result = engine.analyze(wav_bytes(silent=True))
    assert result.verdict == 'uncertain' and result.confidence == 0
    assert result.image_info['windows_scored'] == 0


def test_audio_tail_covered():
    cfg = Settings(audio_max_windows=3)
    decoded = decode_audio(wav_bytes(seconds=17), cfg)
    windows, _ = make_windows(decoded, cfg)
    assert len(windows) == 3 and windows[0].start == 0 and windows[-1].end == 17


def test_opposite_phase_stereo_does_not_erase_speech():
    import soundfile as sf
    mono = np.sin(np.arange(16000)*2*np.pi*220/16000).astype(np.float32)*.2
    data = io.BytesIO()
    sf.write(data,np.stack([mono,-mono],axis=1),16000,format='WAV',subtype='FLOAT')
    decoded = decode_audio(data.getvalue(),Settings())
    assert decoded.phase_cancellation_avoided
    assert np.max(np.abs(decoded.samples)) > .19
    assert len(make_windows(decoded,Settings())[0]) == 1
    from app.services.image_audio.schemas import AnalyzeResponse
    response = AnalyzeResponse(**Engine().analyze(data.getvalue()).as_dict())
    assert response.image.phase_cancellation_avoided is True


@pytest.mark.parametrize('data', [b'', b'RIFFxxxxWAVEbroken', wav_bytes(seconds=0.01)])
def test_bad_audio(data):
    with pytest.raises(AudioRejected):
        decode_audio(data, Settings())


@pytest.mark.parametrize('labels,expected', [({0:'Fake',1:'Real'},[0]),({0:'Real',1:'Fake'},[1]),({0:'AIVoice',1:'HumanVoice'},[0]),({0:'Bonafide',1:'Spoof'},[1]),({0:'ai',1:'hum'},[0])])
def test_label_orders(labels, expected):
    assert resolve_fake_indices(labels)[0] == expected


@pytest.mark.parametrize('labels', [{0:'LABEL_0',1:'LABEL_1'},{0:'real',1:'fake',2:'unknown'},{0:'real_fake',1:'real'}])
def test_ambiguous_labels_refused(labels):
    with pytest.raises(LoadError): resolve_fake_indices(labels)


@pytest.mark.parametrize('score', [float('nan'),float('inf'),-0.1,1.1])
def test_invalid_detector_scores(score):
    assert not DetectorResult('test','audio',score).usable


def test_audio_weight_is_honored():
    result = fuse([Signal('audio',0.9,1.0,'audio')],Thresholds(),FusionConfig(audio_weight=0))
    assert result.verdict == 'uncertain'


@pytest.mark.parametrize('score',[.0001,.008,.99])
def test_single_model_score_is_not_artificially_clamped(score):
    result = fuse([Signal('single',score,1,'synthetic')],Thresholds(),FusionConfig())
    assert result.score == pytest.approx(score)
    assert not result.signals[0]['clamped']


def test_detectra_preprocessing_shape_and_normalization():
    from app.services.image_audio.detectors.onnx_image import prepare_image
    values=prepare_image(Image.new('RGB',(800,400),(255,255,255)))
    assert values.shape == (1,3,384,384) and values.dtype == np.float32
    np.testing.assert_allclose(values[0,:,0,0],(np.ones(3)-[.485,.456,.406])/[.229,.224,.225],rtol=1e-5)


@pytest.mark.parametrize('value,usable',[(0.,True),(float('nan'),False)])
def test_detectra_single_logit_and_invalid_output(value,usable):
    from app.services.image_audio.detectors.onnx_image import ONNXImageClassifier
    from app.services.image_audio.config import ModelSpec
    detector=ONNXImageClassifier(ModelSpec('test','test','synthetic'),'cpu')
    detector._model=SimpleNamespace(run=lambda *a:[np.array([[value]],np.float32)])
    result=detector.predict(Image.new('RGB',(400,400)))
    assert result.usable == usable
    if usable: assert result.p_fake == .5


def test_unknown_media_is_client_error():
    with pytest.raises(ImageRejected): Engine().analyze(b'garbage')


def test_no_image_models_abstains(monkeypatch):
    from app.services.video import model_service
    monkeypatch.setattr(model_service, '_service', None)
    result = Engine().analyze(image_bytes())
    assert result.verdict == 'uncertain' and result.confidence == 0


@pytest.mark.parametrize('scores,faces,expected', [([0.1,0.1],True,'REAL'),([0.9,0.9],True,'FAKE'),([0.49,0.51],True,'UNCERTAIN'),([0.05,0.95],True,'UNCERTAIN'),([0.1,0.1],False,'UNCERTAIN')])
def test_video_aggregation(monkeypatch,scores,faces,expected):
    from app.services.video import analyzer
    from app.preprocessing.frame_extractor import FrameRecord
    values = iter(scores)
    monkeypatch.setattr(analyzer,'get_model_service',lambda: SimpleNamespace(model_name='stub',predict_frame=lambda _: next(values)))
    monkeypatch.setattr(analyzer,'get_video_metadata',lambda _: {'duration':2})
    monkeypatch.setattr(analyzer,'extract_frames',lambda *a,**k: [FrameRecord(i,float(i),Image.new('RGB',(64,64))) for i in range(len(scores))])
    monkeypatch.setattr(analyzer,'detect_and_crop_faces',lambda image: [image] if faces else [])
    result = analyzer.analyze_video('unused')
    assert result.classification == expected
    assert result.fake_probability == pytest.approx(sum(scores)/len(scores))


def test_video_total_failure_is_error(monkeypatch):
    from app.services.video import analyzer
    from app.preprocessing.frame_extractor import FrameRecord
    monkeypatch.setattr(analyzer,'get_model_service',lambda: SimpleNamespace(model_name='stub',predict_frame=lambda _: float('nan')))
    monkeypatch.setattr(analyzer,'get_video_metadata',lambda _: {})
    monkeypatch.setattr(analyzer,'extract_frames',lambda *a,**k: [FrameRecord(0,0.,Image.new('RGB',(64,64)))])
    with pytest.raises(RuntimeError): analyzer.analyze_video('unused')


def test_video_checks_smaller_face(monkeypatch):
    from app.services.video import analyzer
    from app.preprocessing.frame_extractor import FrameRecord
    values = iter([0.01,0.95])
    image = Image.new('RGB',(64,64))
    monkeypatch.setattr(analyzer,'get_model_service',lambda: SimpleNamespace(model_name='stub',predict_frame=lambda _: next(values)))
    monkeypatch.setattr(analyzer,'get_video_metadata',lambda _: {})
    monkeypatch.setattr(analyzer,'extract_frames',lambda *a,**k: [FrameRecord(0,0.,image)])
    monkeypatch.setattr(analyzer,'detect_and_crop_faces',lambda _: [image,image])
    result = analyzer.analyze_video('unused')
    assert result.fake_probability == 0.95 and result.faces_detected == 2


def test_real_video_decode(tmp_path):
    import cv2
    from app.preprocessing.frame_extractor import extract_frames, get_video_metadata
    path = str(tmp_path/'fixture.avi')
    writer = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
    assert writer.isOpened()
    for i in range(20): writer.write(np.full((64,64,3),i*10,dtype=np.uint8))
    writer.release()
    assert get_video_metadata(path)['duration'] == 2
    assert len(extract_frames(path,4)) == 4


def test_text_fact_check_not_a_refutation():
    from app.services.text.stance.debunk import apply_debunk_override
    claim = 'Paris is the capital of France'
    result = apply_debunk_override('supports',0.02,claim,'Fact check: Paris is the capital of France. Misinformation says otherwise.',None)
    assert result.stance == 'supports' and not result.fired


def test_text_generic_nli_labels_refused():
    from app.services.text.stance.nli import StanceDetector
    from app.services.text.exceptions import ModelLoadError
    detector = StanceDetector(device='cpu')
    detector.model = SimpleNamespace(config=SimpleNamespace(id2label={0:'LABEL_0',1:'LABEL_1',2:'LABEL_2'}))
    with pytest.raises(ModelLoadError): detector._resolve_label_indices()


def test_text_no_evidence_unverified():
    from app.services.text.scoring.verdict import decide
    from app.services.text.credibility.sources import CredibilityIndex
    assert decide('Paris is the capital of France',[],CredibilityIndex()).verdict == 'Unverified'


@pytest.mark.parametrize('text',['','hi','Why is the sky blue?','In my opinion this is the best movie ever.'])
def test_text_uncheckable(text):
    from app.services.text.claims.extractor import extract_claims
    from app.services.text.exceptions import FactCheckError
    with pytest.raises(FactCheckError): extract_claims(text)


def test_media_routes(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api import image_audio_routes
    app = FastAPI(); app.include_router(image_audio_routes.router)
    client = TestClient(app)
    assert client.post('/api/v1/image_audio/analyze?modality=image',files={'file':('speech.wav',wav_bytes())}).status_code == 400
    assert client.post('/api/v1/image_audio/analyze?modality=audio',files={'file':('image.png',image_bytes())}).status_code == 400
    assert client.post('/api/v1/image_audio/analyze',files={'file':('fake.jpg',b'garbage','image/jpeg')}).status_code == 400
    assert client.post('/api/v1/image_audio/analyze-base64',json={'image':'!!!!'}).status_code == 400
    monkeypatch.setattr(image_audio_routes,'UPLOAD_CEILING',32)
    assert client.post('/api/v1/image_audio/analyze',files={'file':('big.png',b'x'*33)}).status_code == 413
    assert client.post('/api/v1/image_audio/analyze-base64',json={'image':'A'*100}).status_code == 413
