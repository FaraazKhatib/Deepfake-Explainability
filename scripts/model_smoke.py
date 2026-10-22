"""Offline real-weight smoke tests. Diagnostic inputs do NOT measure deepfake accuracy."""
import gc
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('FC_OFFLINE_ONLY', 'true')

import numpy as np
from PIL import Image, ImageFilter
import torch
torch.set_num_threads(2)

from app.services.image_audio.config import BUILTIN_MODELS
from app.services.image_audio.detectors.hf_audio import HFAudioClassifier
from app.services.image_audio.detectors.hf_image import HFImageClassifier

report = {'purpose': 'Real-weight smoke tests on generated diagnostic controls; not an accuracy benchmark.', 'models': []}
out = ROOT / 'checkpoints/model-smoke.json'

def record(entry):
    report['models'].append(entry)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(entry), flush=True)

for key in ('siglip_image', 'siglip_face', 'deepfake_wav2vec2', 'wavlm_itw', 'ast_asvspoof'):
    spec = next(s for s in BUILTIN_MODELS if s.key == key)
    detector = (HFAudioClassifier if spec.kind == 'audio' else HFImageClassifier)(spec,'cpu')
    started = time.perf_counter()
    try:
        detector.load()
        if spec.kind == 'audio':
            rate = 16000
            t = np.arange(rate*4,dtype=np.float32)/rate
            controls = {'tone_220hz': np.sin(2*np.pi*220*t)*0.2, 'noise': np.random.default_rng(7).normal(0,0.04,len(t)).astype(np.float32)}
        else:
            img = Image.fromarray(np.random.default_rng(7).integers(0,256,(256,384,3),dtype=np.uint8))
            controls = {'flat_gray':Image.new('RGB',(256,256),'gray'),'noise':img,'blurred_noise':img.filter(ImageFilter.GaussianBlur(3))}
        scores = {name: {'score':r.p_fake,'error':r.error} for name, sample in controls.items() for r in [detector.predict(sample)]}
        record({'key':key,'repo':detector.repo_used,'labels':detector.label_mapping,'seconds':round(time.perf_counter()-started,2),'controls':scores})
    except Exception as exc:
        record({'key':key,'error':str(exc)})
    del detector
    gc.collect()

from app.services.video.model_service import ModelService
service = ModelService('dima806/deepfake_vs_real_image_detection','cpu')
try:
    if not service._try_load_hf_model(service.model_name): raise RuntimeError('Cached video model unavailable')
    record({'key':'video_baseline','repo':service.model_name,'controls':{name:service.predict_frame(img) for name,img in {'gray':Image.new('RGB',(224,224),'gray'),'black':Image.new('RGB',(224,224),'black')}.items()}})
except Exception as exc: record({'key':'video_baseline','error':str(exc)})
del service
gc.collect()

from app.services.text.stance.nli import StanceDetector
from app.services.text.retrieval.base import Evidence
detector = StanceDetector(device='cpu')
try:
    detector.load()
    cases = [('Paris is the capital of France.','Paris is the capital of France.'),('Paris is the capital of France.','Paris is not the capital of France.'),('Paris is the capital of France.','Saturn has many moons.')]
    results = []
    for claim,premise in cases:
        result = detector.classify(claim,[Evidence(premise,'https://example.org','example.org','fixture')])[0]
        results.append({'claim':claim,'premise':premise,'stance':result.stance,'entailment':result.entailment,'contradiction':result.contradiction})
    record({'key':'text_nli','repo':detector.model_name,'cases':results})
except Exception as exc: record({'key':'text_nli','error':str(exc)})
