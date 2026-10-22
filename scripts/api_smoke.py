"""Exercise the unified API with real cached models and locally held fixtures."""
import io
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['FC_OFFLINE_ONLY']='true'
os.environ['FC_ENABLE_STYLE']='false'
os.environ.setdefault('VT_MODELS','wavlm_itw,bisher_asv')
import cv2
import numpy as np
import torch
torch.set_num_threads(2)
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.services.text.retrieval.base import RetrievalReport

fixtures=ROOT/'checkpoints/fixtures'
rows=[]
def record(name,response):
    body=response.json()
    for key in ['all_frames','suspicious_frames','heatmap']:
        if key in body:
            body[key]=len(body[key]) if isinstance(body[key],list) else None
    rows.append({'case':name,'status':response.status_code,'body':body})
    (ROOT/'checkpoints/api-smoke.json').write_text(json.dumps(rows,indent=2))
    print(name,response.status_code,body.get('verdict',body.get('classification',body.get('detail'))),flush=True)

with TestClient(app) as client:
    started=time.monotonic()
    while app.state.pipeline is None and not app.state.startup_error and time.monotonic()-started < 120:
        time.sleep(0.2)
    for route in ['/api/v1/video/health','/api/v1/image_audio/models','/api/v1/text/health']:
        record(route,client.get(route))
    for name in ['human-speech-0.flac','human-speech-1.flac','synthetic-speech.wav']:
        record(name,client.post('/api/v1/image_audio/analyze?modality=audio',files={'file':(name,(fixtures/name).read_bytes())}))
    if (fixtures/'astronaut.png').exists():
        source=Image.open(fixtures/'astronaut.png').convert('RGB')
    else:
        source=Image.fromarray(np.random.default_rng(3).integers(0,256,(256,256,3),dtype=np.uint8))
    for fmt,quality in [('PNG',None),('JPEG',30),('WEBP',50)]:
        data=io.BytesIO(); source.save(data,format=fmt,**({'quality':quality} if quality else {}))
        record('image_'+fmt,client.post('/api/v1/image_audio/analyze?modality=image',files={'file':('photo.'+fmt.lower(),data.getvalue())}))
    for name,frame in [('portrait',np.asarray(source.resize((256,256)))[:,:,::-1]),('no_faces',np.zeros((256,256,3),dtype=np.uint8))]:
        path=fixtures/f'{name}.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(256,256))
        if not writer.isOpened(): raise RuntimeError('MJPG writer unavailable')
        for _ in range(20): writer.write(frame)
        writer.release()
        record('video_'+name,client.post('/api/v1/video/analyze',files={'file':(path.name,path.read_bytes(),'video/x-msvideo')}))
    record('video_corrupt',client.post('/api/v1/video/analyze',files={'file':('corrupt.mp4',b'not video','video/mp4')}))
    for text in ['', 'Why is the sky blue?']:
        record('text_invalid_'+text,client.post('/api/v1/text/check-news',json={'text':text}))
    # Network outages must abstain through the real API schema. Live network/NLI
    # behavior is tested separately by news_probe.py.
    with patch('app.services.text.pipeline.gather_evidence',return_value=RetrievalReport(errors={'fixture':'simulated upstream outage'})):
        record('text_no_evidence',client.post('/api/v1/text/check-news',json={'text':'Paris is the capital of France.'}))
