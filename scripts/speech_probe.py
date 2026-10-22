"""Small labelled sanity check, NOT a representative accuracy evaluation."""
import gc
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
import torch
torch.set_num_threads(2)
from app.services.image_audio.config import AUDIO_MODELS, Settings
from app.services.image_audio.audio import decode_audio,make_windows,quantile
from app.services.image_audio.detectors.hf_audio import HFAudioClassifier
from app.services.image_audio.fusion import fuse,Signal
cfg=Settings()
fixtures=ROOT/'checkpoints/fixtures'
cases={p.name:make_windows(decode_audio(p.read_bytes(),cfg),cfg)[0] for p in fixtures.iterdir() if p.suffix in {'.wav','.flac'}}
output={'limitation':'Two human clips from one LibriSpeech speaker and one Windows TTS clip. Cannot estimate general accuracy.','models':[]}
for spec in AUDIO_MODELS:
    detector=HFAudioClassifier(spec,'cpu')
    try:
        detector.load()
        scores={}
        for name,windows in cases.items():
            predictions=[detector.predict(w.samples) for w in windows]
            scores[name]={'score':quantile([p.p_fake for p in predictions if p.usable],cfg.audio_quantile) if any(p.usable for p in predictions) else None,'errors':[p.error for p in predictions if not p.usable]}
        entry={'key':spec.key,'repo':detector.repo_used,'weight':spec.weight,'scores':scores}
    except Exception as exc: entry={'key':spec.key,'error':str(exc)}
    output['models'].append(entry)
    (ROOT/'checkpoints/speech-probe.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(entry),flush=True)
    del detector
    gc.collect()
output['ensembles']={}
for profile,keys in [('full',{s.key for s in AUDIO_MODELS}),('balanced',{'wavlm_itw','bisher_asv'})]:
    output['ensembles'][profile]={}
    for name in cases:
        signals=[Signal(m['key'],m['scores'][name]['score'],m['weight'],'audio') for m in output['models'] if m['key'] in keys and 'scores' in m and m['scores'][name]['score'] is not None]
        output['ensembles'][profile][name]=fuse(signals,cfg.thresholds,cfg.fusion).as_dict()
(ROOT/'checkpoints/speech-probe.json').write_text(json.dumps(output,indent=2))
