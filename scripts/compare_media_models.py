"""Score frozen diagnostic partitions; preserve per-file predictions and revisions.

Run dev first, record selection, then test only the selected profile and baseline.
No author-reported accuracy is used in this comparison.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import replace
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
import torch
torch.set_num_threads(2)
from app.services.image_audio.config import ALL_MODELS, ModelSpec, Settings
from app.services.image_audio.detectors.hf_image import HFImageClassifier
from app.services.image_audio.detectors.hf_audio import HFAudioClassifier
from app.services.image_audio.audio import decode_audio, make_windows, quantile
from app.services.image_audio.preprocessing import decode_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition',choices=['dev','test'],required=True)
    parser.add_argument('--models',required=True)
    args = parser.parse_args()
    folder = ROOT/os.getenv('VT_BENCHMARK_FOLDER','checkpoints/media-benchmark')
    manifest = [json.loads(line) for line in (folder/'manifest.jsonl').read_text().splitlines()]
    rows = [r for r in manifest if r['partition'] == args.partition]
    cfg = Settings()
    specs = {s.key:s for s in ALL_MODELS}
    specs['dima_image'] = ModelSpec('dima_image','dima806/deepfake_vs_real_image_detection','synthetic')
    target = folder/f'{args.partition}-scores.json'
    output = json.loads(target.read_text()) if target.exists() else {'partition':args.partition,'models':{}}
    for key in args.models.split(','):
        spec = specs[key.removesuffix('_long')]; modality = 'audio' if spec.kind == 'audio' else 'image'
        case_cfg = replace(cfg,audio_window_seconds=12.,audio_hop_seconds=6.) if key.endswith('_long') else cfg
        expected_rows = [r for r in rows if r['modality'] == modality]
        previous = output['models'].get(key, {})
        completed = {r['path'] for r in previous.get('results', [])}
        if completed == {r['path'] for r in expected_rows}:
            print('Already scored',key,flush=True); continue
        detector = (HFAudioClassifier if modality == 'audio' else HFImageClassifier)(spec,'cpu')
        detector.load()
        result = {'model':detector.describe(),'results':previous.get('results', []),
                  'audio_window_seconds':case_cfg.audio_window_seconds,'audio_quantile':cfg.audio_quantile}
        for row in expected_rows:
            if row['path'] in completed: continue
            started = time.perf_counter()
            data = (folder/row['path']).read_bytes()
            if modality == 'audio':
                decoded = decode_audio(data,cfg)
                windows,_ = make_windows(decoded,case_cfg)
                predictions = [detector.predict(w.samples,decoded.sample_rate) for w in windows]
            else:
                predictions = [detector.predict(decode_image(data,cfg).image)]
            scores = [p.p_fake for p in predictions if p.usable]
            entry = dict(row, scores=scores, score=quantile(scores,cfg.audio_quantile) if scores else None,
                         errors=[p.error for p in predictions if not p.usable], seconds=round(time.perf_counter()-started,3))
            result['results'].append(entry)
            output['models'][key] = result
            target.write_text(json.dumps(output,indent=2))
            print(key,row['path'],row['label'],entry['score'],flush=True)
        valid = [r for r in result['results'] if r['score'] is not None]
        print('BINARY ACCURACY',key,sum((r['score'] >= .5)==(r['label']=='fake') for r in valid),len(valid),flush=True)
        del detector
        gc.collect()

if __name__ == '__main__': main()
