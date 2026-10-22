"""Exercise the live frontend proxy with original and recompressed labelled media.

Uses development files only; variants are correlated, not extra benchmark samples.
"""
import io
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'checkpoints/media-benchmark'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality',choices=['all','image','audio'],default='all')
    args = parser.parse_args()
    manifest = [json.loads(s) for s in (FOLDER/'manifest.jsonl').read_text().splitlines()]
    scores = json.loads((FOLDER/'dev-scores.json').read_text())['models']
    selected = []
    groups = set()
    for row in manifest:
        if row['partition'] != 'dev': continue
        if args.modality != 'all' and row['modality'] != args.modality: continue
        key = (row['modality'],row['label'],row.get('content_type'))
        if key not in groups:
            selected.append(row); groups.add(key)
    if args.modality in {'all','audio'}:
        # Exercise both a known miss and a successful fake detection through the
        # live audio endpoint; format stability alone can hide an all-real bug.
        positive = next(r for r in scores['wavlm_itw']['results'] if r['label']=='fake' and r['score']>=.65)
        if not any(r['path']==positive['path'] for r in selected):
            selected.append(next(r for r in manifest if r['path']==positive['path']))
    results = []
    target = FOLDER/'api-stress.json'
    if args.modality != 'all' and target.exists():
        results = [r for r in json.loads(target.read_text()) if r['modality'] != args.modality]
    for row in selected:
        path = FOLDER/row['path']; data = path.read_bytes()
        variants = [('original',path.name,data)]
        if row['modality'] == 'image':
            im = Image.open(io.BytesIO(data)).convert('RGB')
            for fmt,quality in [('JPEG',50),('WEBP',50)]:
                output = io.BytesIO(); im.save(output,format=fmt,quality=quality)
                variants.append((fmt.lower()+'50','variant.'+fmt.lower(),output.getvalue()))
        else:
            samples,rate = sf.read(io.BytesIO(data),dtype='float32')
            if samples.ndim == 2: samples = samples.mean(axis=1)
            for name,values,fmt in [('wav',samples,'WAV'),('quieter',samples*.25,'WAV'),
                                    ('opposite_phase',np.stack([samples,-samples],axis=1),'WAV'),
                                    ('mp3',samples,'MP3')]:
                output = io.BytesIO(); sf.write(output,values,rate,format=fmt)
                variants.append((name,'variant.'+fmt.lower(),output.getvalue()))
        base_score = None
        for name,filename,payload in variants:
            response = requests.post(f'http://127.0.0.1:5173/api/v1/image_audio/analyze?modality={row["modality"]}',files={'file':(filename,payload)},timeout=120)
            response.raise_for_status(); body = response.json()
            assert not body.get('errors'),body.get('errors')
            if name == 'original':
                base_score = body['score_ai']
                model_key = 'swin_image' if row['modality']=='image' else 'wavlm_itw'
                expected = next(r['score'] for r in scores[model_key]['results'] if r['path']==path.name)
                assert abs(base_score-min(.98,max(.02,expected))) < .00011, (path,base_score,expected)
            if name == 'opposite_phase':
                assert body['image']['phase_cancellation_avoided']
                assert body['image']['windows_scored'] > 0
                assert abs(body['score_ai']-base_score) < .01
            result = dict(path=path.name,label=row['label'],modality=row['modality'],variant=name,
                          score=body['score_ai'],verdict=body['verdict'],signals=body['signals'])
            results.append(result)
            (FOLDER/'api-stress.json').write_text(json.dumps(results,indent=2))
            print(path.name,name,result['verdict'],result['score'],flush=True)
    print('Completed',len(results),'live uploads',flush=True)

if __name__ == '__main__': main()
