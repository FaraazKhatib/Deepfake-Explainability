"""Small reproducible diagnostics; not a population accuracy estimate.

Video: seeded sample from the first 100 files per FF++ mirror category.
News: seeded, balanced unique claims from FEVER labelled development data.
All predictions, failures and provenance are retained under checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'checkpoints/video-news-benchmark'
OUT.mkdir(exist_ok=True)
REV = 'f2980e9a2f43320503ab2a1ac1b65613621879c5'
REPO = 'gonnerthetooner/FaceForensics'


def fetch():
    rng = random.Random(20261022)
    rows = []
    for category, count in [('original', 12), ('FaceSwap', 6), ('Deepfakes', 6)]:
        listing = json.loads((ROOT / f'checkpoints/new-candidates/video-{category}.json').read_text())
        eligible = [x for x in listing if x['type'] == 'file' and x['path'].endswith('.mp4') and x['size'] < 8_000_000]
        for i, item in enumerate(rng.sample(eligible, count)):
            name = category + '-' + Path(item['path']).name
            target = OUT / name
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{item['path']}"
            if not target.exists():
                r = requests.get(url, timeout=90); r.raise_for_status(); target.write_bytes(r.content)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest == item['lfs']['oid'], name
            rows.append({'path':name, 'label':int(category != 'original'), 'category':category,
                         'partition':'dev' if i < count // 2 else 'test', 'sha256':digest, 'url':url})
            print('Downloaded', name, flush=True)
    (OUT / 'video-manifest.json').write_text(json.dumps(rows, indent=2))
    url = 'https://huggingface.co/datasets/fever/fever/resolve/55e20b98b435ba817a6c9b4e05871932bd164645/v1.0/fever-labelled_dev.parquet'
    target = OUT / 'fever-labelled_dev.parquet'
    if not target.exists():
        r = requests.get(url, timeout=90); r.raise_for_status(); target.write_bytes(r.content)
    import pyarrow.parquet as pq
    unique = {row['id']: row for row in pq.read_table(target).to_pylist()}
    claims = []
    for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
        pool = sorted([r for r in unique.values() if r['label'] == label], key=lambda x:x['id'])
        for row in rng.sample(pool, 4):
            claims.append(dict(id=row['id'], claim=row['claim'], label=label, source=url))
    (OUT / 'news-manifest.json').write_text(json.dumps(claims, indent=2))
    print('Prepared', len(rows), 'videos and', len(claims), 'claims', flush=True)


def video(model, partition):
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT / 'backend'))
    import torch
    torch.set_num_threads(2)
    from app.services.video.model_service import init_model_service
    from app.services.video.analyzer import analyze_video
    svc = init_model_service(model, 'cpu')
    assert svc.is_loaded and svc.model_name == model, svc.model_name
    target = OUT / (model.split('/')[-1] + '-' + partition + '.json')
    results = json.loads(target.read_text()) if target.exists() else []
    done = {x['path'] for x in results}
    for row in json.loads((OUT / 'video-manifest.json').read_text()):
        if row['partition'] != partition or row['path'] in done: continue
        start = time.perf_counter()
        try:
            r = analyze_video(str(OUT / row['path']))
            result = dict(row, score=r.fake_probability, verdict=r.classification,
                          faces=r.faces_detected, warning=r.warning, model=r.model_name,
                          frames=[dict(index=f.frame_index, score=f.fake_probability, face=f.face_detected) for f in r.all_frames])
        except Exception as e:
            result = dict(row, error=str(e))
        result['seconds'] = round(time.perf_counter() - start, 2)
        results.append(result); target.write_text(json.dumps(results, indent=2))
        print(row['path'], result.get('score'), result.get('verdict',result.get('error')), flush=True)


def news(tag):
    target = OUT / f'news-{tag}.json'
    results = json.loads(target.read_text()) if target.exists() else []
    done = {x['id'] for x in results}
    for row in json.loads((OUT / 'news-manifest.json').read_text()):
        if row['id'] in done: continue
        try:
            r = requests.post('http://127.0.0.1:5173/api/v1/text/check-news',
                              json={'text':row['claim'],'max_claims':1,'single_claim':True,'include_evidence':True}, timeout=180)
            r.raise_for_status()
            result = dict(row, body=r.json())
        except Exception as e:
            result = dict(row, error=str(e))
        results.append(result); target.write_text(json.dumps(results, indent=2))
        print(row['claim'], row['label'], result.get('body',{}).get('verdict',result.get('error')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['fetch','video','news'])
    p.add_argument('--model', default='dima806/deepfake_vs_real_image_detection')
    p.add_argument('--partition', choices=['dev','test'], default='dev')
    p.add_argument('--tag', default='baseline')
    a = p.parse_args()
    if a.mode == 'fetch': fetch()
    elif a.mode == 'video': video(a.model, a.partition)
    else: news(a.tag)
