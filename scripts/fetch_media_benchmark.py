"""Download a small, reproducible diagnostic set from public HF datasets.

Labels and partitions are fixed before inference. Images use original files;
audio viewer assets may be transcoded. This is not the official full benchmark. Never use
the garystafford model here: it was trained on the audio dataset being sampled.
"""
import hashlib
import io
import json
from pathlib import Path
import random
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import os

ROOT = Path(__file__).resolve().parents[1] / os.getenv('VT_BENCHMARK_FOLDER','checkpoints/media-benchmark')
ROOT.mkdir(parents=True, exist_ok=True)
IMAGE_REVISION = '820a191bb0844ae74a1bd9eb57b28268257a4053'
AUDIO_REVISION = 'fcf5344bb7f82b54b6b932291326d29750ef1e82'

def get(url, **kwargs):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1)

def rows(dataset, split, offset, length=1):
    return get('https://datasets-server.huggingface.co/rows', params={
        'dataset': dataset, 'config': 'default', 'split': split,
        'offset': offset, 'length': length,
    }).json()['rows']

def save(entry, dataset, split, modality, partition):
    row = entry['row']; index = entry['row_idx']
    asset = row[modality]
    if isinstance(asset, list): asset = asset[0]
    source = asset['src'].split('?')[0]
    if modality == 'audio' and AUDIO_REVISION not in source:
        raise ValueError('Audio dataset revision changed; review labels and sample plan before downloading')
    suffix = Path(source).suffix
    filename = f'{modality}-{index}{suffix}'
    dest = ROOT / filename
    if not dest.exists(): dest.write_bytes(get(asset['src']).content)
    label = ('fake' if row['is_real'] == '1_fake' else 'real') if modality == 'image' else ['real','fake'][row['label']]
    return dict(modality=modality, path=filename, label=label, partition=partition,
                dataset=dataset, split=split, row_index=index, asset_source=source,
                dataset_revision=IMAGE_REVISION if modality=='image' else AUDIO_REVISION,
                source_file=row.get('file_name'), content_type=row.get('content_type'),
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest())

def main():
    manifest = []
    # Mirage's viewer rows endpoint is broken; its small parquet contains URLs,
    # so read metadata and fetch original images without viewer recompression.
    image_ds = 'Yunncheng/Mirage-Test'
    rng = random.Random(int(os.getenv('VT_BENCHMARK_SEED','20261008')))
    excluded = []
    if os.getenv('VT_BENCHMARK_EXCLUDE'):
        excluded = [json.loads(s) for s in Path(os.environ['VT_BENCHMARK_EXCLUDE']).read_text().splitlines()]
    exclude_ids = {(r['modality'],r['row_index']) for r in excluded}
    import pandas as pd
    data = get(f'https://huggingface.co/datasets/{image_ds}/resolve/{IMAGE_REVISION}/test.parquet').content
    (ROOT/'image-metadata.parquet').write_bytes(data)
    frame = pd.read_parquet(io.BytesIO(data))
    groups = {}
    for index, row in enumerate(frame.to_dict('records')):
        if ('image',index) in exclude_ids: continue
        if row['content_type'] == 'Anime': continue  # no real counterpart
        image_path = row['image'] if isinstance(row['image'], str) else row['image']['path']
        image_path = image_path.replace('/resolve/main/',f'/resolve/{IMAGE_REVISION}/')
        row['image'] = {'src': image_path}
        key = (row['content_type'], row['is_real'])
        groups.setdefault(key, []).append({'row_idx':index,'row':row})
    print('Image strata:', {str(k):len(v) for k,v in groups.items()}, flush=True)
    # Eight per domain/label where available; record actual counts transparently.
    tasks = []
    for key, entries in sorted(groups.items()):
        rng.shuffle(entries)
        for i, entry in enumerate(entries[:8]):
            tasks.append((entry, image_ds, 'test', 'image', 'dev' if i % 2 == 0 else 'test'))
    audio_ds = 'garystafford/deepfake-audio-detection'
    # Dataset viewer has 933 real rows followed by 933 fake rows. Assert labels
    # from the returned records, rather than treating that ordering as ground truth.
    for expected, base in [('real',0),('fake',933)]:
        indices = rng.sample([i for i in range(base,base+933) if ('audio',i) not in exclude_ids],30)
        with ThreadPoolExecutor(6) as pool:
            entries = list(pool.map(lambda index: rows(audio_ds,'train',index)[0], indices))
        for i, entry in enumerate(entries):
            assert ['real','fake'][entry['row']['label']] == expected
            tasks.append((entry,audio_ds,'train','audio','dev' if i < 10 else 'test'))
    with ThreadPoolExecutor(6) as pool:
        for result in pool.map(lambda args: save(*args), tasks):
            manifest.append(result)
            print(result['path'],result['label'],result['partition'],flush=True)
    assert len({r['sha256'] for r in manifest}) == len(manifest), 'Duplicate media in sample'
    assert not ({r['sha256'] for r in manifest} & {r['sha256'] for r in excluded}), 'Overlap with excluded sample'
    (ROOT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    print('Saved',len(manifest),'samples',flush=True)

if __name__ == '__main__': main()
