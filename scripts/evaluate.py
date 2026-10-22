"""Evaluate labelled local files/claims without mistaking smoke tests for accuracy.

JSONL rows: {"modality":"image|audio|video", "path":"relative/file", "label":"real|fake"}
or {"modality":"text", "text":"claim", "label":"supported|refuted|disputed|unverified"}.
Media paths resolve relative to the manifest. Keep train/calibration/test sets separate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
parser = argparse.ArgumentParser()
parser.add_argument('manifest',type=Path)
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
rows = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
for row in rows:
    if row.get('modality') not in {'image','audio','video','text'}: parser.error('Invalid modality')
    valid = {'supported','refuted','disputed','unverified'} if row['modality']=='text' else {'real','fake'}
    if row.get('label') not in valid: parser.error(f'Invalid label; expected {valid}')

engine = video = text_pipeline = None
if any(r['modality'] in {'image','audio'} for r in rows):
    from app.services.image_audio.engine import Engine
    engine = Engine(); engine.load()
if any(r['modality']=='video' for r in rows) or (
    any(r['modality']=='image' for r in rows) and engine is not None and not engine.registry.synthetic
):
    from app.config import settings
    from app.services.video.model_service import init_model_service
    from app.services.video.analyzer import analyze_video
    video = init_model_service(settings.model_name,settings.model_device)
if any(r['modality']=='text' for r in rows):
    from app.services.text.pipeline import FactCheckPipeline
    text_pipeline = FactCheckPipeline(enable_style=False); text_pipeline.load()

outputs = []
for row in rows:
    started = time.perf_counter()
    result = dict(row)
    try:
        modality = row['modality']
        if modality == 'text':
            raw = text_pipeline.check(row['text'],treat_as_single_claim=True)
            predicted = {'Supported by available evidence':'supported','Contradicted by available evidence':'refuted','Disputed by available evidence':'disputed','Unverified':'unverified'}[raw['verdict']]
            result['details'] = raw
        else:
            path = (args.manifest.resolve().parent/row['path']).resolve()
            if modality == 'video':
                raw = analyze_video(str(path))
                predicted = {'FAKE':'fake','REAL':'real','UNCERTAIN':'uncertain'}[raw.classification]
                result['score'] = raw.fake_probability
                result['model'] = raw.model_name
            else:
                from app.services.image_audio.audio import sniff_audio_mime
                data = path.read_bytes()
                if (sniff_audio_mime(data) is not None) != (modality == 'audio'):
                    raise ValueError('Manifest modality does not match media header')
                raw = engine.analyze(data,False)
                predicted = {'likely_ai_generated':'fake','likely_authentic':'real','uncertain':'uncertain'}[raw.verdict]
                result['score'], result['signals'] = raw.score_ai, raw.signals
        result['predicted'] = predicted
        result['correct'] = predicted == row['label']
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = round(time.perf_counter()-started,3)
    outputs.append(result)

summary = {}
for modality in sorted({r['modality'] for r in rows}):
    group = [r for r in outputs if r['modality']==modality]
    valid = [r for r in group if 'error' not in r]
    decided = [r for r in valid if r['predicted']!='uncertain']
    real = [r for r in valid if r['label']=='real']
    fake = [r for r in valid if r['label']=='fake']
    summary[modality] = {
        'total':len(group),'errors':len(group)-len(valid),'abstentions':len(valid)-len(decided),
        'coverage':len(decided)/len(group),
        'accuracy_counting_abstentions_and_errors_as_incorrect':sum(r.get('correct',False) for r in group)/len(group),
        'accuracy_on_decided':sum(r['correct'] for r in decided)/len(decided) if decided else None,
        'false_positive_rate':sum(r['predicted']=='fake' for r in real)/len(real) if real else None,
        'false_negative_rate':sum(r['predicted']=='real' for r in fake)/len(fake) if fake else None,
        'confusion':dict(Counter(f"{r['label']} -> {r.get('predicted','error')}" for r in group)),
    }
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps({'summary':summary,'models':engine.status() if engine else None,'results':outputs},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
