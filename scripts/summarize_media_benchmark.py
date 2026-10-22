"""Compare saved model scores with the old and new serving decision rules."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.services.image_audio.config import Settings
from app.services.image_audio.fusion import Signal, fuse

def summarize(models,keys,modality,legacy=False):
    cfg = Settings()
    records = {k:{r['path']:r for r in models[k]['results']} for k in keys}
    paths = set(records[keys[0]])
    assert all(set(records[k]) == paths for k in keys), 'Incomplete score file'
    predictions = []
    for path in sorted(paths):
        row = records[keys[0]][path]
        result = fuse([Signal(k,records[k][path]['score'],models[k]['model']['weight'],
                              models[k]['model']['kind']) for k in keys],cfg.thresholds,cfg.fusion)
        verdict = result.verdict
        # Reproduce the previous spread rule; all other fusion settings unchanged.
        if legacy and result.spread_exceeds_limit: verdict = 'uncertain'
        if modality == 'image' and verdict == 'likely_authentic': verdict = 'uncertain'
        predicted = {'likely_authentic':'real','likely_ai_generated':'fake','uncertain':'uncertain'}[verdict]
        predictions.append({'path':path,'label':row['label'],'score':result.score,'predicted':predicted,
                            'binary_prediction':'fake' if result.score>=.5 else 'real'})
    n = len(predictions); real = [r for r in predictions if r['label']=='real']; fake = [r for r in predictions if r['label']=='fake']
    correct = sum(r['label']==r['predicted'] for r in predictions)
    decided = sum(r['predicted']!='uncertain' for r in predictions)
    return {'n':n,'real_n':len(real),'fake_n':len(fake),
            'binary_score_accuracy_at_0_5':sum(r['label']==r['binary_prediction'] for r in predictions)/n,
            'correct_verdicts':correct,'correct_verdict_rate_abstentions_count_as_misses':correct/n,
            'abstentions':n-decided,'coverage':decided/n,
            'decided_accuracy':correct/decided if decided else None,
            'fakes_detected':sum(r['predicted']=='fake' for r in fake),
            'fake_recall':sum(r['predicted']=='fake' for r in fake)/len(fake),
            'fakes_called_real':sum(r['predicted']=='real' for r in fake),
            'false_alarms':sum(r['predicted']=='fake' for r in real),
            'false_positive_rate':sum(r['predicted']=='fake' for r in real)/len(real),
            'confusion':dict(Counter(f"{r['label']} -> {r['predicted']}" for r in predictions)),
            'predictions':predictions}

def main():
    folder = ROOT/'checkpoints/media-benchmark'
    all_results = {}
    for partition in ['dev','test']:
        models = json.loads((folder/f'{partition}-scores.json').read_text())['models']
        all_results[partition] = {
            'image_before':summarize(models,['dima_image'],'image',True),
            'image_after':summarize(models,['swin_image'],'image'),
            'audio_before':summarize(models,['wavlm_itw','bisher_asv'],'audio',True),
            'audio_after':summarize(models,['wavlm_itw'],'audio'),
        }
    (folder/'summary.json').write_text(json.dumps(all_results,indent=2))
    for partition,groups in all_results.items():
        for name,metrics in groups.items():
            print(partition,name,json.dumps({k:v for k,v in metrics.items() if k!='predictions'}))

if __name__ == '__main__': main()
