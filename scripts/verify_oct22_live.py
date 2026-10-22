"""Live integration checks, not an accuracy benchmark for text or video."""
import json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
base='http://127.0.0.1:5173/api/v1'
out=[]
def save(entry):
 out.append(entry);(ROOT/'checkpoints/new-candidates/live.json').write_text(json.dumps(out,indent=2));print(entry.get('case'),entry.get('status'),str(entry.get('body',{}).get('verdict','')),flush=True)
for i in range(60):
 try:
  if requests.get(base+'/image_audio/health',timeout=2).status_code==200:break
 except requests.RequestException:pass
 time.sleep(1)
else:raise RuntimeError('Backend did not start')
for service in ['image_audio','video','text']:
 r=requests.get(base+'/'+service+'/health',timeout=10);r.raise_for_status();save({'case':service+' health','status':r.status_code,'body':r.json()})
rows=json.loads((ROOT/'checkpoints/new-candidates/media-benchmark-v2-image-test.json').read_text())
chosen=[min(rows,key=lambda x:x['score']),max(rows,key=lambda x:x['score']),sorted(rows,key=lambda x:x['score'])[len(rows)//2]]
for row in chosen:
 path=ROOT/'checkpoints/media-benchmark-v2'/row['path']
 r=requests.post(base+'/image_audio/analyze?modality=image',files={'file':(path.name,path.read_bytes())},timeout=90);r.raise_for_status();body=r.json()
 assert abs(body['score_ai']-row['score'])<.00011,(body['score_ai'],row['score'])
 assert body['signals'][0]['name']=='detectra_openfake'
 save({'case':path.name,'label':row['label'],'status':r.status_code,'body':body})
for name in ['portrait.avi','no_faces.avi']:
 path=ROOT/'checkpoints/fixtures'/name
 r=requests.post(base+'/video/analyze',files={'file':(name,path.read_bytes())},timeout=180)
 save({'case':name,'status':r.status_code,'body':r.json()})
for claim in ['Paris is the capital of France.','Paris is the capital of Germany.','The Moon is a natural satellite of Earth.','The fictional town of Zorbleston elected a purple dragon as mayor yesterday.']:
 r=requests.post(base+'/text/check-news',json={'text':claim,'max_claims':1,'single_claim':True,'include_evidence':True},timeout=180)
 save({'case':claim,'status':r.status_code,'body':r.json()})
