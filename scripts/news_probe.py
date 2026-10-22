"""Real retrieval + NLI sanity checks. Live results vary with retrieved evidence."""
import json
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1'
os.environ['FC_OFFLINE_ONLY']='true'
os.environ['FC_HTTP_TIMEOUT']='10'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
import torch
torch.set_num_threads(2)
from app.services.text.pipeline import FactCheckPipeline
pipeline=FactCheckPipeline(enable_style=False)
pipeline.load()
output={'pipeline':pipeline.info(),'cases':[]}
for claim in ['Paris is the capital of France.','Paris is the capital of Germany.','The fictional town of Zorbleston elected a purple dragon as mayor yesterday.']:
    try: result=pipeline.check(claim,treat_as_single_claim=True)
    except Exception as exc: result={'error':str(exc)}
    output['cases'].append({'claim':claim,'result':result})
    (ROOT/'checkpoints/news-probe.json').write_text(json.dumps(output,indent=2))
    print(claim,result.get('verdict'),result.get('retrieval',{}),flush=True)
