"""Local candidate evaluation, never enables a model automatically."""
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('kind',choices=['image','audio']);p.add_argument('--partition',default='dev');args=p.parse_args()
if args.kind=='audio':sys.path.insert(0,str(ROOT/'checkpoints/pella-runtime'))
sys.path.insert(0,str(ROOT/'backend'))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
torch.set_num_threads(2)
from huggingface_hub import hf_hub_download
from app.services.image_audio.config import Settings
from app.services.image_audio.preprocessing import decode_image
folder=ROOT/os.getenv('VT_BENCHMARK_FOLDER','checkpoints/media-benchmark')
rows=[json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['modality']==args.kind and r['partition']==args.partition]
if args.kind=='image':
 import onnxruntime as ort
 from PIL import Image
 opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
 model=ort.InferenceSession(hf_hub_download('s-gehring/detectra-openfake','model.onnx',revision='ef971ed6ecca0205d8b001e9326727fb5c00e8a8'),sess_options=opt,providers=['CPUExecutionProvider'])
 def predict(path):
  im=decode_image(path.read_bytes(),Settings()).image
  w,h=im.size;scale=440/min(w,h);im=im.resize((int(w*scale),int(h*scale)),Image.Resampling.BILINEAR)
  x=(im.width-384)//2;y=(im.height-384)//2
  arr=np.asarray(im.crop((x,y,x+384,y+384)),dtype=np.float32)/255
  arr=(arr-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
  logit=float(model.run(None,{'pixel_values':arr.transpose(2,0,1)[None]})[0].reshape(-1)[0])
  return float(1/(1+np.exp(-np.clip(logit,-80,80))))
else:
 import importlib.util
 import transformers.utils.import_utils as imports
 imports._torchvision_available=False  # Audio-only isolated runtime needs no vision extension.
 from transformers import Wav2Vec2BertModel,Wav2Vec2BertConfig
 # Reviewed upstream code; instantiate architecture without downloading a second
 # backbone weight file, then strictly load the detector's complete state dict.
 spec=importlib.util.spec_from_file_location('pella_reviewed',ROOT/'checkpoints/new-candidates/pellav2_infer.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 original=Wav2Vec2BertModel.from_pretrained
 Wav2Vec2BertModel.from_pretrained=lambda *a,**k:Wav2Vec2BertModel(Wav2Vec2BertConfig.from_pretrained('facebook/w2v-bert-2.0',revision='da985ba0987f70aaeb84a80f2851cfac8c697a7b'))
 model=module.Detector();Wav2Vec2BertModel.from_pretrained=original
 weights=hf_hub_download('Sadanie/pellav2-audio-deepfake-detector','pellav2_detector.pt',revision='84f7a0d0a22903220b416aed5e45169b6613e9d4')
 model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.eval()
 def predict(path):
  with torch.inference_mode():return float(torch.sigmoid(model(torch.from_numpy(module.load_clip(str(path)))[None])).item())
out=[]
for row in rows:
 start=time.time();score=predict(folder/row['path']);out.append(dict(row,score=score,seconds=time.time()-start))
 (ROOT/f'checkpoints/new-candidates/{folder.name}-{args.kind}-{args.partition}.json').write_text(json.dumps(out,indent=2))
 print(row['path'],row['label'],score,flush=True)
print('Correct binary',sum((r['score']>=.5)==(r['label']=='fake') for r in out),'/',len(out),flush=True)
