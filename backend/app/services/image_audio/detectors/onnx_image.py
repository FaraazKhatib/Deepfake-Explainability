"""Pinned Detectra OpenFake inference; CC-BY-NC-4.0, personal/research use.

Preprocessing and raw single-logit output follow the published model card:
https://huggingface.co/s-gehring/detectra-openfake
"""
import math
import time
import numpy as np
from PIL import Image
from .hf_image import HFImageClassifier
from .base import DetectorResult, LoadError


def prepare_image(image):
    image = image.convert('RGB')
    w,h = image.size
    scale = 440 / min(w,h)
    image = image.resize((int(w*scale),int(h*scale)),Image.Resampling.BILINEAR)
    x,y = (image.width-384)//2,(image.height-384)//2
    values = np.asarray(image.crop((x,y,x+384,y+384)),dtype=np.float32)/255
    values = (values-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return values.transpose(2,0,1)[None]


class ONNXImageClassifier(HFImageClassifier):
    def load(self):
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.spec.repo,'model.onnx',revision=self.spec.revision)
            options = ort.SessionOptions()
            options.intra_op_num_threads=2; options.inter_op_num_threads=1
            self._model=ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])
            if self._model.get_inputs()[0].name != 'pixel_values':
                raise ValueError('Unexpected model input')
            self.repo_used=self.spec.repo
            self.label_mapping='sigmoid(single logit): fake; raw uncalibrated score'
            self.device='cpu'
        except Exception as exc:
            self.error=str(exc)
            raise LoadError(self.error) from exc

    def predict(self,image):
        started=time.perf_counter()
        try:
            if not self.ready: raise ValueError(self.error or 'not loaded')
            logits=self._model.run(None,{'pixel_values':prepare_image(image)})[0]
            if np.asarray(logits).size != 1: raise ValueError('Expected one fake logit')
            z=float(np.asarray(logits).reshape(-1)[0])
            if not math.isfinite(z): raise ValueError('Non-finite model output')
            score=1/(1+math.exp(-max(-80,min(80,z))))
            return DetectorResult(self.key,self.kind,score,latency_ms=(time.perf_counter()-started)*1000,detail=self.spec.notes)
        except Exception as exc:
            return DetectorResult(self.key,self.kind,None,error=str(exc))

    def describe(self):
        return {**super().describe(),'revision':self.spec.revision,'license':'CC-BY-NC-4.0'}
