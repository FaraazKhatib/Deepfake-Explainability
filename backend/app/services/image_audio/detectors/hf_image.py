"""Image-specific HF wrapper; checkpoint preprocessing and label order are authoritative."""
import math
import time

from .base import DetectorResult, LoadError
from .hf_audio import HFAudioClassifier, resolve_fake_indices


class HFImageClassifier(HFAudioClassifier):
    def load(self):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        failures = []
        for repo in (self.spec.repo, *self.spec.fallbacks):
            try:
                revision = {"revision": self.spec.revision} if self.spec.revision else {}
                processor = AutoImageProcessor.from_pretrained(repo, use_fast=False, **revision)
                model = AutoModelForImageClassification.from_pretrained(
                    repo, torch_dtype=self._torch_dtype(), trust_remote_code=False, **revision
                )
                labels = {int(k): v for k, v in model.config.id2label.items()}
                if self.spec.single_logit_fake:
                    if model.config.num_labels != 1:
                        raise LoadError("Verified sigmoid detector requires exactly one output logit.")
                    if self.key == "community_forensics":
                        self._validate_community_forensics(model.config, processor)
                    indices, mapping = [0], "Verified single logit: sigmoid(logit) = fake"
                elif self.spec.fake_index is not None:
                    if self.spec.fake_index not in labels or len(labels) != 2:
                        raise LoadError("Explicit fake_index requires a valid binary classifier.")
                    indices, mapping = [self.spec.fake_index], f"Verified override: {labels}"
                else:
                    indices, mapping = resolve_fake_indices(labels)
                model.eval().to(self.device)
                self._processor, self._model = processor, model
                self._fake_indices, self.label_mapping = indices, mapping
                self.repo_used, self.error = repo, None
                return
            except Exception as exc:
                failures.append(f"{repo}: {type(exc).__name__}: {exc}")
        self.error = " | ".join(failures)
        raise LoadError(self.error)

    @staticmethod
    def _validate_community_forensics(config, processor):
        # Earlier releases silently used the wrong weights, attention heads and
        # a square resize. Refuse those instead of returning plausible bad scores.
        expected = {"hidden_size":384, "num_attention_heads":6, "intermediate_size":1536}
        if any(getattr(config, k, None) != v for k,v in expected.items()):
            raise LoadError("Community Forensics requires the corrected ViT-Small checkpoint.")
        if (processor.size != {"shortest_edge":440}
                or processor.crop_size != {"height":384,"width":384}
                or not processor.do_center_crop):
            raise LoadError("Community Forensics requires aspect-preserving resize 440 and crop 384.")

    def predict(self, image):
        if not self.ready:
            return DetectorResult(self.key, self.kind, None, error=self.error or "not loaded")
        import torch

        started = time.perf_counter()
        try:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs = {k: v.to(self._torch_dtype()) if v.is_floating_point() else v for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits.float()
                if self.spec.single_logit_fake:
                    if logits.shape != (1, 1):
                        raise ValueError("Expected one fake logit per image")
                    score = float(torch.sigmoid(logits[0, 0]).item())
                else:
                    probs = torch.softmax(logits, dim=-1)[0]
                    score = float(probs[self._fake_indices].sum().item())
            if not math.isfinite(score):
                raise ValueError("Model returned a non-finite score")
            return DetectorResult(self.key, self.kind, score,
                                  latency_ms=(time.perf_counter() - started) * 1000,
                                  detail=f"{self.repo_used}. {self.spec.notes}")
        except Exception as exc:
            return DetectorResult(self.key, self.kind, None, error=f"{type(exc).__name__}: {exc}")
