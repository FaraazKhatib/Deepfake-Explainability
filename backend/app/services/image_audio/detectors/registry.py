"""Builds and owns the detector instances.

Loading is best effort. Each checkpoint that fails is recorded and the ensemble continues with
whatever loaded, so one renamed Hugging Face repo cannot take the service down. The status is
exposed verbatim through /api/v1/models.

Two specs in the same pathway can resolve to the same checkpoint, because one spec's fallback is
another's primary. That has to be caught here: identical weights under two names would cast two
votes, double their weight in the fused mean and report perfect agreement with themselves, which
reads as corroboration when it is one opinion counted twice. Deduplication is per pathway rather
than global, since the face and whole image pathways feed a model different pixels and their
readings are genuinely separate observations even when the checkpoint is shared.
"""

from __future__ import annotations

import os

from ..config import (
    ALL_MODELS,
    LOCAL_SPEC_PROBLEMS,
    ModelSpec,
    Settings,
)
from .base import LoadError
from .hf_audio import HFAudioClassifier
from .hf_image import HFImageClassifier


def uses_raw_torch_checkpoint(_spec: ModelSpec) -> bool:
    return False


class Registry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = settings.resolved_device()
        self.synthetic: list[HFImageClassifier] = []
        self.face: list[HFImageClassifier] = []
        self.audio: list[HFAudioClassifier] = []
        self.failures: list[dict] = []
        for problem in LOCAL_SPEC_PROBLEMS:
            self.failures.append({"key": "models.local.json", "kind": "config", "error": problem})

    def _build(self, specs: tuple[ModelSpec, ...], factory=None) -> list:
        # factory is resolved here rather than as a default argument on purpose. A default is
        # evaluated at definition time and would capture the original class, so the tests that
        # monkeypatch registry.HFImageClassifier to a stub loader would silently keep loading the
        # real one and start reaching for the network.
        factory = factory or HFImageClassifier
        loaded: list = []
        claimed: dict[str, str] = {}
        for spec in specs:
            actual_factory = factory
            if spec.key == 'detectra_openfake':
                from .onnx_image import ONNXImageClassifier
                actual_factory = ONNXImageClassifier
            
            detector = actual_factory(spec, self.device, self.settings.dtype)
            try:
                detector.load()
            except LoadError as exc:
                self.failures.append({"key": spec.key, "kind": spec.kind, "error": str(exc)})
                continue
            except Exception as exc:
                self.failures.append(
                    {"key": spec.key, "kind": spec.kind, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            # Only knowable after load, which walks the fallback chain to find something that
            # resolves. The duplicate is dropped rather than kept, and the reference goes out of
            # scope so the weights can be collected.
            repo = detector.repo_used
            if repo is not None and repo in claimed:
                self.failures.append(
                    {
                        "key": spec.key,
                        "kind": spec.kind,
                        "error": (
                            f"resolved to {repo}, already loaded as {claimed[repo]}. Dropped so one "
                            f"checkpoint does not vote twice. Its own repo was unavailable and its "
                            f"fallback collided."
                        ),
                    }
                )
                continue

            if repo is not None:
                claimed[repo] = spec.key
            loaded.append(detector)
        return loaded

    def load_all(self) -> None:
        # Selected using the frozen development sample described in MEDIA_EVALUATION.md.
        # Additional models can suppress valid detections; more votes are not
        # automatically better. "all" remains an experimental comparison pool.
        selected = os.getenv("VT_MODELS", "detectra_openfake,wavlm_itw")
        keys = None if selected == "all" else {k.strip() for k in selected.split(",")}
        specs = tuple(s for s in ALL_MODELS if keys is None or s.key in keys)
        if keys is not None:
            for key in keys - {s.key for s in ALL_MODELS} - {""}:
                self.failures.append({"key": key, "kind": "config", "error": "Unknown model key"})
        self.synthetic = self._build(tuple(s for s in specs if s.kind == "synthetic"))
        self.face = self._build(tuple(s for s in specs if s.kind == "face")) if self.settings.enable_faces else []
        # A fresh claimed map per call is what makes deduplication per pathway. Audio shares the
        # rule but never the map: no audio checkpoint can collide with an image one anyway, since
        # they load through different auto classes.
        if self.settings.enable_audio:
            self.audio = self._build(tuple(s for s in specs if s.kind == "audio"), factory=HFAudioClassifier)

    @property
    def any_ready(self) -> bool:
        return bool(self.synthetic or self.face or self.audio)

    def status(self) -> dict:
        return {
            "device": self.device,
            "dtype": self.settings.dtype,
            "synthetic": [d.describe() for d in self.synthetic],
            "face": [d.describe() for d in self.face],
            "audio": [d.describe() for d in self.audio],
            "failures": self.failures,
            "ensemble_size": len(self.synthetic) + len(self.face) + len(self.audio),
        }
