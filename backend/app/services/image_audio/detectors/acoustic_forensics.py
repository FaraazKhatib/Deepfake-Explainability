"""Acoustic forensic audio detector for physical, spectral, and generative artifact forensics.

Extracts spectral rolloff, Wiener entropy (flatness), high-frequency Nyquist steepness,
temporal dynamic range, silence floor, zero-crossing rate dynamics, and differential kurtosis.
Trained across multiple generative architectures including flow-matching (VoiceBox),
diffusion (FlashSpeech), neural vocoders (xTTS, seedtts, VALLE), and authentic recordings.
"""

from __future__ import annotations

import time
from pathlib import Path
import joblib
import numpy as np
import scipy.signal as signal

from .base import Detector, DetectorResult, LoadError
from ..config import AUDIO, ModelSpec

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "acoustic_forensics.joblib"


def extract_features_from_samples(samples: np.ndarray, sr: int = 16000) -> list[float] | None:
    if len(samples) < 800:
        return None
    samples = samples.astype(np.float32)
    max_val = np.max(np.abs(samples)) + 1e-9
    samples = samples / max_val

    # Resample to 16kHz if needed
    if sr != 16000:
        num_target = int(len(samples) * 16000 / sr)
        samples = signal.resample(samples, num_target)
        sr = 16000

    # 1. Spectral features
    f, t, Sxx = signal.spectrogram(samples, fs=sr, nperseg=512, noverlap=256)
    psd = np.mean(Sxx, axis=1) + 1e-12
    total_energy = np.sum(psd)

    # Frequency band energy ratios
    b_sub300 = float(np.sum(psd[f < 300]) / total_energy)
    b_300_1k = float(np.sum(psd[(f >= 300) & (f < 1000)]) / total_energy)
    b_1k_3k = float(np.sum(psd[(f >= 1000) & (f < 3000)]) / total_energy)
    b_3k_6k = float(np.sum(psd[(f >= 3000) & (f < 6000)]) / total_energy)
    b_6k_75k = float(np.sum(psd[(f >= 6000) & (f < 7500)]) / total_energy)
    b_above75k = float(np.sum(psd[f >= 7500]) / total_energy)

    e_6_75 = np.sum(psd[(f >= 6000) & (f < 7500)]) + 1e-12
    e_75_8 = np.sum(psd[f >= 7500]) + 1e-12
    nyquist_rolloff_ratio = float(e_75_8 / e_6_75)

    # Spectral flatness (Wiener entropy)
    geom_mean = np.exp(np.mean(np.log(psd)))
    arith_mean = np.mean(psd)
    flatness = float(geom_mean / arith_mean)

    # Centroid & Rolloff
    cum_psd = np.cumsum(psd) / total_energy
    r50 = float(f[np.searchsorted(cum_psd, 0.50)])
    r85 = float(f[np.searchsorted(cum_psd, 0.85)])
    r95 = float(f[np.searchsorted(cum_psd, 0.95)])
    centroid = float(np.sum(f * psd) / total_energy)
    spectral_spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * psd) / total_energy))

    # 2. Temporal & Silence features
    frame_len = int(sr * 0.04)  # 40ms
    frames = len(samples) // frame_len
    if frames == 0:
        return None
    rms_vals = [float(np.sqrt(np.mean(samples[i * frame_len : (i + 1) * frame_len] ** 2))) for i in range(frames)]
    rms_arr = np.array(rms_vals)
    zero_ratio = float(np.mean(rms_arr < 1e-5))
    floor_p05 = float(np.percentile(rms_arr, 5))
    floor_p15 = float(np.percentile(rms_arr, 15))
    rms_mean = float(np.mean(rms_arr))
    rms_std = float(np.std(rms_arr))
    dynamic_range = float(np.max(rms_arr) / (floor_p05 + 1e-6))

    # Zero-crossing rate
    zcr_vals = np.array([float(np.mean(np.diff(np.sign(samples[i * frame_len : (i + 1) * frame_len])) != 0)) for i in range(frames)])
    zcr_mean = float(np.mean(zcr_vals))
    zcr_std = float(np.std(zcr_vals))
    zcr_skew = float(np.mean(((zcr_vals - zcr_mean) / (zcr_std + 1e-9)) ** 3)) if zcr_std > 0 else 0.0

    # 3. High-order differential statistics
    d1 = np.diff(samples)
    d2 = np.diff(d1)
    kurt_d1 = float(np.mean(((d1 - np.mean(d1)) / (np.std(d1) + 1e-9)) ** 4))
    kurt_d2 = float(np.mean(((d2 - np.mean(d2)) / (np.std(d2) + 1e-9)) ** 4))

    # 4. Spectral flux
    if Sxx.shape[1] > 1:
        flux_vals = np.sqrt(np.sum(np.diff(Sxx, axis=1) ** 2, axis=0))
        flux_mean = float(np.mean(flux_vals))
        flux_std = float(np.std(flux_vals))
    else:
        flux_mean = 0.0
        flux_std = 0.0

    return [
        b_sub300, b_300_1k, b_1k_3k, b_3k_6k, b_6k_75k, b_above75k,
        nyquist_rolloff_ratio, flatness, r50, r85, r95, centroid, spectral_spread,
        zero_ratio, floor_p05, floor_p15, rms_mean, rms_std, dynamic_range,
        zcr_mean, zcr_std, zcr_skew,
        kurt_d1, kurt_d2, flux_mean, flux_std,
    ]


class AcousticForensicsDetector(Detector):
    def __init__(self, spec: ModelSpec | None = None):
        self.key = "acoustic_forensics"
        self.kind = AUDIO
        self.spec = spec or ModelSpec(
            key="acoustic_forensics",
            repo=str(WEIGHTS_PATH),
            kind=AUDIO,
            weight=1.5,
            notes="Acoustic forensic artifact classifier; specialized for flow-matching, diffusion, and neural vocoder artifacts.",
            is_local=True,
        )
        self.model = None
        self.error: str | None = None
        self.repo_used: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.error is None

    def load(self) -> None:
        if not WEIGHTS_PATH.is_file():
            self.error = f"Weights not found at {WEIGHTS_PATH}"
            raise LoadError(self.error)
        try:
            self.model = joblib.load(WEIGHTS_PATH)
            self.repo_used = str(WEIGHTS_PATH)
            self.error = None
        except Exception as exc:
            self.error = f"Failed to load acoustic forensics weights: {exc}"
            raise LoadError(self.error)

    def predict(self, samples: np.ndarray, sampling_rate: int = 16000) -> DetectorResult:
        if not self.ready:
            return DetectorResult(
                key=self.key,
                kind=self.kind,
                p_fake=None,
                error=self.error or "not loaded",
            )
        started = time.perf_counter()
        try:
            feats = extract_features_from_samples(samples, sampling_rate)
            if feats is None:
                raise ValueError("Insufficient audio samples to compute acoustic features")
            probs = self.model.predict_proba([feats])[0]
            p_fake = float(probs[1])
            p_fake = min(max(p_fake, 0.0), 1.0)
        except Exception as exc:
            return DetectorResult(
                key=self.key,
                kind=self.kind,
                p_fake=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

        return DetectorResult(
            key=self.key,
            kind=self.kind,
            p_fake=p_fake,
            latency_ms=(time.perf_counter() - started) * 1000,
            detail=self.spec.notes,
        )

    def describe(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "configured_repo": str(WEIGHTS_PATH),
            "loaded_repo": self.repo_used,
            "revision": None,
            "ready": self.ready,
            "label_mapping": "0=Real, 1=Fake",
            "weight": self.spec.weight,
            "notes": self.spec.notes,
            "error": self.error,
        }
