# Detector audit and model trials — 6–7 October 2026

The project is a React/Vite interface and a unified FastAPI application. Image/audio share an engine, video samples frames and classifies faces, and text retrieves evidence and uses NLI. **Text is a fact checker, not an AI-authorship detector. Video does not analyze the soundtrack or temporal motion.**

This work prioritizes the requested English speech, face swaps, and news claims. It improves correctness and adds model trials. It does **not** establish a new general-purpose accuracy percentage.

## Rollback checkpoints

The supplied folder was not a Git repository. `checkpoints/01-before-audit.zip` preserves all 87 original source/config/assets files, including the frontend lockfile. Its SHA-256 is `48a9b9ee73a9693488665ea815e12f4ed84353cff15756b9569de7005f6bd233`. A per-file checksum manifest accompanies it. The restore operation was exercised successfully into `checkpoints/baseline-source`.

`checkpoints/02-tested-update.zip` is the final source snapshot. Dependencies, uploaded media, cached model weights, build products and test outputs are excluded from source archives. The tested Python package versions are saved in `checkpoints/tested-environment.txt`; model revisions are exposed in registry status. Model Hub revisions are not pinned, so future Hub changes still require revalidation.

To restore without overwriting current work:

```powershell
.venv/Scripts/python.exe scripts/checkpoint.py restore 01-before-audit --destination restored-original
```

Run the restored backend/frontend from that new directory. It contains the old dependency specifications; reuse or recreate its environment as appropriate. The `legacy` launch profile restores **model selection only**, retaining the correctness fixes.

## Problems found and fixed

- **Video score bias:** fake-looking frames received quadratic extra weight. Replaced with equal frame averaging and a genuine uncertain band. Conflicting frames, missing faces, or failed inference cannot produce a confident binary result.
- **Hidden inference failures:** failed video predictions became 0.5, which could produce “FAKE.” Failed frames are excluded and reported; total failure returns unavailable. Non-finite/out-of-range detector outputs are rejected.
- **Missed secondary faces:** video checked only the largest face. It now scores up to eight detected face crops per frame. The maximum face score is a review signal, not a calibrated probability.
- **Unavailable face path:** image face detection was a stub; video depended on an uninstalled optional MTCNN package. Added an OpenCV Haar fallback. Integration testing found OpenCV 5 lacked `CascadeClassifier`; requirements now constrain OpenCV to 4.x. Haar still misses some profiles, occlusions and small faces.
- **Unsafe label assumptions:** video guessed the fake class, and text guessed NLI ordering. Both now refuse ambiguous labels. Image/audio also reject partially unknown class maps and invalid explicit indices.
- **Audio preprocessing:** entirely silent windows were still classified. They now abstain. Audio weights were incorrectly using the provenance weight; this is fixed. Each model receives its required sample rate. PyAV integer scaling now uses actual sample dtype instead of observed loudness. Audio duration is bounded during decoding rather than only after full expansion.
- **Ensemble disagreement:** a result could say “likely AI” even when models disagreed enough to erase confidence. Strong disagreement now returns uncertain in either direction.
- **Input handling:** bounded upload reads, strict base64 decoding/size checks, clean client errors for unknown formats, and modality checks for the image/audio pages. Blank and very small images are rejected rather than scored as evidence.
- **Event-loop blocking:** video model inference now runs in a worker thread behind a concurrency limit. Browser origins use the configured local allowlist.
- **Text extraction:** single-sentence questions/opinions bypassed filtering; long claims were silently cut at 400 characters. They now receive explicit validation errors. Merely saying “fact check” no longer triggers a lexical refutation, and a supporting NLI result is not inverted by debunk keywords.
- **Explanations:** text confidence bands were multiplied as numbers, displaying `NaN%`; the page also hid explanations and source links. These are restored. Video now displays warnings and plain-language verdicts. Media percentages are described as signal strength/estimates. Unimplemented provenance and heatmaps are disclosed.

## Model decisions

**Default English speech: `abhishtagatya/wavlm-base-960h-itw-deepfake` + `Bisher/wav2vec2_ASV_deepfake_audio_detection`.** These were already available in the ten-model pool. Both passed the small labelled speech sanity check, while several other members produced obvious errors. The full pool remains selectable with `VT_MODELS=all`, and the original ten audio models are available through the legacy profile. This is a provisional selection based on development examples, not a held-out benchmark or fitted ensemble.

**Image candidates implemented:** `Ateeqq/ai-vs-human-image-detector` for whole images and `prithivMLmods/Deepfake-Detect-Siglip2` for face crops. Added a real image-specific HF wrapper and registry support; previously the “image classifier” symbol was aliased to the audio wrapper and no dedicated image models were registered. Both candidates loaded and ran. The whole-image candidate falsely flagged a WebP-compressed NASA photograph in testing, so these are **opt-in**, not promoted defaults. Balanced mode retains the shared baseline image classifier.

**Video candidate implemented:** the existing configurable loader supports `prithivMLmods/Deepfake-Detect-Siglip2`, selected in the candidate profile. The default remains `dima806/deepfake_vs_real_image_detection`; the stronger changes here are face handling, scoring and uncertainty. Neither checkpoint has been validated here against a labelled face-swap video corpus. A newer image encoder is not automatically a better temporal/video detector.

**Text candidate configured:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` is selected by the candidate profile. It is larger and trained on additional NLI datasets. Its weights were **not downloaded or benchmarked in this run**; the working base model remains the default. Improving retrieval coverage and source handling matters independently of NLI size.

**CommunityForensics deferred:** its current card reports corrected weights/configuration and requires Transformers >=5.4 preprocessing; this project uses Transformers 4.x. A blind model-name replacement would be unsafe. This needs a separate compatibility migration and held-out image evaluation.

Primary model references:

- [Ateeqq model card](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) — includes an overfitting warning.
- [SigLIP2 deepfake model card](https://huggingface.co/prithivMLmods/Deepfake-Detect-Siglip2) — classifier architecture and reversed class ordering.
- [Current video baseline](https://huggingface.co/dima806/deepfake_vs_real_image_detection) — the author warns about concept drift.
- [Larger DeBERTa NLI candidate](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) — broader NLI training; published figures are not this project's accuracy.
- [CommunityForensics compatibility notes](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT).
- [AST model card](https://huggingface.co/MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection) — task-specific reported evaluation does not guarantee reliability on uploaded speech.

## Tests and empirical observations

The original snapshot failed **17 of the initial 49 regression tests**; the updated code passed those same 49. The final suite has **54 passing tests**, including blank images, duration limits, conflicting model votes, multiple faces and runtime face-backend availability. The frontend production build and Python compilation also pass.

`checkpoints/model-smoke.json` records real inference from two candidate image models, three audio models, the baseline video classifier and base NLI model. Tones/noise elicited extreme and conflicting scores; blank images could receive extreme scores. These diagnostic controls informed the safeguards but are not labelled deepfake accuracy tests.

`checkpoints/speech-probe.json` records all ten existing audio models on two human LibriSpeech recordings and one Windows-generated synthetic English clip. The proposed two-model pair produced approximately 0.02, 0.02 and 0.949 AI scores respectively through the real API. The full ten-model pool strongly disagreed on all three. Several individual models missed the synthetic clip; others called both human clips synthetic. **Two human clips from the same speaker and one older TTS voice cannot establish generalization to accents, unseen voice cloners, recording devices or compression. Selection used these examples, so they are development data.**

`checkpoints/news-probe.json` records live retrieval plus NLI:

- “Paris is the capital of France.” → supported.
- “Paris is the capital of Germany.” → contradicted.
- A fictional town electing a purple dragon → unverified.

GDELT returned HTTP 429 on two checks; Wikipedia/Google News continued and the missing-source diagnostics were preserved. These simple claims are sanity checks, not a breaking-news benchmark.

`checkpoints/api-smoke.json` exercises real startup, health/model status, all three speech files, PNG/JPEG/WebP images, generated portrait/no-face videos, corrupt video, invalid text and simulated evidence failure. `api-smoke-image-candidates.json` preserves the earlier candidate-image experiment and its false positive. The portrait videos are repeated photographs, **not labelled face swaps**.

The final API run detected a face in all 16 sampled portrait frames and returned a low manipulation score (0.0852). The no-face video abstained. Corrupt video returned 422; invalid text returned 400/422; unavailable evidence returned Unverified. All three photo encodings remained uncertain under the conservative baseline image policy. The three modality health/status checks returned 200, with the OpenCV face backend available.

Fixtures: two validation rows from [LibriSpeech dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy), a local Windows TTS clip, and the [scikit-image NASA astronaut sample](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/astronaut.png). No personal uploads were used.

## Remaining limitations and next accuracy work

- No representative held-out corpus was supplied. A meaningful accuracy claim needs real/clone English speech from multiple speakers/generators, genuine/face-swap videos across identities and compression settings, and dated news claims with adjudicated evidence. Split by speaker/identity/generator, not random near-duplicate files.
- Scores and thresholds are uncalibrated. The repository mentions calibration scripts/data that are absent. `scripts/evaluate.py` now provides repeatable labelled evaluation with coverage, abstentions, errors, confusion counts and false-positive/false-negative rates; it does not fit calibration automatically.
- The source-credibility CSV is absent. Most domains use the unrated default; explicit institutional suffix rules still apply. A reviewed source list, deduplication across related subdomains/syndication, and retrieval of complete contextual evidence remain useful work. Lexical debunk matching remains heuristic.
- Audio has a silence gate, not a validated speech/non-speech detector. Music/noise, phase-cancelling stereo, partially cloned clips and unseen generators need dedicated evaluation. Duration is capped at 600 seconds; long clips must be shortened.
- Video uses sparse frame samples and image classifiers. It can miss brief swaps between samples, temporal inconsistencies and lip-sync attacks. It does not check audio. Multiple-face maximum scoring has its own false-positive tradeoff.
- Provenance verification and heatmaps remain unimplemented. They are explicitly reported as unavailable, not fabricated.
- This is still a local development service: no user authentication/rate limiting, request-level multipart disk quotas or production job queue. CORS is not authentication. Use a suitably configured gateway before exposing it publicly.

Promote a candidate only after measuring improvement on an untouched test set at an acceptable false-positive rate, with abstention coverage and CPU latency reported alongside accuracy. Fine-tuning on representative failure cases and calibrating on a separate split are more defensible routes to substantial improvement than accumulating checkpoints.

## Browser connection fix — 8 October 2026

The running UI at `127.0.0.1:5173` encountered “Failed to fetch” because it sent requests directly to `localhost:8001` while the default backend CORS allowlist omitted the numeric loopback origin. The unused Vite proxy also targeted port 8000. Development now sends same-origin `/api` requests through a proxy to `127.0.0.1:8001`; the backend's default allowlist includes both loopback spellings for future starts. Health checks for video, image/audio and text returned 200 through this proxy, and an actual PNG upload returned a valid analysis with HTTP 200. The frontend production build passes. Source checkpoints `03-before-browser-connection-fix` and `04-browser-connection-fixed` preserve the before/after states.
# Image/audio follow-up, 8 October 2026

The broader [124-file investigation](MEDIA_EVALUATION.md) supersedes the earlier
three-speech-clip model selection. Defaults are now pinned Swin image detection
and WavLM audio alone. Reserved tests exposed substantial remaining audio misses;
do not interpret the earlier smoke tests as an accuracy measurement. Checkpoint
05 preserves the previous source; checkpoint 06 preserves this tested update.

