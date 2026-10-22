# Image and audio investigation — 8 October 2026

The reported missed fakes are reproducible. The previous image classifier and
two-model speech ensemble missed modern generated content. This update improves
the measured sample results, but audio remains weak on modern synthetic voices.

## What changed

- Images now use `haywoodsloan/ai-image-detector-deploy` (Swin), pinned to
  `4ae1822603cb1ec980c217cfde1ef1b8e0477a16`. The video classifier remains separate.
- Audio now uses WavLM alone, pinned to
  `1a9795f0ce971f9a1e2a512c443dbd889777cf50`. Bisher frequently voted real on
  neural fakes, suppressing WavLM's correct detections into uncertain results.
- Models agreeing on a decisive label no longer lose that verdict solely because
  their score magnitudes differ. Conflicting readings still trigger abstention.
- Opposite-polarity stereo no longer averages audible speech into silence. The
  strongest channel is retained when the mono mix loses most of its energy;
  the API reports this and explains it in the notes.
- Corrected Community Forensics single-logit inference is available experimentally,
  using sigmoid and verified aspect-preserving preprocessing. It did not perform
  well enough on the development sample to become the default. Gustking XLS-R is
  also available as an experimental audio option, not the default.
- Low audio scores explicitly warn that synthetic voices were missed in testing.

## Data and method

124 unique, labelled media files were downloaded before scoring. Selection used
only the development partition; `checkpoints/media-benchmark/selection.json`
records the chosen models and unchanged thresholds before reserved test inference.
No weights were trained or probability calibration fitted.

- Images: 64 original files from [Mirage-Test](https://huggingface.co/datasets/Yunncheng/Mirage-Test),
  revision `820a191bb0844ae74a1bd9eb57b28268257a4053`. Eight real and eight fake files
  in each of Animal, Human, Object and Scene; half assigned to development and
  half to testing. Random seed 20261008. Anime was excluded because there is no
  matching real stratum. Original files were used because the viewer rows API
  failed; no image-viewer thumbnails were benchmarked.
- Audio: 60 files from [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection),
  revision `fcf5344bb7f82b54b6b932291326d29750ef1e82`. Thirty random files per label:
  10 per label for development, 20 per label reserved for testing. The source
  corpus contains English human recordings and neural TTS from six platforms.
  Viewer audio assets were used and their source URLs and SHA256 hashes saved.
  The sample's per-generator coverage was not independently verified. Some actual
  files are stereo despite the card's general mono-format description.
- These are file-disjoint diagnostic samples, **not** a speaker-disjoint audio
  benchmark or a representative measurement of arbitrary user uploads. Training
  overlap cannot be ruled out for all community models. The Gary Stafford model
  was excluded because this dataset was used to train it.
- Four image checkpoints and seven audio checkpoints were compared on development
  data. A 12-second WavLM window trial was worse than the existing 4-second windows.
  Image thresholds stay 0.35/0.65; audio retains 4-second windows, 2-second hops
  and the 85th-percentile aggregation. No test-set threshold tuning was performed.

## Reserved test results

**Images: 32 files, 16 fake and 16 real.**

- Fake files correctly flagged: **0/16 → 13/16 (81.25%)**.
- Real files falsely flagged: **0/16 → 1/16 (6.25%)**.
- Uncertain results: **32/32 → 18/32** (15 real and 3 fake).
- If raw model scores are forced into binary labels at 0.5, accuracy is
  **53.125% → 87.5%**, a 34.375 percentage-point increase on this sample.
  This is **not the app's verdict accuracy**: the existing image policy keeps
  low, uncalibrated scores uncertain rather than affirming authenticity.
- Actual correct app verdicts, counting abstentions as misses: **0/32 → 13/32**.
  Of 14 definitive new image verdicts, 13 were correct. Reporting that 92.86%
  figure alone would conceal the 18 abstentions and is not an overall accuracy claim.

**Audio: 40 files, 20 fake and 20 real.**

- Fake files correctly flagged: **0/20 → 9/20 (45%)**.
- Fake files incorrectly called real: **11/20 → 11/20**. This serious weakness
  remains; the change recovers previously uncertain detections, not these misses.
- Real files falsely flagged: **1/20 → 1/20 (5%)**.
- Uncertain results: **10/40 → 0/40**.
- Actual correct app verdicts, counting abstentions as misses:
  **18/40 (45%) → 28/40 (70%)**, a 25 percentage-point increase on this sample.
- Forced binary score accuracy changes **65% → 70%**. Most of the app-level gain
  comes from removing the model that suppressed usable detections.

All per-file scores, labels, revisions, timing, confusion counts and hashes are
retained in `checkpoints/media-benchmark/`. The before comparison reconstructs the
previous spread rule and previous image-authenticity guard using saved raw scores.

## Verification and reproduction

61 regression tests pass, including label orders, invalid scores, decoders,
resampling, opposite-phase stereo, response serialization, single-logit sigmoid,
agreement versus disagreement, routes, video and text regression checks.

The live upload script exercises original images, JPEG quality 50, WebP quality 50,
original audio, WAV, quieter speech, opposite-phase stereo and MP3 through the
frontend proxy. Transformed files are correlated stress cases, not extra independent
accuracy samples. See `checkpoints/media-benchmark/api-stress.json` for results.
All 39 live uploads completed successfully; the tested transformations preserved
their original verdicts. Original API scores matched the offline comparison, and
opposite-phase stereo remained audible and exposed its diagnostic flag correctly.

```powershell
# Benchmark metadata reader is optional; not needed to serve the app.
.venv/Scripts/python.exe -m pip install pyarrow
.venv/Scripts/python.exe scripts/fetch_media_benchmark.py
.venv/Scripts/python.exe scripts/compare_media_models.py --partition dev --models dima_image,community_forensics,siglip_image,swin_image,wavlm_itw,bisher_asv,zeyadd_deepfake_v1,wav2vec2_base_itw,xlsr_deepfake,deepfake_wav2vec2,xlsr_general,wavlm_itw_long
# Freeze a selection before inspecting test scores.
.venv/Scripts/python.exe scripts/compare_media_models.py --partition test --models dima_image,swin_image,wavlm_itw,bisher_asv
.venv/Scripts/python.exe scripts/summarize_media_benchmark.py
.venv/Scripts/python.exe scripts/media_api_stress.py
```

## Checkpoints and remaining work

`05-before-image-audio-evaluation` preserves the source before this investigation.
`06-image-audio-evaluated` preserves the tested updated source. The archives exclude
dependencies and model caches; inference weights are pinned in configuration.

For a quick model-selection rollback, stop the backend and run:

```powershell
./scripts/start.ps1 -Profile previous
```

For the exact earlier source, restore to a new directory without overwriting work:

```powershell
.venv/Scripts/python.exe scripts/checkpoint.py restore 05-before-image-audio-evaluation --destination checkpoints/restored-before-media-evaluation
```

The largest remaining issue is out-of-distribution neural speech: 55% of the
reserved fakes still passed as real. More arbitrary ensemble members and looser
thresholds did not solve it in development trials. A next substantial accuracy
project needs a larger generator- and speaker-separated corpus covering the user's
actual upload types, then fine-tuning/calibration and a fresh untouched test set.
Image editing, screenshots, face swaps, music and mixed real/fake splices were not
validated by this generated-image and speech benchmark. Low scores do not prove
authenticity, and these diagnostic percentages must not be advertised as general
system accuracy.
