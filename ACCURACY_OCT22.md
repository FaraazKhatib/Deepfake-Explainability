# Accuracy follow-up — 22 October 2026

## Repeated 2% score

The fusion code clipped scores to 0.02–0.98 to prevent one ensemble member
dominating other members. The default had only one image detector, so every raw
reading below 0.02 became the same 2% result. Single-detector scores now retain
their reading; multi-detector fusion retains its protective clipping. This is a
reporting correction, not evidence of better accuracy. The frontend labels the
value as a model score and explains that low readings do not prove authenticity.

## Image upgrade

Default: `s-gehring/detectra-openfake`, revision
`ef971ed6ecca0205d8b001e9326727fb5c00e8a8`, using ONNX Runtime on CPU.
The model's published preprocessing is implemented explicitly: bilinear
short-edge resize 440, center crop 384, ImageNet normalization, single-logit sigmoid.
No unknown label order, remote Python execution or random head initialization.

[Model card and attribution](https://huggingface.co/s-gehring/detectra-openfake):
Detectra by TensorFold / Community Forensics by Jeongsoo Park and Andrew Owens,
fine-tuned by s-gehring using OpenFake and additional data. The weights are
**CC-BY-NC-4.0, non-commercial only**. The user confirmed college/personal use
before this became the default. Commercial deployments must choose another model
or obtain permission; `-Profile oct08` restores the prior model selection.

## Fresh diagnostic evaluation

Downloaded another 124 labelled files with seed 20261022. The downloader excludes
the earlier 124 files by row ID and checks for duplicate SHA256 content. Images
come from Mirage-Test; speech comes from the Gary Stafford corpus. Revisions,
labels, partitions and hashes are in `checkpoints/media-benchmark-v2/manifest.jsonl`.
Development and test files are disjoint. This still does not establish speaker-
or generator-disjointness, nor eliminate possible overlap with community models'
undisclosed training data. No thresholds were fitted to the new test results.

Development images, 32 files:
- Old Swin binary score classification at 0.5: 23/32.
- Detectra OpenFake: 31/32.
- At the existing app threshold 0.65: fake detections 10/16 → 13/16;
  false alarms among real images 3/16 → 0/16.

Reserved images, another 32 files:
- Binary score classification at 0.5: **25/32 (78.125%) → 27/32 (84.375%)**.
- At the app threshold: fake detections **11/16 → 11/16**; false alarms
  **2/16 → 0/16**. Thus the measured improvement here is fewer false alarms,
  not better fake recall. Five fake images still went undetected.
- Low image scores remain uncertain under the existing uncalibrated-authenticity
  guard. Binary score accuracy is not the same as actual app-verdict accuracy.
- On the earlier development images, Detectra also scored 27/32 versus Swin 25/32.

Reserved speech, 40 files:
- Current WavLM: **31/40 binary classifications correct (77.5%)**.
- Fake detections: **11/20**; real files falsely flagged: **0/20**.
- **9/20 synthetic clips were still called real.** These are new-sample results
  for an unchanged audio model, not an accuracy improvement over the last run.

## New audio candidate could not be enabled

Investigated [Pella v2](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector),
also listed in the [Podonos benchmark](https://github.com/podonos/audio-dfd-benchmark).
Those external scores are not this app's accuracy. Its published weights at
revision `84f7a0d0a22903220b416aed5e45169b6613e9d4` fail to load under both the
existing Torch runtime and an isolated matching Torch/TorchAudio 2.11 runtime.

The download SHA256 matches the published LFS value exactly:
`2279c24f6e4b672c85eb2593f93cbbad83a7c1e9486fe3bcdc246c13fcca80e6`.
Python's ZIP reader independently reports a bad local header for `best/data/451`
at byte offset 1178301200. This is a problem with that published artifact, not
evidence that Pella's predictions are good or bad. No missing weights were guessed,
no unsafe pickle loading was enabled, and this candidate is not serving requests.
The separate runtime under `checkpoints/pella-runtime` does not replace the app's
working Torch installation. Other modern image options were researched but not
locally validated and are not claimed as upgrades.

## Verification, scope and rollback

73 regression tests pass, including the single-model floor regression, ONNX
normalization and invalid-logit handling. Frontend production build passes.
Live image uploads through the frontend were checked against saved offline
predictions and the loaded model name. Video and text control-input results are
recorded in `checkpoints/new-candidates/live.json`; these are integration checks,
not a representative face-swap or news-claim accuracy benchmark. Video and text
models were not changed or assigned new accuracy claims.

Checkpoint `07-before-expanded-accuracy-review` contains the exact previous source.
Checkpoint `08-detectra-score-fix` contains this update. To revert model selection,
stop the backend and run `./scripts/start.ps1 -Profile oct08`. To restore source
without overwriting current work:

```powershell
.venv/Scripts/python.exe scripts/checkpoint.py restore 07-before-expanded-accuracy-review --destination checkpoints/restored-before-oct22
```

The user's actual failing uploads are not retained by the API and were unavailable
for this investigation. Generated imagery, face swaps, small edits, screenshots,
compressed copies and new generators are different failure cases. A strong result
on this small generated-image sample does not validate every one of them. Audio
still needs a usable modern checkpoint or training on a broader, separated corpus;
video still needs a labelled face-swap evaluation, and text a sourced claim corpus.

## Labelled video and claim diagnostics

A seeded 24-video sample from the pinned `gonnerthetooner/FaceForensics`
mirror (revision `f2980e9a2f43320503ab2a1ac1b65613621879c5`) includes
12 originals, six FaceSwap and six Deepfakes clips. Sampling used the first
100 files in each category, restricted to files below 8 MB. Downloads match
published LFS SHA256 hashes. This is a small convenience sample, not the official
identity-disjoint FF++ benchmark; training overlap cannot be ruled out.

On 12 development videos, dima scored 9/12 at a binary 0.5 cutoff versus
5/12 for SigLIP2. Neither flagged any fake with the app's stricter verdict rules.
SigLIP2 was rejected. On 12 reserved clips, unchanged dima scored 8/12 at 0.5,
flagged 3/6 fakes, falsely flagged 0/6 originals, and abstained on 8/12 clips.
These results expose poor coverage; they do not establish a video improvement.
Per-frame scores and manifests are in `checkpoints/video-news-benchmark/`.

The live Moon control exposed false contradictions from headlines about other
planets. The lexical relevance gate now requires at least two shared content
terms for multi-term claims, in addition to its existing overlap threshold.
Six regression cases check unrelated headlines, relevant support and genuine
contradictions. This conservative heuristic can miss paraphrases; it is not
semantic entity resolution or proof that news accuracy is fixed.

Twelve seeded balanced unique FEVER labelled-development claims were checked
through the live API, with all responses and failures retained in
`checkpoints/video-news-benchmark/news-baseline.json`. FEVER labels refer to an
older Wikipedia snapshot; current web evidence is different, and the NLI model
was trained on FEVER. These diagnostics must not be presented as an independent
current-news accuracy benchmark. Results: 10 Unverified, one correctly contradicted claim, and one short claim
rejected with HTTP 422. Four of the Unverified outcomes match FEVER NEI labels;
six are missed supported/refuted claims. Exact label agreement is 5/12 counting
the rejected input as a failure, but this is not a valid current-news accuracy
estimate for the reasons above. No new text-model accuracy is claimed.

The additional Eliya Forensics audio model's card and inference source were
inspected, but its weights have not been evaluated. It is not enabled and no
performance claim is made for it.
