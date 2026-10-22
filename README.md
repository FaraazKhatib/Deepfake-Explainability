# Unified Deepfake Detection Platform

A comprehensive, unified platform for detecting AI-generated content across text, image, audio, and video modalities.

## Project Structure
\\\
project/
├── frontend/                     # React/Vite unified interface
│   ├── src/components/           # Reusable UI components
│   ├── src/pages/                # Modality specific pages
│   └── src/App.jsx               # Main router & Sidebar
├── backend/                      # Unified FastAPI backend
│   ├── app/main.py               # Main API entrypoint
│   └── app/services/             # Isolated logic for each modality
│       ├── text/                 # Fact checker & text manipulation
│       ├── image_audio/          # Image and audio veritrust engine
│       └── video/                # Video deepfake analyzer
├── requirements.txt              # Consolidated dependencies
└── README.md
\\\

## Supported Detection Modes
- **Text**: Analyzes news claims against live sources (GDELT, Google News RSS, Wikipedia) and returns NLI-based verdicts.
- **Image**: Detects AI-generated or manipulated images using Vision Transformers.
- **Audio**: Detects voice cloning and synthetic speech.
- **Video**: Frame-by-frame analysis to detect face manipulations and deepfakes.

## How to Run

### 1. Backend
Ensure you have Python 3.11+ installed.
```bash
python -m pip install -r requirements.txt
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### 2. Frontend
Ensure you have Node.js installed.
```bash
cd frontend
npm install
npm run dev
```
Then open your browser to `http://localhost:5173`.

## Architecture Details

## Tested launch profiles and rollback

See [AUDIT.md](AUDIT.md) for findings, model trials, test evidence and remaining limitations.
From the project root on Windows, after installing requirements into `.venv`:

```powershell
./scripts/start.ps1 -Profile balanced
# Previous two-audio-model profile (source fixes retained):
./scripts/start.ps1 -Profile previous
# Alternative model trials (larger downloads; accuracy not established):
./scripts/start.ps1 -Profile candidate
# Original model selection, with correctness fixes retained:
./scripts/start.ps1 -Profile legacy
```

Balanced is also the default for a normal backend launch: Detectra OpenFake for images,
WavLM for audio, the original video classifier, and base DeBERTa NLI.
See [MEDIA_EVALUATION.md](MEDIA_EVALUATION.md) for the 124-file diagnostic study,
reserved-sample results, known misses and rollback. These are not universal accuracy figures.
The October 22 update uses a **non-commercial image model**, approved for this
college/personal project. See [ACCURACY_OCT22.md](ACCURACY_OCT22.md).
Use `./scripts/start.ps1 -Profile oct08` to restore the prior Swin/WavLM selection.

```powershell
.venv/Scripts/python.exe -m pip install pytest
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/checkpoint.py save my-checkpoint
.venv/Scripts/python.exe scripts/checkpoint.py restore 01-before-audit --destination restored-original
.venv/Scripts/python.exe scripts/evaluate.py labelled.jsonl --output checkpoints/evaluation.json
```

Evaluation JSONL uses `modality`, `path`, and `label` (`real`/`fake`) for media;
text uses `modality: text`, `text`, and `label` (`supported`/`refuted`/`disputed`/`unverified`).
Paths resolve relative to the manifest. Text evaluation uses live public retrieval.
Keep evaluation data separate from model-selection/calibration data.

Source checkpoints omit dependencies, uploaded media and model caches. Restore creates
a new directory and checks every file hash; it never overwrites current work.
The codebase was refactored to share a common modern React frontend while preserving the precise ML logic and preprocessing of the original individual modalities. The API uses FastAPI with modular routers for \/text\, \/image_audio\, and \/video\.
