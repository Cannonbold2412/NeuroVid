# Copilot Instructions for NeuroAD V2

## Build & Run

```bash
# Setup (first time)
cd tribe-ad-v2
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup_tribe.py   # clones tribev2 repo + downloads model weights (~3GB)
huggingface-cli login           # required for LLaMA 3.2 access

# Start server
uvicorn app.main:app --reload   # http://127.0.0.1:8000
```

## Architecture

**Pipeline**: Video → Frames (1 FPS) → TRIBE v2 brain vectors → 12 cognitive signals → KMeans cluster → insights

```
app/
├── main.py            # FastAPI lifespan: initializes tribe model + pattern engine
├── routes.py          # POST /analyze endpoint - orchestrates the pipeline
└── services/
    ├── video.py       # OpenCV frame extraction (stream_sampled_frames)
    ├── tribe.py       # TRIBE v2 inference (get_brain_vector) - creates temp video per frame
    ├── signals.py     # Maps 20k brain dimensions → 12 signals using specific index ranges
    ├── patterns.py    # KMeans (6 clusters) trained on data/top_patterns.csv
    └── insights.py    # Threshold-based recommendations (SIGNAL_RULES dict)
```

**Critical services loaded at startup** (see `main.py` lifespan):
1. Pattern engine (KMeans from CSV)
2. TRIBE v2 model (HuggingFace `facebook/tribev2`)

## Key Conventions

### Fail-Fast Policy
No simulation, no placeholder, no fallback. If TRIBE v2 fails to load or inference errors occur, the system raises `RuntimeError` immediately. Check `tribe.py` for the pattern.

### Brain Vector → Signal Mapping
The 12 cognitive signals are derived from specific index ranges of the ~20k TRIBE output:
- `saliency`: mean of indices 0-100
- `motion`: std of indices 100-200
- `emotion_intensity`: mean of indices 200-300
- etc. (see `signals.py` for full mapping)

### Signal Normalization
All signals are scaled to 0-10 using `_normalize(value, low, high)` with domain-specific bounds defined in `signals.py`.

### Pattern Clusters
6 KMeans clusters are used to classify videos. Cluster definitions are inferred from `data/top_patterns.csv` (15 reference patterns).

### Video Processing
- Supported formats: `.mp4`, `.mov` only (see `ALLOWED_EXTENSIONS` in `video.py`)
- Frame sampling: 1 FPS default (configurable via `sample_fps` parameter)
- TRIBE inference requires creating a temp single-frame video per frame

## API

Single endpoint: `POST /analyze` accepts video file upload, returns:
```json
{
  "signals": { /* 12 cognitive signals, 0-10 scale */ },
  "cluster": 0-5,
  "insights": ["actionable recommendation..."],
  "timing_seconds": 1.48,
  "overall_score": 5.99
}
```

## Dependencies

- Python 3.10+ required (TRIBE v2 requirement)
- PyTorch 2.5+, CUDA 12.x optional but recommended
- `tribev2` package installed in editable mode from `models/tribe/tribev2-src/`
- 8GB+ RAM minimum, 16GB recommended
