# 🧠 NeuroAd Optimizer V2

Production-ready FastAPI backend for video cognitive analysis using **real TRIBE v2 inference** from Meta/Facebook Research.

## 🎯 What It Does

1. **Accepts video upload** (`.mp4`, `.mov`)
2. **Extracts frames** at 1 FPS using OpenCV
3. **Runs real TRIBE v2 model** - predicts fMRI brain responses to visual stimuli
4. **Converts brain vectors** to 12 core cognitive signals
5. **Compares with patterns** using KMeans clustering
6. **Returns actionable insights** for ad optimization

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Video      │───▶│   TRIBE v2   │───▶│   Signal     │
│   Service    │    │   Service    │    │   Mapper     │
│ (OpenCV 1fps)│    │ (Brain Vec)  │    │ (12 signals) │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
                    ┌──────────────┐           │
                    │   Insight    │◀──────────┤
                    │   Engine     │           │
                    └──────────────┘           ▼
                           │           ┌──────────────┐
                           └──────────▶│   Pattern    │
                                       │   Engine     │
                                       │ (KMeans 6cl) │
                                       └──────────────┘
```

## 📁 Project Structure

```text
tribe-ad-v2/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── routes.py            # POST /analyze endpoint
│   └── services/
│       ├── video.py         # Frame extraction (OpenCV)
│       ├── tribe.py         # TRIBE v2 inference (REAL model)
│       ├── signals.py       # Brain vector → 12 signals
│       ├── patterns.py      # KMeans pattern matching
│       └── insights.py      # Actionable recommendations
├── models/
│   └── tribe/
│       ├── tribev2-src/     # Cloned TRIBE repository
│       └── cache/           # HuggingFace model weights
├── data/
│   └── top_patterns.csv     # Reference patterns (15 rows)
├── scripts/
│   └── setup_tribe.py       # One-command TRIBE setup
├── requirements.txt
└── README.md
```

## 🚀 Quick Setup

### Prerequisites

- **Python 3.10+** (required by TRIBE v2)
- **Git** (for cloning TRIBE repo)
- **8GB+ RAM** (16GB recommended)
- **GPU** (optional, recommended for speed)

### Step 1: Create Virtual Environment

```bash
cd tribe-ad-v2
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup TRIBE v2

```bash
python scripts/setup_tribe.py
```

This script will:
- Clone `https://github.com/facebookresearch/tribev2.git` into `models/tribe/tribev2-src`
- Install TRIBE v2 package in editable mode
- Download `facebook/tribev2` model weights (~3GB) into `models/tribe/cache`

### Step 4: HuggingFace Authentication (for LLaMA 3.2)

TRIBE v2 uses Meta's LLaMA 3.2-3B which requires HuggingFace authentication:

```bash
# Install CLI if needed
pip install huggingface-hub

# Login with your token
huggingface-cli login
```

Get your token at: https://huggingface.co/settings/tokens

### Step 5: Run Server

```bash
uvicorn app.main:app --reload
```

Server starts at `http://127.0.0.1:8000`

## 📡 API Reference

### POST /analyze

Upload a video for cognitive analysis.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_ad.mp4"
```

**Response:**
```json
{
  "signals": {
    "saliency": 6.8,
    "motion": 5.9,
    "novelty": 6.1,
    "emotion_intensity": 6.7,
    "emotion_valence": 5.8,
    "relatability": 6.0,
    "distinctiveness": 5.4,
    "repetition": 5.9,
    "story_coherence": 6.3,
    "cognitive_load": 4.8,
    "clarity": 6.5,
    "info_density": 5.7
  },
  "cluster": 2,
  "insights": [
    "Low distinctiveness -> creative appears generic; introduce a unique visual motif.",
    "Low clarity -> sharpen value proposition and make the core message explicit earlier."
  ],
  "timing_seconds": 1.4821,
  "overall_score": 5.991
}
```

## 🧬 12 Cognitive Signals (0-10 Scale)

| Category | Signal | Description |
|----------|--------|-------------|
| **Attention** | `saliency` | Visual attention strength |
| | `motion` | Dynamic movement detection |
| | `novelty` | Uniqueness vs. prior frames |
| **Emotion** | `emotion_intensity` | Emotional activation level |
| | `emotion_valence` | Positive vs. negative tone |
| | `relatability` | Audience connection |
| **Memory** | `distinctiveness` | Memorability potential |
| | `repetition` | Pattern consistency |
| | `story_coherence` | Narrative flow |
| **Cognitive** | `cognitive_load` | Processing complexity |
| | `clarity` | Message sharpness |
| | `info_density` | Information concentration |

## 🔬 Signal Mapping (Brain Vector → Signals)

TRIBE v2 outputs ~20k vertex predictions on fsaverage5 cortical mesh. Signals are derived as:

```python
# Attention signals
saliency = mean(brain[0:100])
motion = std(brain[100:200])
novelty = 1 - similarity(frame_n, frame_n+1)

# Emotion signals
emotion_intensity = mean(brain[200:300])
emotion_valence = mean(brain[300:350])
relatability = mean(brain[350:400])

# Memory signals
distinctiveness = variance(brain[400:500])
repetition = similarity across frames
story_coherence = temporal consistency

# Cognitive signals
cognitive_load = mean(brain[500:650])
clarity = inverse_entropy(brain[0:800])
info_density = mean(brain[650:800])
```

## 🎨 Pattern Clusters

Videos are classified into 6 clusters based on KMeans analysis of top-performing patterns:

| Cluster | Profile |
|---------|---------|
| 0 | High-energy, emotional content |
| 1 | Balanced professional style |
| 2 | Story-driven narrative |
| 3 | Information-heavy explainer |
| 4 | Premium high-impact |
| 5 | Simple low-complexity |

## ⚠️ TRIBE v2 Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | - | 8 GB (CUDA) |
| Storage | 10 GB | 20 GB |

### Software

- Python 3.10+ (3.11 recommended)
- PyTorch 2.5+
- CUDA 12.x (for GPU acceleration)

### Model Access

TRIBE v2 uses gated models requiring approval:
- **LLaMA 3.2-3B**: https://huggingface.co/meta-llama/Llama-3.2-3B
- **TRIBE v2 Weights**: https://huggingface.co/facebook/tribev2

## 🛡️ Fail-Fast Policy

**No simulation. No placeholder. No fallback.**

If TRIBE v2 model cannot be loaded or inference fails:
- Startup raises `RuntimeError` with explicit message
- API returns `500` with detailed error
- System refuses to operate with fake data

This ensures production integrity.

## 🐛 Troubleshooting

### "TRIBE v2 package is not installed"
```bash
python scripts/setup_tribe.py
```

### "Failed to load TRIBE v2 model weights"
```bash
# Check HuggingFace login
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

### "CUDA out of memory"
Set environment variable to use CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/macOS
set CUDA_VISIBLE_DEVICES=        # Windows
```

### "No frames could be extracted"
- Verify video is valid MP4/MOV
- Check video duration > 1 second
- Test with `ffprobe your_video.mp4`

## 📚 References

- **TRIBE v2 Paper**: [A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/)
- **TRIBE v2 Repository**: https://github.com/facebookresearch/tribev2
- **Model Weights**: https://huggingface.co/facebook/tribev2

## 📄 License

This project integrates TRIBE v2 which is licensed under **CC-BY-NC-4.0** (non-commercial use only).

