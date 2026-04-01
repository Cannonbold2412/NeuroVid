# 🎬 How to Run NeuroAD V2: Video → 12 Core Cognitive Signals

## 🎯 What This Does
This project takes a **video file** as input and returns **12 cognitive signals** (0-10 scale) by:
1. Extracting video frames at 1 FPS 
2. Running TRIBE v2 brain model to predict neural responses
3. Converting 20k brain vector dimensions into 12 meaningful cognitive metrics
4. Providing actionable insights for ad optimization

## 🚀 Complete Setup Guide

### Step 1: Navigate to Project Directory
```bash
cd c:\Users\Lenovo\Desktop\NeuroAD\tribe-ad-v2
```

### Step 2: Create Virtual Environment (if not exists)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup TRIBE v2 Model
```bash
python scripts\setup_tribe.py
```

This will:
- Clone TRIBE v2 repository into `models\tribe\tribev2-src`
- Install TRIBE v2 package 
- Download model weights (~3GB) from HuggingFace

### Step 5: Start the FastAPI Server
```bash
uvicorn app.main:app --reload
```

Server will start at: `http://127.0.0.1:8000`

## 📹 How to Process a Video

### Method 1: Using the API (Recommended)

**Test with the included sample video:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\Users\Lenovo\Desktop\NeuroAD\tribe-ad-v2\test_video.mp4"
```

**With your own video:**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@your_video.mp4"
```

### Method 2: Using Python Script

```python
import requests

# Upload video
url = "http://127.0.0.1:8000/analyze"
files = {"file": open("your_video.mp4", "rb")}
response = requests.post(url, files=files)

# Get 12 cognitive signals
result = response.json()
print("🧠 Cognitive Signals:")
for signal, value in result["signals"].items():
    print(f"  {signal}: {value}/10")

print(f"\n📊 Overall Score: {result['overall_score']}/10")
print(f"🎯 Pattern Cluster: {result['cluster']}")
print(f"💡 Insights: {result['insights']}")
```

## 🧬 The 12 Cognitive Signals Explained

| **Category** | **Signal** | **What It Measures** |
|--------------|------------|---------------------|
| **Attention** | `saliency` | Visual attention strength |
| | `motion` | Dynamic movement detection |
| | `novelty` | Uniqueness vs. previous content |
| **Emotion** | `emotion_intensity` | Emotional activation level |
| | `emotion_valence` | Positive vs. negative emotion |
| | `relatability` | Audience connection potential |
| **Memory** | `distinctiveness` | Memorability potential |
| | `repetition` | Pattern consistency |
| | `story_coherence` | Narrative flow quality |
| **Cognitive** | `cognitive_load` | Mental processing complexity |
| | `clarity` | Message sharpness |
| | `info_density` | Information concentration |

## 📊 Expected Output Format

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

## 🔬 How It Works (Technical)

1. **Frame Extraction**: OpenCV extracts frames at 1 FPS
2. **TRIBE v2 Processing**: Real Meta/Facebook TRIBE v2 model predicts brain responses
3. **Signal Mapping**: 20k brain vector dimensions mapped to 12 signals using specific ranges:
   ```python
   saliency = mean(brain_vector[0:100])
   motion = std(brain_vector[100:200]) 
   emotion_intensity = mean(brain_vector[200:300])
   # ... and so on
   ```
4. **Pattern Classification**: KMeans clustering assigns video to 1 of 6 performance patterns
5. **Insight Generation**: Actionable recommendations based on signal strengths

## 🛠️ Testing

You can test immediately with the included `test_video.mp4`:

```bash
# Start server
uvicorn app.main:app --reload

# In another terminal
curl -X POST "http://127.0.0.1:8000/analyze" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@test_video.mp4"
```

## ⚠️ Important Notes

- **No Simulation**: This uses the real TRIBE v2 model, no fake data
- **Hardware**: 8GB+ RAM recommended, GPU optional but helpful
- **Video Formats**: Supports MP4, MOV, AVI 
- **Processing Time**: ~1-3 seconds per video depending on length and hardware

## 🐛 Troubleshooting

**TRIBE v2 not found:**
```bash
python scripts\setup_tribe.py
```

**HuggingFace authentication needed:**
```bash
pip install huggingface-hub
huggingface-cli login
```

**CUDA memory issues:**
```bash
set CUDA_VISIBLE_DEVICES=
# Then restart server
```

---

🎉 **Ready to analyze videos and get 12 cognitive signals!**