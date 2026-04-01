"""
TRIBE v2 Service
================
Integrates Meta's TRIBE v2 brain encoding model for real neural response prediction.

TRIBE v2 predicts fMRI brain responses to naturalistic stimuli (video, audio, text)
using state-of-the-art feature extractors (LLaMA 3.2, V-JEPA2, Wav2Vec-BERT).

Repository: https://github.com/facebookresearch/tribev2
Model: https://huggingface.co/facebook/tribev2

CRITICAL: This service uses the REAL TRIBE model.
NO simulation, NO placeholder, NO fallback logic.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace repo ID must use forward slashes (not Windows backslashes)
MODEL_ID: str = "facebook/tribev2"
TRIBE_CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "tribe" / "cache"

_model: Optional[object] = None


def initialize_tribe_model() -> None:
    """
    Initialize the TRIBE v2 model from HuggingFace.
    
    This loads the official facebook/tribev2 model weights and prepares
    the model for inference. Raises RuntimeError if model cannot be loaded.
    
    Requirements:
    - tribev2 package must be installed (run scripts/setup_tribe.py)
    - HuggingFace authentication for LLaMA 3.2 access
    - Sufficient memory (8GB+ RAM, 8GB+ VRAM recommended)
    """
    global _model

    try:
        from tribev2 import TribeModel
    except Exception as exc:
        raise RuntimeError(
            "TRIBE v2 package is not installed. Run scripts/setup_tribe.py before starting the API."
        ) from exc

    TRIBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ensure cache folder uses OS-native path but repo_id uses forward slashes
    cache_path = str(TRIBE_CACHE_DIR)
    repo_id = str(PurePosixPath(MODEL_ID))  # Guarantees forward slashes
    
    logger.info(f"Loading TRIBE v2 from '{repo_id}' (cache: {cache_path})")
    
    # Force CPU mode for all extractors since this system doesn't have CUDA-enabled PyTorch
    config_update = {
        "data.video_feature.image.device": "cpu",
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
    }
    
    try:
        _model = TribeModel.from_pretrained(
            repo_id,
            cache_folder=cache_path,
            device="cpu",
            config_update=config_update
        )
        logger.info("TRIBE v2 model loaded successfully")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TRIBE v2 model weights from '{repo_id}'. "
            "Ensure setup_tribe.py completed successfully and HuggingFace auth is configured."
        ) from exc


def get_brain_vector(frame: np.ndarray) -> np.ndarray:
    """
    Get brain activation vector from TRIBE v2 for a single video frame.
    
    TRIBE v2 predicts fMRI-like brain activity on the fsaverage5 cortical
    mesh (~20k vertices). This function creates a single-frame video and
    runs the official TRIBE inference pipeline.
    
    Args:
        frame: RGB image array with shape [H, W, C]
        
    Returns:
        1D numpy array of brain activations (flattened from TRIBE output)
        
    Raises:
        RuntimeError: If TRIBE model not initialized or inference fails
        ValueError: If frame has invalid shape
        
    Note: NO simulation or fallback. Real model inference only.
    """
    if _model is None:
        raise RuntimeError("TRIBE v2 model is not initialized")

    if frame.ndim != 3:
        raise ValueError("Expected frame with shape [H, W, C]")

    # TRIBE v2 exposes event-based inference on media files; we create a
    # one-frame temporary video and run the official predict pipeline.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_path = Path(temp_video.name)

    try:
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open temporary video writer for TRIBE inference")
        writer.write(bgr)
        writer.release()

        events = _model.get_events_dataframe(video_path=str(temp_path))
        preds, _segments = _model.predict(events, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"TRIBE v2 inference failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    vector = np.ravel(np.asarray(preds))
    if vector.size < 800:
        raise RuntimeError(f"TRIBE output has insufficient dimensionality: {vector.size}")
    return vector
