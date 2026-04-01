from __future__ import annotations

from typing import Sequence

import numpy as np

SignalDict = dict[str, float]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _adjacent_similarities(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([0.0], dtype=float)
    return np.array([
        _cosine_similarity(matrix[i], matrix[i + 1])
        for i in range(matrix.shape[0] - 1)
    ])


def _normalize(value: float, low: float, high: float) -> float:
    clipped = min(max(value, low), high)
    scaled = (clipped - low) / (high - low)
    return round(10.0 * scaled, 4)


def _inverse_entropy(values: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(values, bins=bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    max_entropy = np.log(bins)
    return float(1.0 - (entropy / max_entropy))


def compute_signals(brain_vectors: Sequence[np.ndarray]) -> SignalDict:
    if not brain_vectors:
        raise ValueError("No brain vectors provided")

    matrix = np.vstack(brain_vectors)
    if matrix.shape[1] < 800:
        raise ValueError("Brain vectors must have at least 800 dimensions")

    adjacent_sim = _adjacent_similarities(matrix)
    repetition_raw = float(np.mean(adjacent_sim))
    novelty_raw = float(1.0 - repetition_raw)
    temporal_drift = float(np.mean(np.linalg.norm(np.diff(matrix, axis=0), axis=1))) if len(matrix) > 1 else 0.0
    coherence_raw = float(1.0 / (1.0 + temporal_drift))

    raw = {
        "saliency": float(np.mean(matrix[:, 0:100])),
        "motion": float(np.std(matrix[:, 100:200])),
        "novelty": novelty_raw,
        "emotion_intensity": float(np.mean(matrix[:, 200:300])),
        "emotion_valence": float(np.mean(matrix[:, 300:350])),
        "relatability": float(np.mean(matrix[:, 350:400])),
        "distinctiveness": float(np.var(matrix[:, 400:500])),
        "repetition": repetition_raw,
        "story_coherence": coherence_raw,
        "cognitive_load": float(np.mean(matrix[:, 500:650])),
        "clarity": _inverse_entropy(matrix[:, 0:800].ravel()),
        "info_density": float(np.mean(matrix[:, 650:800])),
    }

    return {
        "saliency": _normalize(raw["saliency"], -3.0, 3.0),
        "motion": _normalize(raw["motion"], 0.0, 3.0),
        "novelty": _normalize(raw["novelty"], -1.0, 1.5),
        "emotion_intensity": _normalize(raw["emotion_intensity"], -3.0, 3.0),
        "emotion_valence": _normalize(raw["emotion_valence"], -3.0, 3.0),
        "relatability": _normalize(raw["relatability"], -3.0, 3.0),
        "distinctiveness": _normalize(raw["distinctiveness"], 0.0, 5.0),
        "repetition": _normalize(raw["repetition"], -1.0, 1.0),
        "story_coherence": _normalize(raw["story_coherence"], 0.0, 1.0),
        "cognitive_load": _normalize(raw["cognitive_load"], -3.0, 3.0),
        "clarity": _normalize(raw["clarity"], 0.0, 1.0),
        "info_density": _normalize(raw["info_density"], -3.0, 3.0),
    }
