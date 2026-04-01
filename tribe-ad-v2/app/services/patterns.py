from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

SIGNAL_COLUMNS = [
    "saliency",
    "motion",
    "novelty",
    "emotion_intensity",
    "emotion_valence",
    "relatability",
    "distinctiveness",
    "repetition",
    "story_coherence",
    "cognitive_load",
    "clarity",
    "info_density",
]

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "top_patterns.csv"

_kmeans: KMeans | None = None


def initialize_pattern_engine() -> None:
    global _kmeans

    if not DATA_PATH.exists():
        raise RuntimeError(f"Pattern dataset missing: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = set(SIGNAL_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(f"Pattern dataset missing columns: {sorted(missing)}")

    X = df[SIGNAL_COLUMNS].astype(float).values
    _kmeans = KMeans(n_clusters=6, random_state=42, n_init=20)
    _kmeans.fit(X)


def predict_cluster(signals: dict[str, float]) -> int:
    if _kmeans is None:
        raise RuntimeError("Pattern engine not initialized")

    row = [[signals[col] for col in SIGNAL_COLUMNS]]
    return int(_kmeans.predict(row)[0])
