from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

import cv2
import numpy as np
from fastapi import UploadFile

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


def allowed_video_extension(filename: str | None) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


async def save_upload_to_tempfile(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file extension")

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            temp.write(chunk)
        return Path(temp.name)


def stream_sampled_frames(video_path: Path, sample_fps: int = 1) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    frame_interval = max(int(round(source_fps / sample_fps)), 1)

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield rgb
            frame_idx += 1
    finally:
        capture.release()
