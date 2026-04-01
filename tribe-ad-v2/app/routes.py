import logging
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.insights import generate_insights
from app.services.patterns import predict_cluster
from app.services.signals import compute_signals
from app.services.tribe import get_brain_vector
from app.services.video import allowed_video_extension, save_upload_to_tempfile, stream_sampled_frames

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)) -> dict:
    if not allowed_video_extension(file.filename):
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov files are supported")

    started = perf_counter()
    try:
        temp_path = await save_upload_to_tempfile(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        brain_vectors = []
        frame_count = 0
        try:
            for frame in stream_sampled_frames(temp_path, sample_fps=1):
                brain_vectors.append(get_brain_vector(frame))
                frame_count += 1
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("TRIBE inference error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not brain_vectors:
            raise HTTPException(status_code=400, detail="No frames could be extracted from the video")

        signals = compute_signals(brain_vectors)
        cluster = predict_cluster(signals)
        insights = generate_insights(signals)

        elapsed = perf_counter() - started
        logger.info("analyze completed: frames=%s elapsed=%.3fs cluster=%s", frame_count, elapsed, cluster)

        return {
            "signals": signals,
            "cluster": cluster,
            "insights": insights,
            "timing_seconds": round(elapsed, 4),
            "overall_score": round(sum(signals.values()) / len(signals), 3),
        }
    finally:
        temp_path.unlink(missing_ok=True)
