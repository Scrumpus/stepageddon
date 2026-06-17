"""
Generation Router
Handles step chart generation requests
"""

import os
import uuid
import logging
from urllib.parse import urlparse

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

from ml import MLChartGenerator
from src.audius.client import AudiusClient
from src.config import settings
from src.generation.schemas import GenerateRequest
from src.generation.service import download_audio
from src.generation.utils import AudioProcessor
from src.jamendo.client import JamendoClient
from src.rate_limit import limiter
from src.storage import get_storage

_ALLOWED_URL_HOSTS = ("audius.co", "jamendo.com")

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level singletons. Initialized at import time so the ML checkpoint
# crashes the app at startup if it's missing/incompatible (rather than at
# first request).
audio_processor = AudioProcessor()
audius_client = AudiusClient()
jamendo_client = JamendoClient()
storage = get_storage()

logger.info(f"Loading ML generator from {settings.ML_MODEL_PATH}...")
# Style profiles live next to the training data the checkpoint was built from.
# We probe a couple of conventional locations so the API doesn't hard-fail
# when profiles haven't been generated yet — `style='auto'` then falls back
# to default knobs (see MLChartGenerator._resolve_style).
_style_profiles_path = getattr(settings, 'STYLE_PROFILES_PATH', None)
if _style_profiles_path is None:
    for _candidate in (
        os.path.join(os.path.dirname(settings.ML_MODEL_PATH), 'style_profiles.json'),
        os.path.join('ml', 'checkpoints', 'style_profiles.json'),
    ):
        if os.path.exists(_candidate):
            _style_profiles_path = _candidate
            break
ml_generator = MLChartGenerator(
    model_path=settings.ML_MODEL_PATH,
    style_profiles_path=_style_profiles_path,
    timing_trim=settings.GENERATION_TIMING_OFFSET_MS / 1000.0,
)
logger.info("ML chart generator loaded successfully")


def _charts_payload(charts: dict) -> dict:
    """Serialize a {difficulty: Chart} map to its JSON response shape."""
    return {name: chart.to_json_dict() for name, chart in charts.items()}


@router.post("/generate-steps")
@limiter.limit("5/minute;30/hour")
async def generate_steps_from_file(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("auto"),
):
    """
    Generate step charts for every difficulty from an uploaded audio file.

    Args:
        file: Audio file (MP3, WAV, OGG, FLAC)
        style: Style profile name (e.g. 'Stream-Heavy') or 'auto'
    """
    logger.info(f"Received file upload: {file.filename}")

    allowed_extensions = [".mp3", ".wav", ".ogg", ".flac"]
    file_ext = os.path.splitext(file.filename or "")[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Reject oversized uploads before buffering them into memory.
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB",
                )
        except ValueError:
            pass  # malformed header — fall through to the read-time check

    song_id = str(uuid.uuid4())
    audio_key = f"{song_id}{file_ext}"

    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB",
        )

    file_path = storage.write_bytes(audio_key, content)
    logger.info(f"Saved file: {audio_key}")

    # From here on, any failure must clean up the stored file.
    try:
        duration = audio_processor.get_duration(str(file_path))
        if duration > settings.MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        logger.info(f"Generating charts for all difficulties (style={style})...")
        charts = ml_generator.generate_all_difficulties(str(file_path), style=style)
    except HTTPException:
        storage.delete(audio_key)
        raise
    except Exception as e:
        storage.delete(audio_key)
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")

    # All charts share the same tempo/duration (one audio analysis); pick
    # any one for song_info.
    any_chart = next(iter(charts.values()))

    response = {
        "song_id": song_id,
        "charts": _charts_payload(charts),
        "song_info": {
            "title": file.filename,
            "duration": any_chart.duration,
            "tempo": any_chart.tempo,
            "timing_offset": any_chart.timing_offset,
            "source": "upload"
        },
        "audio_url": f"/api/audio/{audio_key}"
    }

    total_steps = sum(len(c.steps) for c in charts.values())
    logger.info(
        f"✓ Generated {total_steps} steps across {len(charts)} difficulties "
        f"for {song_id}"
    )
    return JSONResponse(content=response)


def _validate_audio_url(url: str) -> None:
    """Reject anything that isn't a well-formed http(s) URL on an allow-listed host."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL.")
    host = parsed.netloc.lower().split(":")[0]
    # Match the bare host or any subdomain (e.g., www.jamendo.com).
    if not any(host == h or host.endswith("." + h) for h in _ALLOWED_URL_HOSTS):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Please use Audius or Jamendo links.",
        )


@router.post("/generate-steps-url")
@limiter.limit("5/minute;30/hour")
async def generate_steps_from_url(
    request: Request,
    body: GenerateRequest,
):
    """
    Generate step charts for every difficulty from a URL (YouTube or Spotify).
    """
    logger.info(f"Received URL: {body.url}")
    _validate_audio_url(body.url)

    song_id = str(uuid.uuid4())
    audio_key = f"{song_id}.mp3"
    file_path = storage.reserve_path(audio_key)

    committed = False
    try:
        logger.info("Downloading audio...")
        download_result = await download_audio(
            body.url,
            str(file_path),
            audius_client=audius_client,
            jamendo_client=jamendo_client,
        )
        storage.commit(audio_key, file_path)
        committed = True

        metadata = download_result["metadata"]

        if metadata["duration"] > settings.MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        logger.info(
            f"Generating charts for all difficulties (style={body.style})..."
        )
        charts = ml_generator.generate_all_difficulties(
            str(file_path), style=body.style,
        )
    except HTTPException:
        if committed:
            storage.delete(audio_key)
        raise
    except Exception as e:
        if committed:
            storage.delete(audio_key)
        logger.error(f"URL generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed")

    any_chart = next(iter(charts.values()))

    response = {
        "song_id": song_id,
        "charts": _charts_payload(charts),
        "song_info": {
            "title": metadata["title"],
            "artist": metadata.get("artist", "Unknown"),
            "duration": any_chart.duration,
            "tempo": any_chart.tempo,
            "timing_offset": any_chart.timing_offset,
            "thumbnail": metadata.get("thumbnail", ""),
            "source": metadata["source"],
            "is_preview": download_result.get("is_preview", False)
        },
        "audio_url": f"/api/audio/{audio_key}"
    }

    total_steps = sum(len(c.steps) for c in charts.values())
    logger.info(
        f"✓ Generated {total_steps} steps across {len(charts)} difficulties "
        f"for {song_id}"
    )
    return JSONResponse(content=response)
