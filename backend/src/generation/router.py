"""
Generation Router
Handles step chart generation requests
"""

import os
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from ml import MLChartGenerator
from src.config import settings
from src.generation.schemas import GenerateRequest
from src.generation.service import download_audio
from src.generation.utils import AudioProcessor
from src.spotify.client import SpotifyClient
from src.storage import get_storage
from src.youtube.client import YouTubeClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level singletons. Initialized at import time so the ML checkpoint
# crashes the app at startup if it's missing/incompatible (rather than at
# first request).
audio_processor = AudioProcessor()
spotify_client = SpotifyClient()
youtube_client = YouTubeClient()
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
)
logger.info("ML chart generator loaded successfully")


def _charts_payload(charts: dict) -> dict:
    """Serialize a {difficulty: Chart} map to its JSON response shape."""
    return {name: chart.to_json_dict() for name, chart in charts.items()}


@router.post("/generate-steps")
async def generate_steps_from_file(
    file: UploadFile = File(...),
    style: str = Form("auto"),
):
    """
    Generate step charts for every difficulty from an uploaded audio file.

    Args:
        file: Audio file (MP3, WAV, OGG, FLAC)
        style: Style profile name (e.g. 'Stream-Heavy') or 'auto'
    """
    try:
        logger.info(f"Received file upload: {file.filename}")

        allowed_extensions = [".mp3", ".wav", ".ogg", ".flac"]
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        song_id = str(uuid.uuid4())
        audio_key = f"{song_id}{file_ext}"

        content = await file.read()

        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )

        file_path = storage.write_bytes(audio_key, content)
        logger.info(f"Saved file: {audio_key}")

        duration = audio_processor.get_duration(str(file_path))
        if duration > settings.MAX_DURATION_SECONDS:
            storage.delete(audio_key)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        logger.info(f"Generating charts for all difficulties (style={style})...")
        charts = ml_generator.generate_all_difficulties(str(file_path), style=style)

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-steps-url")
async def generate_steps_from_url(
    request: GenerateRequest,
):
    """
    Generate step charts for every difficulty from a URL (YouTube or Spotify).

    Args:
        request: URL + style
    """
    try:
        logger.info(f"Received URL: {request.url}")

        if not any(domain in request.url for domain in ["youtube.com", "youtu.be", "spotify.com"]):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL. Please use YouTube or Spotify links."
            )

        song_id = str(uuid.uuid4())
        audio_key = f"{song_id}.mp3"
        file_path = storage.reserve_path(audio_key)

        logger.info("Downloading audio...")
        download_result = await download_audio(
            request.url,
            str(file_path),
            spotify_client=spotify_client,
            youtube_client=youtube_client,
        )
        storage.commit(audio_key, file_path)

        metadata = download_result["metadata"]

        if metadata["duration"] > settings.MAX_DURATION_SECONDS:
            storage.delete(audio_key)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        logger.info(
            f"Generating charts for all difficulties (style={request.style})..."
        )
        charts = ml_generator.generate_all_difficulties(
            str(file_path), style=request.style,
        )
        any_chart = next(iter(charts.values()))

        response = {
            "song_id": song_id,
            "charts": _charts_payload(charts),
            "song_info": {
                "title": metadata["title"],
                "artist": metadata.get("artist", "Unknown"),
                "duration": any_chart.duration,
                "tempo": any_chart.tempo,
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
