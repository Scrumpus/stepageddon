"""
Generation Router
Handles step chart generation requests
"""

import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services import AudioProcessor, AudioDownloader, get_storage
from ml import MLChartGenerator
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
audio_processor = AudioProcessor()

audio_downloader = AudioDownloader()

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
        os.path.join('ml', 'training_data', 'style_profiles.json'),
    ):
        if os.path.exists(_candidate):
            _style_profiles_path = _candidate
            break
ml_generator = MLChartGenerator(
    model_path=settings.ML_MODEL_PATH,
    style_profiles_path=_style_profiles_path,
)
logger.info("ML chart generator loaded successfully")


class GenerateRequest(BaseModel):
    """Request model for URL-based generation"""
    url: str = Field(..., description="YouTube or Spotify URL")
    difficulty: str = Field("medium", description="Difficulty level")
    style: str = Field(
        "auto",
        description="Style profile name (e.g. 'Stream-Heavy', 'Jump-Heavy') or 'auto'",
    )


class GenerateResponse(BaseModel):
    """Response model for step generation"""
    song_id: str
    steps: list
    song_info: dict
    audio_url: str


@router.post("/generate-steps")
async def generate_steps_from_file(
    file: UploadFile = File(...),
    difficulty: str = Form("medium"),
    style: str = Form("auto"),
):
    """
    Generate step chart from uploaded audio file

    Args:
        file: Audio file (MP3, WAV, OGG, FLAC)
        difficulty: beginner, easy, medium, hard, or challenge
        style: Style profile name (e.g. 'Stream-Heavy') or 'auto'
    """
    try:
        logger.info(f"Received file upload: {file.filename}, difficulty: {difficulty}")

        # Validate file type
        allowed_extensions = [".mp3", ".wav", ".ogg", ".flac"]
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Generate unique ID
        song_id = str(uuid.uuid4())
        audio_key = f"{song_id}{file_ext}"

        content = await file.read()

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )

        file_path = storage.write_bytes(audio_key, content)
        logger.info(f"Saved file: {audio_key}")

        # Check duration
        duration = audio_processor.get_duration(str(file_path))
        if duration > settings.MAX_DURATION_SECONDS:
            storage.delete(audio_key)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        logger.info(f"Generating {difficulty} steps (style={style})...")
        chart = ml_generator.generate_from_audio(str(file_path), difficulty, style=style)
        steps = chart.to_json_dict()

        # Prepare response
        response = {
            "song_id": song_id,
            "steps": steps,
            "song_info": {
                "title": file.filename,
                "duration": chart.duration,
                "tempo": chart.tempo,
                "source": "upload"
            },
            "audio_url": f"/api/audio/{audio_key}"
        }

        logger.info(f"✓ Generated {len(chart.steps)} steps for {song_id}")
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
    Generate step chart from URL (YouTube or Spotify)

    Args:
        request: URL and difficulty
    """
    try:
        logger.info(f"Received URL: {request.url}, difficulty: {request.difficulty}")

        # Validate URL format
        if not any(domain in request.url for domain in ["youtube.com", "youtu.be", "spotify.com"]):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL. Please use YouTube or Spotify links."
            )

        # Generate unique ID
        song_id = str(uuid.uuid4())
        audio_key = f"{song_id}.mp3"
        file_path = storage.reserve_path(audio_key)

        # Download audio
        logger.info("Downloading audio...")
        download_result = await audio_downloader.download_from_url(
            request.url,
            str(file_path),
        )
        storage.commit(audio_key, file_path)

        metadata = download_result["metadata"]

        # Check duration
        if metadata["duration"] > settings.MAX_DURATION_SECONDS:
            storage.delete(audio_key)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        logger.info(
            f"Generating {request.difficulty} steps (style={request.style})..."
        )
        chart = ml_generator.generate_from_audio(
            str(file_path), request.difficulty, style=request.style,
        )
        steps = chart.to_json_dict()

        # Prepare response
        response = {
            "song_id": song_id,
            "steps": steps["steps"],
            "song_info": {
                "title": metadata["title"],
                "artist": metadata.get("artist", "Unknown"),
                "duration": chart.duration,
                "tempo": chart.tempo,
                "thumbnail": metadata.get("thumbnail", ""),
                "source": metadata["source"],
                "is_preview": download_result.get("is_preview", False)
            },
            "audio_url": f"/api/audio/{audio_key}"
        }

        logger.info(f"✓ Generated {len(chart.steps)} steps for {song_id}")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
