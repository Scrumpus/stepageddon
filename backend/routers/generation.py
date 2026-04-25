"""
Generation Router
Handles step chart generation requests
"""

import os
import uuid
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from services import AudioProcessor, AudioDownloader
from services.chart_persistence import persist_user_song
from modules.step_generator import ChartGenerationPipeline, ChartExporter
from core.config import settings
from db import get_session

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
audio_processor = AudioProcessor()

audio_downloader = AudioDownloader()

# Initialize ML generator if enabled
ml_generator = None
if settings.USE_ML_GENERATION:
    try:
        from ml import MLChartGenerator
        logger.info(f"Loading ML generator from {settings.ML_MODEL_PATH}...")
        ml_generator = MLChartGenerator(model_path=settings.ML_MODEL_PATH)
        logger.info("ML chart generator loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load ML generator, falling back to algorithmic: {e}", exc_info=True)
else:
    logger.info("USE_ML_GENERATION is false; using algorithmic generator")


class GenerateRequest(BaseModel):
    """Request model for URL-based generation"""
    url: str = Field(..., description="YouTube or Spotify URL")
    difficulty: str = Field("medium", description="Difficulty level")


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
    session: AsyncSession = Depends(get_session),
):
    """
    Generate step chart from uploaded audio file

    Args:
        file: Audio file (MP3, WAV, OGG, FLAC)
        difficulty: beginner, easy, medium, hard, or challenge
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
        song_uuid = uuid.uuid4()
        song_id = str(song_uuid)

        # Save uploaded file
        file_path = os.path.join(
            settings.AUDIO_STORAGE_PATH,
            f"{song_id}{file_ext}"
        )

        content = await file.read()

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
            )

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved file: {file_path}")

        # Check duration
        duration = audio_processor.get_duration(file_path)
        if duration > settings.MAX_DURATION_SECONDS:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        # Generate steps (ML or algorithmic)
        logger.info(f"Generating {difficulty} steps...")
        if ml_generator is not None:
            chart = ml_generator.generate_from_audio(file_path, difficulty)
        else:
            chart = ChartGenerationPipeline.generate_from_audio(file_path, difficulty)
        steps = ChartExporter.to_json(chart)

        await persist_user_song(
            session,
            song_id=song_uuid,
            title=file.filename,
            artist=None,
            audio_relpath=f"{song_id}{file_ext}",
            chart=chart,
            difficulty_name=difficulty,
            difficulty_level=0,
            generator="ml" if ml_generator is not None else "algorithmic",
        )

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
            "audio_url": f"/api/audio/{song_id}{file_ext}"
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
    session: AsyncSession = Depends(get_session),
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
        song_uuid = uuid.uuid4()
        song_id = str(song_uuid)
        file_path = os.path.join(settings.AUDIO_STORAGE_PATH, f"{song_id}.mp3")

        # Download audio
        logger.info("Downloading audio...")
        download_result = await audio_downloader.download_from_url(
            request.url,
            file_path
        )

        metadata = download_result["metadata"]

        # Check duration
        if metadata["duration"] > settings.MAX_DURATION_SECONDS:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max duration: {settings.MAX_DURATION_SECONDS}s"
            )

        # Generate steps (ML or algorithmic)
        logger.info(f"Generating {request.difficulty} steps...")
        if ml_generator is not None:
            logger.info("ML Generator")
            chart = ml_generator.generate_from_audio(file_path, request.difficulty)
        else:
            logger.info("Algorithmic Generator")
            chart = ChartGenerationPipeline.generate_from_audio(file_path, request.difficulty)
        steps = ChartExporter.to_json(chart)

        await persist_user_song(
            session,
            song_id=song_uuid,
            title=metadata["title"],
            artist=metadata.get("artist") or None,
            audio_relpath=f"{song_id}.mp3",
            chart=chart,
            difficulty_name=request.difficulty,
            difficulty_level=0,
            generator="ml" if ml_generator is not None else "algorithmic",
        )

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
            "audio_url": f"/api/audio/{song_id}.mp3"
        }

        logger.info(f"✓ Generated {len(chart.steps)} steps for {song_id}")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
