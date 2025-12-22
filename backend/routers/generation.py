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
from typing import Optional

from services import AudioProcessor, StepGenerator, AudioDownloader
from services.step_generator_new import ChartGenerationPipeline, ChartExporter
from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
audio_processor = AudioProcessor()

# Step generator - set use_ai=False for pure algorithmic, True for AI
# You can also make this configurable via environment variable
USE_AI_GENERATION = os.getenv("USE_AI_GENERATION", "false").lower() == "true"
step_generator = StepGenerator(use_ai=USE_AI_GENERATION)

audio_downloader = AudioDownloader()

logger.info(f"Step generation mode: {'AI + Algorithmic' if USE_AI_GENERATION else 'Pure Algorithmic'}")


class GenerateRequest(BaseModel):
    """Request model for URL-based generation"""
    url: str = Field(..., description="YouTube or Spotify URL")
    difficulty: str = Field("intermediate", description="Difficulty level")


class GenerateResponse(BaseModel):
    """Response model for step generation"""
    song_id: str
    steps: list
    song_info: dict
    audio_url: str


@router.post("/generate-steps")
async def generate_steps_from_file(
    file: UploadFile = File(...),
    difficulty: str = Form("intermediate")
):
    """
    Generate step chart from uploaded audio file
    
    Args:
        file: Audio file (MP3, WAV, OGG, FLAC)
        difficulty: beginner, intermediate, or expert
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
        
        # Analyze audio
        logger.info("Analyzing audio...")
        analysis = audio_processor.analyze_audio(file_path)
        
        # Generate steps
        logger.info(f"Generating {difficulty} steps...")
        steps = await step_generator.generate_steps(
            analysis,
            difficulty,
            song_info={"title": file.filename}
        )

        new_steps = ChartGenerationPipeline.generate_from_audio(file_path, difficulty)
        new_steps_json = ChartExporter.to_json(new_steps)

        # New generator
        
        # Prepare response
        response = {
            "song_id": song_id,
            "steps": steps,
            "new_steps_json": new_steps_json,
            "song_info": {
                "title": file.filename,
                "duration": analysis["duration"],
                "tempo": analysis["tempo"],
                "source": "upload"
            },
            "audio_url": f"/api/audio/{song_id}{file_ext}"
        }
        
        logger.info(f"✓ Generated {len(steps)} steps for {song_id}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-steps-url")
async def generate_steps_from_url(request: GenerateRequest):
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
        
        # Analyze audio
        logger.info("Analyzing audio...")
        analysis = audio_processor.analyze_audio(file_path)
        
        # Generate steps
        logger.info(f"Generating {request.difficulty} steps...")
        steps = await step_generator.generate_steps(
            analysis,
            request.difficulty,
            song_info=metadata
        )

        new_steps = ChartGenerationPipeline.generate_from_audio(file_path, request.difficulty)
        new_steps_json = ChartExporter.to_json(new_steps)
        
        # Prepare response
        response = {
            "song_id": song_id,
            "steps": steps,
            "new_steps_json": new_steps_json,
            "song_info": {
                "title": metadata["title"],
                "artist": metadata.get("artist", "Unknown"),
                "duration": analysis["duration"],
                "tempo": analysis["tempo"],
                "thumbnail": metadata.get("thumbnail", ""),
                "source": metadata["source"],
                "is_preview": download_result.get("is_preview", False)
            },
            "audio_url": f"/api/audio/{song_id}.mp3"
        }
        
        logger.info(f"✓ Generated {len(steps)} steps for {song_id}")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
