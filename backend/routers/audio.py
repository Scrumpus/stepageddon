"""
Audio Router
Handles audio file streaming
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/audio/{filename}")
async def stream_audio(filename: str):
    """
    Stream audio file
    
    Args:
        filename: Audio filename (e.g., song_id.mp3)
    """
    try:
        file_path = os.path.join(settings.AUDIO_STORAGE_PATH, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Audio file not found: {filename}")
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Determine media type
        ext = os.path.splitext(filename)[1].lower()
        media_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac"
        }
        media_type = media_types.get(ext, "audio/mpeg")
        
        return FileResponse(
            file_path,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio streaming failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to stream audio")
