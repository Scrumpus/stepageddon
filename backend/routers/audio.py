"""
Audio Router
Handles audio + image asset streaming. Supports nested paths under
AUDIO_STORAGE_PATH (used by DDR-seeded songs in `audio_storage/ddr/<slug>/...`).
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@router.get("/audio/{relpath:path}")
async def stream_audio(relpath: str):
    """Stream audio or image asset from AUDIO_STORAGE_PATH.

    Accepts nested paths (e.g. ``ddr/butterfly/Butterfly.ogg``). Resolves and
    rejects any path that escapes the storage root.
    """
    storage_root = Path(settings.AUDIO_STORAGE_PATH).resolve()
    try:
        target = (storage_root / relpath).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")

    # Reject path traversal
    if storage_root != target and storage_root not in target.parents:
        logger.warning("Rejected traversal attempt: %s", relpath)
        raise HTTPException(status_code=403, detail="Forbidden")

    if not target.is_file():
        logger.warning("Asset not found: %s", relpath)
        raise HTTPException(status_code=404, detail="Asset not found")

    media_type = MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(
        str(target),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )
