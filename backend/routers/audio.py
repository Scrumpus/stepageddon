"""
Audio Router
Streams audio + image assets via the configured storage backend. Supports
nested keys (e.g. ``ddr/butterfly/Butterfly.ogg``).
"""

import logging

from fastapi import APIRouter

from services import get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/audio/{relpath:path}")
async def stream_audio(relpath: str):
    """Stream an asset stored under ``relpath`` (the storage key)."""
    return get_storage().serve(relpath)
