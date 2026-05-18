"""Song-search endpoint backed by YouTube Music."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from src.ytmusic.client import YTMusicClient
from src.ytmusic.schemas import SongSearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level singleton — matches the pattern used by SpotifyClient /
# YouTubeClient in generation/router.py. ytmusicapi's HTTP session is reused
# across requests.
ytmusic_client = YTMusicClient()


@router.get("/search-songs", response_model=SongSearchResponse)
async def search_songs(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    limit: int = Query(10, ge=1, le=25, description="Max results to return"),
) -> SongSearchResponse:
    """Search YouTube Music for songs matching `q`.

    Returns up to `limit` results. ytmusicapi is unofficial and occasionally
    breaks when YouTube changes its internal API — exceptions surface as 502
    so the frontend can fall back to URL paste.
    """
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        # ytmusicapi is sync/blocking — push it off the event loop.
        results = await asyncio.to_thread(ytmusic_client.search, query, limit)
    except RuntimeError as e:
        # Client failed to initialize at startup.
        logger.error(f"YT Music client unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail="Song search is unavailable. Paste a link instead.",
        )
    except Exception as e:
        logger.error(f"YT Music search failed for q={query!r}: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Song search failed. Try again or paste a link.",
        )

    return SongSearchResponse(results=results)
