"""Song-discovery endpoints: query-less browse feeds backed by Audius + Jamendo.

Reuses the search module's client singletons and ``interleave`` merge helper so
discovery results map into the exact same ``SongSearchResult`` shape — only the
"fetch a list" call differs from search.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.discover import genres
from src.search.router import audius_client, interleave, jamendo_client
from src.search.schemas import SongSearchResponse, SongSearchResult

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/discover/genres")
async def discover_genres() -> dict:
    """Canonical genre list (key + label) for the Discover chips."""
    return {"genres": genres.list_genres()}


@router.get("/discover", response_model=SongSearchResponse)
async def discover(
    feed: str = Query("trending", pattern="^(trending|underground)$"),
    genre: Optional[str] = Query(None, description="Canonical genre key"),
    limit: int = Query(20, ge=1, le=50),
) -> SongSearchResponse:
    """Browse tracks with no search query.

    ``feed=trending`` merges Audius trending + Jamendo popularity (optionally
    filtered by ``genre``). ``feed=underground`` surfaces Audius
    underground-trending only — Jamendo has no equivalent, so it's skipped. A
    single failing provider degrades to the other; only when every queried
    provider errors do we return 502 (mirrors ``search_songs``).
    """

    def _ok(res, provider: str) -> List[SongSearchResult]:
        if isinstance(res, Exception):
            logger.error(f"{provider} discover failed (feed={feed}): {res}")
            return []
        return res

    if feed == "underground":
        # Audius-only; Jamendo has no underground concept.
        try:
            results = await audius_client.trending(underground=True, limit=limit)
        except Exception as e:
            logger.error(f"Audius underground failed: {e}")
            raise HTTPException(status_code=502, detail="Discovery failed. Try again.")
        return SongSearchResponse(results=results[:limit])

    audius_res, jamendo_res = await asyncio.gather(
        audius_client.trending(genre=genres.audius_genre(genre), limit=limit),
        jamendo_client.trending(tag=genres.jamendo_tag(genre), limit=limit),
        return_exceptions=True,
    )

    if isinstance(audius_res, Exception) and isinstance(jamendo_res, Exception):
        raise HTTPException(status_code=502, detail="Discovery failed. Try again.")

    merged = interleave(
        _ok(audius_res, "Audius"),
        _ok(jamendo_res, "Jamendo"),
    )[:limit]
    return SongSearchResponse(results=merged)
