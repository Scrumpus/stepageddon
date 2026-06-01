"""Song-search endpoint backed by Audius + Jamendo (merged)."""

from __future__ import annotations

import asyncio
import logging
from itertools import zip_longest
from typing import List

from fastapi import APIRouter, HTTPException, Query

from src.audius.client import AudiusClient
from src.jamendo.client import JamendoClient
from src.search.schemas import SongSearchResponse, SongSearchResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level singletons. The clients hold only config (no shared HTTP
# session — each call opens its own aiohttp session), so they're safe to reuse
# across requests and across the generation router.
audius_client = AudiusClient()
jamendo_client = JamendoClient()


def interleave(*lists: List[SongSearchResult]) -> List[SongSearchResult]:
    """Round-robin merge so neither provider dominates the top of the list.

    Shared with the discovery router so browse feeds merge identically to search.
    """
    merged: List[SongSearchResult] = []
    for row in zip_longest(*lists):
        merged.extend(item for item in row if item is not None)
    return merged


@router.get("/search-songs", response_model=SongSearchResponse)
async def search_songs(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    limit: int = Query(10, ge=1, le=25, description="Max results to return"),
) -> SongSearchResponse:
    """Search Audius + Jamendo for songs matching `q`.

    Both providers are queried concurrently. If one fails we log it and return
    whatever the other produced, so a single provider outage degrades rather
    than fails. Results are round-robin interleaved and capped to `limit`.
    Only when *both* providers error do we surface a 502.
    """
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    audius_res, jamendo_res = await asyncio.gather(
        audius_client.search(query, limit),
        jamendo_client.search(query, limit),
        return_exceptions=True,
    )

    def _ok(res, provider: str) -> List[SongSearchResult]:
        if isinstance(res, Exception):
            logger.error(f"{provider} search failed for q={query!r}: {res}")
            return []
        return res

    if isinstance(audius_res, Exception) and isinstance(jamendo_res, Exception):
        raise HTTPException(
            status_code=502,
            detail="Song search failed. Try again or paste a link.",
        )

    merged = interleave(
        _ok(audius_res, "Audius"),
        _ok(jamendo_res, "Jamendo"),
    )[:limit]

    return SongSearchResponse(results=merged)
