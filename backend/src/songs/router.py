"""Songs read API."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.charts.schemas import ChartSummaryDTO
from src.charts.service import get_charts_for_song
from src.database import get_session
from src.songs.models import SongSource
from src.songs.schemas import SongDetailDTO, SongSummaryDTO
from src.songs.service import get_song, list_songs
from src.songs.utils import _to_summary

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/songs", response_model=list[SongSummaryDTO])
async def get_songs(
    source: Optional[SongSource] = None,
    game: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> list[SongSummaryDTO]:
    songs = await list_songs(
        session, source=source, game=game, q=q, limit=limit, offset=offset
    )
    return [_to_summary(s) for s in songs]


@router.get("/songs/{song_id}", response_model=SongDetailDTO)
async def get_song_detail(
    song_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> SongDetailDTO:
    song = await get_song(session, song_id)
    if song is None:
        raise HTTPException(404, "Song not found")
    charts = await get_charts_for_song(session, song_id)
    summary = _to_summary(song)
    return SongDetailDTO(
        **summary.model_dump(),
        offset=song.offset,
        bpms=song.bpms,
        charts=[
            ChartSummaryDTO(
                id=c.id,
                difficulty_name=c.difficulty_name,
                difficulty_level=c.difficulty_level,
                chart_type=c.chart_type,
                step_count=c.step_count,
                hold_count=c.hold_count,
                generator=c.generator,
            )
            for c in charts
        ],
    )
