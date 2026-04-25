"""Songs + charts read API."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from db.models import SongSource
from repositories import (
    get_chart,
    get_charts_for_song,
    get_song,
    list_songs,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SongSummaryDTO(BaseModel):
    id: uuid.UUID
    title: str
    artist: Optional[str]
    source: SongSource
    slug: str
    audio_url: str
    banner_url: Optional[str]
    jacket_url: Optional[str]
    bg_url: Optional[str]
    tempo: float
    duration: float
    ddr_game: Optional[str]
    ddr_release_index: Optional[int]


class ChartSummaryDTO(BaseModel):
    id: uuid.UUID
    difficulty_name: str
    difficulty_level: int
    chart_type: str
    step_count: int
    hold_count: int
    generator: str


class SongDetailDTO(SongSummaryDTO):
    offset: float
    bpms: Optional[list]
    charts: list[ChartSummaryDTO]


class ChartDTO(BaseModel):
    id: uuid.UUID
    song_id: uuid.UUID
    difficulty_name: str
    difficulty_level: int
    chart_type: str
    steps: list
    step_count: int
    hold_count: int
    radar_values: Optional[list]
    generator: str


def _audio_url(relpath: Optional[str]) -> Optional[str]:
    if not relpath:
        return None
    return f"/api/audio/{relpath}"


def _to_summary(song) -> SongSummaryDTO:
    return SongSummaryDTO(
        id=song.id,
        title=song.title,
        artist=song.artist,
        source=song.source,
        slug=song.slug,
        audio_url=_audio_url(song.audio_path) or "",
        banner_url=_audio_url(song.banner_path),
        jacket_url=_audio_url(song.jacket_path),
        bg_url=_audio_url(song.bg_path),
        tempo=song.tempo,
        duration=song.duration,
        ddr_game=song.ddr_game,
        ddr_release_index=song.ddr_release_index,
    )


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


@router.get("/songs/{song_id}/charts/{difficulty}", response_model=ChartDTO)
async def get_song_chart(
    song_id: uuid.UUID,
    difficulty: str,
    chart_type: str = "dance-single",
    session: AsyncSession = Depends(get_session),
) -> ChartDTO:
    chart = await get_chart(session, song_id, difficulty, chart_type)
    if chart is None:
        raise HTTPException(404, "Chart not found")
    return ChartDTO(
        id=chart.id,
        song_id=chart.song_id,
        difficulty_name=chart.difficulty_name,
        difficulty_level=chart.difficulty_level,
        chart_type=chart.chart_type,
        steps=chart.steps,
        step_count=chart.step_count,
        hold_count=chart.hold_count,
        radar_values=chart.radar_values,
        generator=chart.generator,
    )
