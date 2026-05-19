"""Charts read API (currently a single endpoint nested under /songs)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.charts.schemas import ChartDTO
from src.charts.service import get_chart
from src.database import get_session

router = APIRouter()


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
