"""Chart queries."""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Chart


async def get_charts_for_song(
    session: AsyncSession, song_id: uuid.UUID
) -> list[Chart]:
    stmt = (
        select(Chart)
        .where(Chart.song_id == song_id)
        .order_by(Chart.difficulty_level, Chart.difficulty_name)
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_chart(
    session: AsyncSession,
    song_id: uuid.UUID,
    difficulty_name: str,
    chart_type: str = "dance-single",
) -> Optional[Chart]:
    stmt = select(Chart).where(
        Chart.song_id == song_id,
        func.lower(Chart.difficulty_name) == difficulty_name.lower(),
        Chart.chart_type == chart_type,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
