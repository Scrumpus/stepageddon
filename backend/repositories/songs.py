"""Song queries."""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Song, SongSource


async def list_songs(
    session: AsyncSession,
    *,
    source: Optional[SongSource] = None,
    game: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Song]:
    stmt = select(Song)
    if source is not None:
        stmt = stmt.where(Song.source == source)
    if game is not None:
        stmt = stmt.where(Song.ddr_game == game)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Song.title.ilike(like), Song.artist.ilike(like)))
    stmt = stmt.order_by(
        Song.source, Song.ddr_release_index.nulls_last(), Song.title
    ).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_song(session: AsyncSession, song_id: uuid.UUID) -> Optional[Song]:
    return await session.get(Song, song_id)
