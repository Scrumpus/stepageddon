"""Playlist queries + mutations."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.playlists.models import Playlist, PlaylistSong
from src.songs.models import Song


async def list_playlists(
    session: AsyncSession, *, q: Optional[str] = None, limit: int = 100, offset: int = 0
) -> list[Playlist]:
    stmt = select(Playlist)
    if q:
        stmt = stmt.where(Playlist.name.ilike(f"%{q}%"))
    stmt = stmt.order_by(Playlist.name).limit(limit).offset(offset)
    return list((await session.execute(stmt)).scalars().all())


async def get_playlist(
    session: AsyncSession, playlist_id: uuid.UUID, *, with_songs: bool = False
) -> Optional[Playlist]:
    stmt = select(Playlist).where(Playlist.id == playlist_id)
    if with_songs:
        stmt = stmt.options(selectinload(Playlist.items).selectinload(PlaylistSong.song))
    return (await session.execute(stmt)).scalar_one_or_none()


async def create_playlist(
    session: AsyncSession, *, name: str, description: Optional[str] = None
) -> Playlist:
    playlist = Playlist(name=name, description=description)
    session.add(playlist)
    await session.flush()
    return playlist


async def update_playlist(
    session: AsyncSession,
    playlist_id: uuid.UUID,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Playlist]:
    playlist = await session.get(Playlist, playlist_id)
    if playlist is None:
        return None
    if name is not None:
        playlist.name = name
    if description is not None:
        playlist.description = description
    playlist.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return playlist


async def delete_playlist(session: AsyncSession, playlist_id: uuid.UUID) -> bool:
    result = await session.execute(
        delete(Playlist).where(Playlist.id == playlist_id)
    )
    return (result.rowcount or 0) > 0


async def _next_position(session: AsyncSession, playlist_id: uuid.UUID) -> int:
    stmt = select(func.coalesce(func.max(PlaylistSong.position), -1)).where(
        PlaylistSong.playlist_id == playlist_id
    )
    return int((await session.execute(stmt)).scalar_one()) + 1


async def add_song(
    session: AsyncSession,
    playlist_id: uuid.UUID,
    song_id: uuid.UUID,
    *,
    position: Optional[int] = None,
) -> Optional[PlaylistSong]:
    """Append (or insert at position) a song into a playlist.

    Returns None if either playlist or song is missing. Inserting at a
    specific ``position`` shifts later items down by one to keep
    ``(playlist_id, position)`` unique.
    """
    playlist = await session.get(Playlist, playlist_id)
    song = await session.get(Song, song_id)
    if playlist is None or song is None:
        return None

    if position is None:
        target_pos = await _next_position(session, playlist_id)
    else:
        target_pos = max(0, position)
        # Shift items at >= target_pos down by 1 (decrement-from-top to avoid
        # transient unique-constraint clashes).
        existing_max = await _next_position(session, playlist_id) - 1
        for p in range(existing_max, target_pos - 1, -1):
            await session.execute(
                update(PlaylistSong)
                .where(
                    PlaylistSong.playlist_id == playlist_id,
                    PlaylistSong.position == p,
                )
                .values(position=p + 1)
            )

    item = PlaylistSong(
        playlist_id=playlist_id, song_id=song_id, position=target_pos
    )
    session.add(item)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        return None

    playlist.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return item


async def remove_song(
    session: AsyncSession, playlist_id: uuid.UUID, song_id: uuid.UUID
) -> bool:
    """Remove a song from a playlist and compact remaining positions."""
    item = (
        await session.execute(
            select(PlaylistSong).where(
                PlaylistSong.playlist_id == playlist_id,
                PlaylistSong.song_id == song_id,
            )
        )
    ).scalar_one_or_none()
    if item is None:
        return False
    removed_pos = item.position
    await session.delete(item)
    await session.flush()

    # Compact positions above the removed slot
    await session.execute(
        update(PlaylistSong)
        .where(
            PlaylistSong.playlist_id == playlist_id,
            PlaylistSong.position > removed_pos,
        )
        .values(position=PlaylistSong.position - 1)
    )

    playlist = await session.get(Playlist, playlist_id)
    if playlist is not None:
        playlist.updated_at = datetime.now(timezone.utc)
    return True


async def reorder_songs(
    session: AsyncSession,
    playlist_id: uuid.UUID,
    song_ids: list[uuid.UUID],
) -> Optional[list[PlaylistSong]]:
    """Replace the playlist's ordering with the supplied song_id sequence.

    All existing items must appear exactly once in ``song_ids``. Returns the
    new ordering, or None if the playlist doesn't exist or the input doesn't
    match the current membership.
    """
    playlist = await session.get(Playlist, playlist_id)
    if playlist is None:
        return None
    current = (
        await session.execute(
            select(PlaylistSong).where(PlaylistSong.playlist_id == playlist_id)
        )
    ).scalars().all()
    current_ids = {item.song_id for item in current}
    if set(song_ids) != current_ids or len(song_ids) != len(current_ids):
        return None

    # Two-phase update to dodge the unique constraint: park everything at
    # negative positions first, then assign final positions.
    for i, item in enumerate(current):
        item.position = -(i + 1)
    await session.flush()

    by_song = {item.song_id: item for item in current}
    for new_pos, song_id in enumerate(song_ids):
        by_song[song_id].position = new_pos
    await session.flush()

    playlist.updated_at = datetime.now(timezone.utc)
    return await get_ordered_items(session, playlist_id)


async def get_ordered_items(
    session: AsyncSession, playlist_id: uuid.UUID
) -> list[PlaylistSong]:
    stmt = (
        select(PlaylistSong)
        .where(PlaylistSong.playlist_id == playlist_id)
        .order_by(PlaylistSong.position)
        .options(selectinload(PlaylistSong.song))
    )
    return list((await session.execute(stmt)).scalars().all())
