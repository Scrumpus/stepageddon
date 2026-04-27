"""Playlist CRUD + membership endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from repositories import (
    add_song,
    create_playlist,
    delete_playlist,
    get_ordered_items,
    get_playlist,
    list_playlists,
    remove_song,
    reorder_songs,
    update_playlist,
)
from routers.songs import SongSummaryDTO, _to_summary

router = APIRouter()


class PlaylistDTO(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    song_count: int
    created_at: datetime
    updated_at: datetime


class PlaylistDetailDTO(PlaylistDTO):
    songs: list[SongSummaryDTO]


class PlaylistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None


class PlaylistUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None


class AddSongRequest(BaseModel):
    song_id: uuid.UUID
    position: Optional[int] = Field(default=None, ge=0)


class ReorderRequest(BaseModel):
    song_ids: list[uuid.UUID]


def _to_dto(playlist, song_count: int) -> PlaylistDTO:
    return PlaylistDTO(
        id=playlist.id,
        name=playlist.name,
        description=playlist.description,
        song_count=song_count,
        created_at=playlist.created_at,
        updated_at=playlist.updated_at,
    )


@router.get("/playlists", response_model=list[PlaylistDTO])
async def get_playlists(
    q: Optional[str] = None,
    limit: int = Query(100, le=500),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> list[PlaylistDTO]:
    playlists = await list_playlists(session, q=q, limit=limit, offset=offset)
    out: list[PlaylistDTO] = []
    for p in playlists:
        items = await get_ordered_items(session, p.id)
        out.append(_to_dto(p, len(items)))
    return out


@router.post("/playlists", response_model=PlaylistDTO, status_code=201)
async def create(
    body: PlaylistCreate, session: AsyncSession = Depends(get_session)
) -> PlaylistDTO:
    playlist = await create_playlist(
        session, name=body.name, description=body.description
    )
    return _to_dto(playlist, 0)


@router.get("/playlists/{playlist_id}", response_model=PlaylistDetailDTO)
async def get_detail(
    playlist_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> PlaylistDetailDTO:
    playlist = await get_playlist(session, playlist_id)
    if playlist is None:
        raise HTTPException(404, "Playlist not found")
    items = await get_ordered_items(session, playlist_id)
    return PlaylistDetailDTO(
        **_to_dto(playlist, len(items)).model_dump(),
        songs=[_to_summary(item.song) for item in items],
    )


@router.patch("/playlists/{playlist_id}", response_model=PlaylistDTO)
async def patch(
    playlist_id: uuid.UUID,
    body: PlaylistUpdate,
    session: AsyncSession = Depends(get_session),
) -> PlaylistDTO:
    playlist = await update_playlist(
        session,
        playlist_id,
        name=body.name,
        description=body.description,
    )
    if playlist is None:
        raise HTTPException(404, "Playlist not found")
    items = await get_ordered_items(session, playlist_id)
    return _to_dto(playlist, len(items))


@router.delete("/playlists/{playlist_id}", status_code=204, response_class=Response)
async def remove(
    playlist_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> Response:
    ok = await delete_playlist(session, playlist_id)
    if not ok:
        raise HTTPException(404, "Playlist not found")
    return Response(status_code=204)


@router.post("/playlists/{playlist_id}/songs", response_model=PlaylistDetailDTO)
async def add(
    playlist_id: uuid.UUID,
    body: AddSongRequest,
    session: AsyncSession = Depends(get_session),
) -> PlaylistDetailDTO:
    item = await add_song(
        session, playlist_id, body.song_id, position=body.position
    )
    if item is None:
        raise HTTPException(404, "Playlist or song not found, or song already in playlist")
    return await get_detail(playlist_id, session)


@router.delete(
    "/playlists/{playlist_id}/songs/{song_id}", response_model=PlaylistDetailDTO
)
async def remove_member(
    playlist_id: uuid.UUID,
    song_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> PlaylistDetailDTO:
    ok = await remove_song(session, playlist_id, song_id)
    if not ok:
        raise HTTPException(404, "Playlist or membership not found")
    return await get_detail(playlist_id, session)


@router.put("/playlists/{playlist_id}/order", response_model=PlaylistDetailDTO)
async def reorder(
    playlist_id: uuid.UUID,
    body: ReorderRequest,
    session: AsyncSession = Depends(get_session),
) -> PlaylistDetailDTO:
    result = await reorder_songs(session, playlist_id, body.song_ids)
    if result is None:
        raise HTTPException(
            400,
            "Reorder rejected — playlist not found or song_ids don't match current membership",
        )
    return await get_detail(playlist_id, session)
