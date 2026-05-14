"""Pydantic DTOs for the playlists API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.songs.schemas import SongSummaryDTO


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
