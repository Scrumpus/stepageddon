"""Schemas for the YouTube Music search endpoint."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SongSearchResult(BaseModel):
    videoId: str
    title: str
    artist: str
    album: Optional[str] = None
    duration_seconds: Optional[int] = None
    thumbnail: str
    url: str


class SongSearchResponse(BaseModel):
    results: List[SongSearchResult] = Field(default_factory=list)
