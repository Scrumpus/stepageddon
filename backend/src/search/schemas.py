"""Shared schemas for the song-search endpoint.

One result shape across providers (Audius, Jamendo). ``source`` tags where a
row came from; ``url`` is a real provider URL that ``/api/generate-steps-url``
re-resolves to download the full track.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SongSearchResult(BaseModel):
    id: str
    source: Literal["audius", "jamendo"]
    title: str
    artist: str
    album: Optional[str] = None
    duration_seconds: Optional[int] = None
    thumbnail: str
    url: str


class SongSearchResponse(BaseModel):
    results: List[SongSearchResult] = Field(default_factory=list)
