"""Pydantic DTOs for the songs API."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel

from src.charts.schemas import ChartSummaryDTO
from src.songs.models import SongSource


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


class SongDetailDTO(SongSummaryDTO):
    offset: float
    bpms: Optional[list]
    charts: list[ChartSummaryDTO]
