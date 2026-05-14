"""Request/response schemas for the generation API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for URL-based generation"""
    url: str = Field(..., description="YouTube or Spotify URL")
    style: str = Field(
        "auto",
        description="Style profile name (e.g. 'Stream-Heavy', 'Jump-Heavy') or 'auto'",
    )


class GenerateResponse(BaseModel):
    """Response model for step generation"""
    song_id: str
    charts: dict
    song_info: dict
    audio_url: str
