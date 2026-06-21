"""Request/response schemas for the generation API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for URL-based generation"""
    url: str = Field(..., description="YouTube or Spotify URL")
    style: str = Field(
        "auto",
        description="Style profile name (e.g. 'Stream-Heavy', 'Jump-Heavy') or 'auto'",
    )


class ExportRequest(BaseModel):
    """Export freshly generated (not-yet-persisted) charts as a StepMania zip.

    Mirrors the ``generate-steps`` response the menu flow already holds: the
    ``charts`` map is ``{difficulty: chart.to_json_dict()}`` and ``song_info``
    carries title/artist/tempo/timing_offset.
    """
    song_info: dict = Field(..., description="title, artist, tempo, timing_offset")
    audio_url: str = Field(..., description="/api/audio/{key} of the generated audio")
    charts: dict = Field(..., description="{difficulty: chart JSON with steps}")


class GenerateResponse(BaseModel):
    """Response model for step generation"""
    song_id: str
    charts: dict
    song_info: dict
    audio_url: str
