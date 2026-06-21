"""Charts read API (currently a single endpoint nested under /songs)."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.charts.schemas import ChartDTO
from src.charts.service import get_chart, get_charts_for_song
from src.charts.sm_export import (
    build_simfile,
    build_song_zip,
    content_disposition,
    sanitize_filename,
)
from src.database import get_session
from src.songs.service import get_song
from src.storage import get_storage

router = APIRouter()


@router.get("/songs/{song_id}/charts/{difficulty}", response_model=ChartDTO)
async def get_song_chart(
    song_id: uuid.UUID,
    difficulty: str,
    chart_type: str = "dance-single",
    session: AsyncSession = Depends(get_session),
) -> ChartDTO:
    chart = await get_chart(session, song_id, difficulty, chart_type)
    if chart is None:
        raise HTTPException(404, "Chart not found")
    return ChartDTO(
        id=chart.id,
        song_id=chart.song_id,
        difficulty_name=chart.difficulty_name,
        difficulty_level=chart.difficulty_level,
        chart_type=chart.chart_type,
        steps=chart.steps,
        step_count=chart.step_count,
        hold_count=chart.hold_count,
        radar_values=chart.radar_values,
        generator=chart.generator,
    )


@router.get("/songs/{song_id}/export")
async def export_song(
    song_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Export a song + all its charts as a zipped StepMania song folder."""
    song = await get_song(session, song_id)
    if song is None:
        raise HTTPException(404, "Song not found")
    charts = await get_charts_for_song(session, song_id)
    if not charts:
        raise HTTPException(404, "Song has no charts")

    storage = get_storage()
    folder = sanitize_filename(song.title, default=str(song_id))

    assets: dict[str, bytes] = {}
    audio_name = f"{folder}{Path(song.audio_path).suffix or '.ogg'}"
    assets[audio_name] = storage.read_bytes(song.audio_path)
    for path, name in (
        (song.banner_path, "banner"),
        (song.jacket_path, "jacket"),
        (song.bg_path, "background"),
    ):
        if path:
            try:
                fname = f"{name}{Path(path).suffix or '.png'}"
                assets[fname] = storage.read_bytes(path)
            except HTTPException:
                pass  # missing optional asset — skip it

    meta = {
        "title": song.title,
        "artist": song.artist,
        "music": audio_name,
        "offset": song.offset,
        "bpms": song.bpms or [[0.0, song.tempo]],
        "banner": next((n for n in assets if n.startswith("banner")), None),
        "jacket": next((n for n in assets if n.startswith("jacket")), None),
        "background": next((n for n in assets if n.startswith("background")), None),
    }
    chart_dicts = [
        {
            "difficulty_name": c.difficulty_name,
            "difficulty_level": c.difficulty_level,
            "steps": c.steps,
            "radar_values": c.radar_values,
        }
        for c in charts
    ]

    sm_text = build_simfile(meta, chart_dicts)
    zip_bytes = build_song_zip(sm_text, folder, assets)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": content_disposition(f"{folder}.zip")},
    )
