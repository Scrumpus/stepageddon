"""Chart queries + persistence.

Reads: :func:`get_chart`, :func:`get_charts_for_song`.

Writes: :func:`persist_user_song` (single ML-generated chart on a new USER
song) and :func:`persist_ddr_song` (idempotent UPSERT for a parsed DDR song +
all its charts). Shared by the seed script and the generation router so the
JSONB step shape stays identical regardless of source.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.charts.models import Chart
from src.charts.schemas import Chart as PydanticChart
from src.charts.schemas import StepType
from src.songs.models import Song, SongSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def get_charts_for_song(
    session: AsyncSession, song_id: uuid.UUID
) -> list[Chart]:
    stmt = (
        select(Chart)
        .where(Chart.song_id == song_id)
        .order_by(Chart.difficulty_level, Chart.difficulty_name)
    )
    return list((await session.execute(stmt)).scalars().all())


async def get_chart(
    session: AsyncSession,
    song_id: uuid.UUID,
    difficulty_name: str,
    chart_type: str = "dance-single",
) -> Optional[Chart]:
    stmt = select(Chart).where(
        Chart.song_id == song_id,
        func.lower(Chart.difficulty_name) == difficulty_name.lower(),
        Chart.chart_type == chart_type,
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _step_to_dict(step) -> dict:
    """Mirror ``Chart.to_json_dict()`` per-step shape so frontend stays unchanged."""
    out = {
        "time": round(step.time, 3),
        "arrows": [a.value for a in step.arrows],
        "type": step.step_type.value,
        "beat_subdivision": step.beat_subdivision.value,
    }
    if step.step_type == StepType.HOLD and step.hold_duration is not None:
        out["hold_duration"] = round(step.hold_duration, 3)
    return out


def _hold_count(steps) -> int:
    return sum(1 for s in steps if s.step_type == StepType.HOLD)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def persist_user_song(
    session: AsyncSession,
    *,
    song_id: uuid.UUID,
    title: str,
    artist: Optional[str],
    audio_relpath: str,
    chart: PydanticChart,
    difficulty_name: str,
    difficulty_level: int,
    generator: str,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert a USER song + its single generated chart. Returns (song_id, chart_id)."""
    song_values = {
        "id": song_id,
        "title": title,
        "artist": artist,
        "source": SongSource.USER.value,
        "slug": str(song_id),
        "audio_path": audio_relpath,
        "tempo": float(chart.tempo),
        "duration": float(chart.duration),
        "offset": 0.0,
    }
    song_stmt = pg_insert(Song).values(**song_values).on_conflict_do_nothing(
        index_elements=["id"]
    ).returning(Song.id)
    result = await session.execute(song_stmt)
    row = result.first()
    persisted_song_id = row[0] if row else song_id

    steps_json = [_step_to_dict(s) for s in chart.steps]
    chart_stmt = (
        pg_insert(Chart)
        .values(
            song_id=persisted_song_id,
            difficulty_name=difficulty_name,
            difficulty_level=difficulty_level,
            chart_type="dance-single",
            steps=steps_json,
            step_count=len(chart.steps),
            hold_count=_hold_count(chart.steps),
            generator=generator,
        )
        .returning(Chart.id)
    )
    chart_id = (await session.execute(chart_stmt)).scalar_one()
    return persisted_song_id, chart_id


async def persist_ddr_song(
    session: AsyncSession,
    *,
    parsed,  # src.songs.utils.ParsedSong — runtime import to avoid songs<->charts cycle
    slug: str,
    audio_relpath: str,
    banner_relpath: Optional[str],
    jacket_relpath: Optional[str],
    bg_relpath: Optional[str],
    duration: float,
    ddr_game: str,
    ddr_release_index: Optional[int],
) -> uuid.UUID:
    """Idempotent UPSERT of a DDR song + all its parsed charts. Returns song_id."""
    song_values = {
        "title": parsed.title,
        "artist": parsed.artist,
        "source": SongSource.DDR.value,
        "slug": slug,
        "audio_path": audio_relpath,
        "banner_path": banner_relpath,
        "jacket_path": jacket_relpath,
        "bg_path": bg_relpath,
        "tempo": float(parsed.display_bpm),
        "duration": float(duration),
        "offset": float(parsed.offset),
        "bpms": [list(t) for t in parsed.bpms],
        "ddr_game": ddr_game,
        "ddr_release_index": ddr_release_index,
    }
    song_stmt = pg_insert(Song).values(**song_values)
    song_stmt = song_stmt.on_conflict_do_update(
        index_elements=["source", "ddr_game", "title"],
        index_where=text("source = 'DDR'"),
        set_={
            "artist": song_stmt.excluded.artist,
            "slug": song_stmt.excluded.slug,
            "audio_path": song_stmt.excluded.audio_path,
            "banner_path": song_stmt.excluded.banner_path,
            "jacket_path": song_stmt.excluded.jacket_path,
            "bg_path": song_stmt.excluded.bg_path,
            "tempo": song_stmt.excluded.tempo,
            "duration": song_stmt.excluded.duration,
            "offset": song_stmt.excluded.offset,
            "bpms": song_stmt.excluded.bpms,
            "ddr_release_index": song_stmt.excluded.ddr_release_index,
        },
    ).returning(Song.id)
    song_id = (await session.execute(song_stmt)).scalar_one()

    for pc in parsed.charts:
        await _upsert_chart(session, song_id, pc)
    return song_id


async def _upsert_chart(
    session: AsyncSession, song_id: uuid.UUID, pc
) -> None:
    """Upsert one parsed chart for a song. ``pc`` is a src.songs.utils.ParsedChart."""
    steps_json = [_step_to_dict(s) for s in pc.steps]
    chart_stmt = pg_insert(Chart).values(
        song_id=song_id,
        difficulty_name=pc.difficulty_name,
        difficulty_level=pc.difficulty_level,
        chart_type=pc.chart_type,
        steps=steps_json,
        step_count=len(pc.steps),
        hold_count=_hold_count(pc.steps),
        radar_values=pc.radar_values,
        generator="ddr-official",
    )
    chart_stmt = chart_stmt.on_conflict_do_update(
        constraint="uq_charts_song_difficulty_type",
        set_={
            "difficulty_level": chart_stmt.excluded.difficulty_level,
            "steps": chart_stmt.excluded.steps,
            "step_count": chart_stmt.excluded.step_count,
            "hold_count": chart_stmt.excluded.hold_count,
            "radar_values": chart_stmt.excluded.radar_values,
            "generator": chart_stmt.excluded.generator,
        },
    )
    await session.execute(chart_stmt)
