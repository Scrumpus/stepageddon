"""StepMania (.sm/.ssc) parser → list[Step] using the `simfile` library.

Wraps `simfile` so the rest of the codebase deals only with the project's
existing `Step` / `Direction` / `StepType` / `BeatSubdivision` enums.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.timing.engine import Beat as TimingBeat
from simfile.timing.engine import TimingEngine

from modules.step_generator.schemas import (
    BeatSubdivision,
    Direction,
    Step,
    StepType,
)

logger = logging.getLogger(__name__)


# StepMania column → game arrow direction. dance-single uses 4 columns.
COLUMN_TO_DIRECTION: dict[int, Direction] = {
    0: Direction.LEFT,
    1: Direction.DOWN,
    2: Direction.UP,
    3: Direction.RIGHT,
}


def _beat_subdivision(beat) -> BeatSubdivision:
    """Map a fractional beat to the existing BeatSubdivision enum."""
    frac = Fraction(beat).limit_denominator(192) % 1
    if frac == 0:
        return BeatSubdivision.QUARTER
    if frac.denominator <= 2:
        return BeatSubdivision.EIGHTH
    return BeatSubdivision.SIXTEENTH


def _safe_int(s, default: int = 0) -> int:
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return default


def _safe_float(s, default: float = 0.0) -> float:
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return default


def _parse_bpms(raw: str) -> list[tuple[float, float]]:
    """`#BPMS:0.000=31.250,4.000=134.922` → [(0.0, 31.25), (4.0, 134.922)]."""
    out: list[tuple[float, float]] = []
    if not raw:
        return out
    for entry in raw.replace("\n", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            continue
        beat_s, bpm_s = entry.split("=", 1)
        try:
            out.append((float(beat_s), float(bpm_s)))
        except ValueError:
            continue
    return out


def _parse_radar(raw) -> Optional[list[float]]:
    if raw is None:
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    try:
        return [float(p) for p in parts] if parts else None
    except ValueError:
        return None


@dataclass
class ParsedChart:
    difficulty_name: str
    difficulty_level: int
    chart_type: str
    steps: list[Step]
    radar_values: Optional[list[float]] = None


@dataclass
class ParsedSong:
    title: str
    artist: Optional[str]
    offset: float
    display_bpm: float
    bpms: list[tuple[float, float]]
    music_file: Optional[str]
    banner: Optional[str]
    background: Optional[str]
    jacket: Optional[str]
    charts: list[ParsedChart] = field(default_factory=list)


def _convert_chart(chart, engine: TimingEngine) -> list[Step]:
    """Convert a single .sm Chart into a list of `Step` instances.

    Hold-tail matching is per-column and stateful across the whole chart.
    Notes at the same beat collapse into a single `Step` (jumps).
    """
    notes = list(NoteData(chart))
    notes.sort(key=lambda n: (float(n.beat), n.column))

    # Column → list of unmatched HOLD_HEAD/ROLL_HEAD events (beat, time)
    open_holds: dict[int, list[tuple[float, float]]] = {}

    # Beat → {"arrows": list[Direction], "step_type": StepType,
    #         "hold_durations": list[float|None], "subdivision": BeatSubdivision}
    grouped: dict[float, dict] = {}

    for n in notes:
        col = n.column
        if col not in COLUMN_TO_DIRECTION:  # ignore non-single columns if present
            continue
        beat_f = float(n.beat)
        t = float(engine.time_at(TimingBeat(n.beat)))
        nt = n.note_type

        if nt == NoteType.TAIL:
            # Match to most recent open head on this column
            opens = open_holds.get(col, [])
            if not opens:
                logger.debug("Orphan tail at beat %s col %s — skipping", beat_f, col)
                continue
            head_beat, head_time = opens.pop()
            head_entry = grouped.get(head_beat)
            if head_entry is None:
                continue
            duration = max(0.0, t - head_time)
            # Update the matching arrow's hold_duration
            for i, dirn in enumerate(head_entry["arrows"]):
                if dirn is COLUMN_TO_DIRECTION[col] and head_entry["hold_durations"][i] is None:
                    head_entry["hold_durations"][i] = duration
                    break
            continue

        if nt in (NoteType.MINE, NoteType.FAKE, NoteType.KEYSOUND, NoteType.ATTACK, NoteType.LIFT):
            # Skip in v1
            continue

        # TAP / HOLD_HEAD / ROLL_HEAD all produce a Step entry
        entry = grouped.setdefault(
            beat_f,
            {
                "time": t,
                "arrows": [],
                "hold_durations": [],
                "subdivision": _beat_subdivision(n.beat),
                "has_hold": False,
            },
        )
        entry["arrows"].append(COLUMN_TO_DIRECTION[col])

        if nt in (NoteType.HOLD_HEAD, NoteType.ROLL_HEAD):
            entry["hold_durations"].append(None)  # filled when we see the TAIL
            entry["has_hold"] = True
            open_holds.setdefault(col, []).append((beat_f, t))
        else:
            entry["hold_durations"].append(None)

    # Build Step list in time order
    out: list[Step] = []
    for beat_f in sorted(grouped):
        entry = grouped[beat_f]
        # If any arrow on this step is a hold-head, the step type is HOLD; the
        # hold_duration we expose is the longest among held arrows on this step.
        held_durations = [d for d in entry["hold_durations"] if d is not None]
        if entry["has_hold"] and held_durations:
            step_type = StepType.HOLD
            hold_duration = max(held_durations)
        else:
            step_type = StepType.TAP
            hold_duration = None
        out.append(
            Step(
                time=round(entry["time"], 4),
                arrows=entry["arrows"],
                step_type=step_type,
                hold_duration=round(hold_duration, 4) if hold_duration is not None else None,
                beat_subdivision=entry["subdivision"],
            )
        )
    return out


def parse_sm(path: Path) -> ParsedSong:
    """Parse a .sm/.ssc file. Drops non-`dance-single` charts in v1.

    For songs with multiple charts at the same `(difficulty_name, chart_type)`
    (alternate stepartist versions), keeps the one with the highest meter to
    satisfy the `(song_id, difficulty_name, chart_type)` uniqueness constraint.
    """
    sf = simfile.open(str(path))
    engine = TimingEngine(TimingData(sf))

    raw_display = getattr(sf, "displaybpm", None)
    bpms = _parse_bpms(getattr(sf, "bpms", "") or "")
    if raw_display and str(raw_display).strip() not in ("", "*"):
        try:
            display_bpm = float(str(raw_display).split(":", 1)[0].split("-", 1)[0])
        except ValueError:
            display_bpm = bpms[0][1] if bpms else 120.0
    else:
        display_bpm = bpms[0][1] if bpms else 120.0

    by_key: dict[tuple[str, str], ParsedChart] = {}
    for chart in sf.charts:
        chart_type = (chart.stepstype or "dance-single").strip()
        if chart_type != "dance-single":
            continue  # phase-2 territory
        difficulty = (chart.difficulty or "").strip() or "Unknown"
        meter = _safe_int(chart.meter, default=0)
        try:
            steps = _convert_chart(chart, engine)
        except Exception as e:
            logger.warning("Failed to parse chart %s in %s: %s", difficulty, path, e)
            continue
        if not steps:
            continue
        radar = _parse_radar(getattr(chart, "radarvalues", None))
        parsed = ParsedChart(
            difficulty_name=difficulty,
            difficulty_level=meter,
            chart_type=chart_type,
            steps=steps,
            radar_values=radar,
        )
        key = (difficulty, chart_type)
        existing = by_key.get(key)
        if existing is None or meter > existing.difficulty_level:
            by_key[key] = parsed

    return ParsedSong(
        title=str(sf.title or path.parent.name).strip(),
        artist=(str(sf.artist).strip() or None) if getattr(sf, "artist", None) else None,
        offset=_safe_float(getattr(sf, "offset", "0"), default=0.0),
        display_bpm=display_bpm,
        bpms=bpms,
        music_file=(str(sf.music).strip() or None) if getattr(sf, "music", None) else None,
        banner=(str(sf.banner).strip() or None) if getattr(sf, "banner", None) else None,
        background=(str(sf.background).strip() or None) if getattr(sf, "background", None) else None,
        jacket=(str(getattr(sf, "jacket", "")).strip() or None) if getattr(sf, "jacket", None) else None,
        charts=list(by_key.values()),
    )
