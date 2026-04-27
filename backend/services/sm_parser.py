"""StepMania (.sm/.ssc) parsing — single source of truth.

Two views over one parse:

* :func:`parse_sm_file` returns a :class:`SmFile` of flat per-event ``Note``
  records (head/tail kept as separate events, ``arrow`` as an int 0-3).
  Used by ML training data prep.

* :func:`parse_sm` returns a :class:`ParsedSong` of grouped :class:`Step`
  records (jumps collapsed; HOLD heads carry ``hold_duration``; uses the
  project's Pydantic enums). Used by the seed script and runtime DB writer.

Both share the simfile loading, TimingEngine, dance-single filtering, and
header extraction below.
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


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ARROW_NAMES = ["left", "down", "up", "right"]

# StepMania column → game arrow direction. dance-single uses 4 columns.
COLUMN_TO_DIRECTION: dict[int, Direction] = {
    0: Direction.LEFT,
    1: Direction.DOWN,
    2: Direction.UP,
    3: Direction.RIGHT,
}

DIFFICULTY_MAP = {
    "beginner": 0,
    "easy": 1,
    "basic": 1,
    "medium": 2,
    "hard": 3,
    "challenge": 4,
    "expert": 4,
    "edit": 4,
}

NOTE_TYPE_MAP = {
    NoteType.TAP: "tap",
    NoteType.HOLD_HEAD: "hold_head",
    NoteType.TAIL: "hold_tail",
    NoteType.ROLL_HEAD: "roll_head",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """``#BPMS:0.000=31.250,4.000=134.922`` → ``[(0.0, 31.25), (4.0, 134.922)]``."""
    out: list[tuple[float, float]] = []
    if not raw:
        return out
    for entry in raw.replace("\n", "").split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
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


def _beat_subdivision(beat) -> BeatSubdivision:
    frac = Fraction(beat).limit_denominator(192) % 1
    if frac == 0:
        return BeatSubdivision.QUARTER
    if frac.denominator <= 2:
        return BeatSubdivision.EIGHTH
    return BeatSubdivision.SIXTEENTH


def _resolve_display_bpm(raw_display, bpms: list[tuple[float, float]]) -> float:
    if raw_display and str(raw_display).strip() not in ("", "*"):
        try:
            return float(str(raw_display).split(":", 1)[0].split("-", 1)[0])
        except ValueError:
            pass
    return bpms[0][1] if bpms else 120.0


# ---------------------------------------------------------------------------
# Low-level (ML) view: flat per-event Notes
# ---------------------------------------------------------------------------

@dataclass
class Note:
    """Single per-event note: head/tail/tap kept separate, arrow as int."""
    time: float
    arrow: int          # 0-3 (L/D/U/R)
    note_type: str      # 'tap' | 'hold_head' | 'hold_tail' | 'roll_head'


@dataclass
class NoteChart:
    game_type: str        # always 'dance-single' (others filtered out)
    difficulty: str       # 'Beginner', 'Easy', 'Medium', 'Hard', 'Challenge', 'Edit'
    difficulty_id: int    # 0-4 mapped via DIFFICULTY_MAP
    meter: int            # numeric difficulty rating
    notes: list[Note] = field(default_factory=list)
    radar_values: Optional[list[float]] = None


@dataclass
class SmFile:
    title: str = ""
    artist: str = ""
    offset: float = 0.0
    primary_bpm: float = 120.0
    display_bpm: float = 120.0
    bpms: list[tuple[float, float]] = field(default_factory=list)
    music_file: Optional[str] = None
    banner: Optional[str] = None
    background: Optional[str] = None
    jacket: Optional[str] = None
    charts: list[NoteChart] = field(default_factory=list)


def parse_sm_file(filepath: str) -> Optional[SmFile]:
    """Parse a .sm/.ssc file into flat per-event notes.

    Returns ``None`` on parse error or if no usable dance-single charts exist.
    Drops non-``dance-single`` charts. Note types other than tap/hold_head/
    hold_tail/roll_head (mines, fakes, lifts, attacks) are filtered out.
    """
    try:
        sf = simfile.open(str(filepath))
    except Exception as e:
        logger.warning("Failed to open %s: %s", filepath, e)
        return None

    timing = TimingData(sf)
    engine = TimingEngine(timing)

    bpms = _parse_bpms(getattr(sf, "bpms", "") or "")
    primary_bpm = bpms[0][1] if bpms else 120.0
    display_bpm = _resolve_display_bpm(getattr(sf, "displaybpm", None), bpms)

    result = SmFile(
        title=str(sf.title or "").strip(),
        artist=str(getattr(sf, "artist", "") or "").strip(),
        offset=_safe_float(getattr(sf, "offset", "0"), default=0.0),
        primary_bpm=primary_bpm,
        display_bpm=display_bpm,
        bpms=bpms,
        music_file=(str(sf.music).strip() or None) if getattr(sf, "music", None) else None,
        banner=(str(sf.banner).strip() or None) if getattr(sf, "banner", None) else None,
        background=(str(sf.background).strip() or None) if getattr(sf, "background", None) else None,
        jacket=(str(getattr(sf, "jacket", "")).strip() or None) if getattr(sf, "jacket", None) else None,
    )

    for chart in sf.charts:
        chart_type = (chart.stepstype or "").strip()
        if chart_type != "dance-single":
            continue

        diff_key = (chart.difficulty or "").lower().strip()
        difficulty_id = DIFFICULTY_MAP.get(diff_key, 2)
        meter = _safe_int(chart.meter, default=1)

        notes: list[Note] = []
        try:
            for n in NoteData(chart):
                if n.note_type not in NOTE_TYPE_MAP:
                    continue
                if n.column not in COLUMN_TO_DIRECTION:
                    continue
                t = float(engine.time_at(TimingBeat(n.beat)))
                notes.append(Note(time=t, arrow=n.column, note_type=NOTE_TYPE_MAP[n.note_type]))
        except Exception as e:
            logger.debug("Failed to parse notes for %s %s: %s", result.title, chart.difficulty, e)
            continue

        if not notes:
            continue

        result.charts.append(
            NoteChart(
                game_type=chart_type,
                difficulty=(chart.difficulty or "").strip() or "Unknown",
                difficulty_id=difficulty_id,
                meter=meter,
                notes=notes,
                radar_values=_parse_radar(getattr(chart, "radarvalues", None)),
            )
        )

    if not result.charts:
        logger.debug("No dance-single charts in %s", filepath)
        return None

    return result


# ---------------------------------------------------------------------------
# High-level (runtime/seed) view: grouped Steps with hold durations
# ---------------------------------------------------------------------------

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


def _group_into_steps(notes: list[Note], chart) -> list[Step]:
    """Collapse per-event notes into Steps (jumps merged, holds matched).

    ``chart`` is the underlying simfile Chart, only consulted to recover the
    raw ``beat`` for each event so we can derive ``beat_subdivision``.
    """
    # We need the raw beat for subdivision derivation, which `parse_sm_file`
    # discards. Re-iterate the chart in lockstep with `notes` (same order).
    raw_iter = (
        n for n in NoteData(chart)
        if n.note_type in NOTE_TYPE_MAP and n.column in COLUMN_TO_DIRECTION
    )

    # Column → list of unmatched HOLD_HEAD/ROLL_HEAD events (beat_f, time)
    open_holds: dict[int, list[tuple[float, float]]] = {}
    grouped: dict[float, dict] = {}

    for note, raw in zip(notes, raw_iter):
        beat_f = float(raw.beat)
        col = note.arrow
        nt = note.note_type
        t = note.time

        if nt == "hold_tail":
            opens = open_holds.get(col, [])
            if not opens:
                continue
            head_beat, head_time = opens.pop()
            head_entry = grouped.get(head_beat)
            if head_entry is None:
                continue
            duration = max(0.0, t - head_time)
            for i, dirn in enumerate(head_entry["arrows"]):
                if dirn is COLUMN_TO_DIRECTION[col] and head_entry["hold_durations"][i] is None:
                    head_entry["hold_durations"][i] = duration
                    break
            continue

        # tap / hold_head / roll_head produce a Step entry
        entry = grouped.setdefault(
            beat_f,
            {
                "time": t,
                "arrows": [],
                "hold_durations": [],
                "subdivision": _beat_subdivision(raw.beat),
                "has_hold": False,
            },
        )
        entry["arrows"].append(COLUMN_TO_DIRECTION[col])
        entry["hold_durations"].append(None)
        if nt in ("hold_head", "roll_head"):
            entry["has_hold"] = True
            open_holds.setdefault(col, []).append((beat_f, t))

    out: list[Step] = []
    for beat_f in sorted(grouped):
        entry = grouped[beat_f]
        held = [d for d in entry["hold_durations"] if d is not None]
        if entry["has_hold"] and held:
            step_type = StepType.HOLD
            hold_duration = max(held)
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
    """Parse a .sm/.ssc file into ``Step``-grouped charts.

    For songs that ship multiple charts at the same ``(difficulty, type)``
    (alternate stepartists), keeps the highest-meter one to satisfy the DB's
    ``(song_id, difficulty_name, chart_type)`` uniqueness constraint.
    """
    sf_lowlevel = parse_sm_file(str(path))
    if sf_lowlevel is None:
        return ParsedSong(
            title=path.parent.name,
            artist=None,
            offset=0.0,
            display_bpm=120.0,
            bpms=[],
            music_file=None,
            banner=None,
            background=None,
            jacket=None,
            charts=[],
        )

    # Re-open once to access raw simfile Chart objects (needed for beat info
    # used in subdivision derivation). simfile.open is cheap (text parse).
    sf = simfile.open(str(path))
    sm_charts_by_key = {}
    for sc in sf.charts:
        key = ((sc.difficulty or "").strip() or "Unknown", (sc.stepstype or "").strip())
        sm_charts_by_key.setdefault(key, []).append(sc)

    by_key: dict[tuple[str, str], ParsedChart] = {}
    for nc in sf_lowlevel.charts:
        key = (nc.difficulty, nc.game_type)
        # match the simfile Chart for this NoteChart by (difficulty, type, meter)
        candidates = sm_charts_by_key.get(key, [])
        sm_chart = next(
            (c for c in candidates if _safe_int(c.meter, default=1) == nc.meter),
            candidates[0] if candidates else None,
        )
        if sm_chart is None:
            continue
        try:
            steps = _group_into_steps(nc.notes, sm_chart)
        except Exception as e:
            logger.warning("Group step build failed for %s %s: %s", path, nc.difficulty, e)
            continue
        if not steps:
            continue
        parsed = ParsedChart(
            difficulty_name=nc.difficulty,
            difficulty_level=nc.meter,
            chart_type=nc.game_type,
            steps=steps,
            radar_values=nc.radar_values,
        )
        existing = by_key.get(key)
        if existing is None or nc.meter > existing.difficulty_level:
            by_key[key] = parsed

    return ParsedSong(
        title=sf_lowlevel.title or path.parent.name,
        artist=sf_lowlevel.artist or None,
        offset=sf_lowlevel.offset,
        display_bpm=sf_lowlevel.display_bpm,
        bpms=sf_lowlevel.bpms,
        music_file=sf_lowlevel.music_file,
        banner=sf_lowlevel.banner,
        background=sf_lowlevel.background,
        jacket=sf_lowlevel.jacket,
        charts=list(by_key.values()),
    )
