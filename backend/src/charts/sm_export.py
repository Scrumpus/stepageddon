"""Export charts back to StepMania ``.sm`` simfiles (inverse of the importer).

The importer (:func:`src.songs.utils.parse_sm`) converts ``.sm`` note grids into
``Step`` records whose ``time`` is in **seconds**. Export reverses that:

* seconds → beats via ``TimingEngine.beat_at`` (the exact inverse of the
  ``time_at`` the importer uses), so the round-trip is symmetric;
* beats → grid text via ``NoteData.from_notes``;
* metadata + every difficulty serialized into one ``SMSimfile``.

:func:`build_song_zip` wraps the ``.sm`` with the audio (and banner/jacket) into
a drop-in StepMania song folder.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from fractions import Fraction
from typing import Callable, Optional
from urllib.parse import quote

from simfile.notes import Note, NoteData, NoteType
from simfile.sm import SMChart, SMSimfile
from simfile.timing import Beat, TimingData
from simfile.timing.engine import TimingEngine

from src.songs.utils import COLUMN_TO_DIRECTION

logger = logging.getLogger(__name__)

# Arrow direction (game space) → StepMania dance-single column.
DIRECTION_TO_COLUMN: dict[str, int] = {
    d.value: col for col, d in COLUMN_TO_DIRECTION.items()
}

# Grid snap resolution: 48 rows/beat resolves 4th/8th/12th/16th/24th/32nd notes.
SNAP = 48

# Canonical StepMania difficulty slots.
_VALID_SLOTS = {"Beginner", "Easy", "Medium", "Hard", "Challenge", "Edit"}
_SLOT_ALIASES = {
    "basic": "Easy",
    "light": "Easy",
    "another": "Medium",
    "trick": "Medium",
    "standard": "Medium",
    "maniac": "Hard",
    "ssr": "Hard",
    "heavy": "Hard",
    "expert": "Challenge",
    "smaniac": "Challenge",
    "oni": "Challenge",
}


def _difficulty_slot(name: str) -> str:
    """Normalize a chart difficulty name to a StepMania slot."""
    if not name:
        return "Edit"
    canon = name.strip().title()
    if canon in _VALID_SLOTS:
        return canon
    return _SLOT_ALIASES.get(name.strip().lower(), "Edit")


def _snap_beat(beat: float) -> Beat:
    """Snap a float beat to the nearest 1/``SNAP`` and return a simfile Beat.

    Clamps below 0: notes can land slightly negative when an authored offset
    places beat 0 after audio start, and the grid encoder can't represent
    negative beats.
    """
    return Beat(Fraction(max(0, round(beat * SNAP)), SNAP))


def _steps_to_notes(steps: list[dict], beat_of: Callable[[float], float]) -> str:
    """Convert step dicts (seconds) into a serialized ``.sm`` note grid."""
    notes: list[Note] = []
    for step in steps:
        head = _snap_beat(beat_of(step["time"]))
        is_hold = step.get("type") == "hold" and step.get("hold_duration")
        tail: Optional[Beat] = None
        if is_hold:
            tail = _snap_beat(beat_of(step["time"] + step["hold_duration"]))
            if tail <= head:  # degenerate hold → at least one snap unit long
                tail = head + Beat(Fraction(1, SNAP))
        for arrow in step["arrows"]:
            col = DIRECTION_TO_COLUMN.get(arrow)
            if col is None:
                continue
            if is_hold:
                notes.append(Note(beat=head, column=col, note_type=NoteType.HOLD_HEAD))
                notes.append(Note(beat=tail, column=col, note_type=NoteType.TAIL))
            else:
                notes.append(Note(beat=head, column=col, note_type=NoteType.TAP))

    notes.sort(key=lambda n: (n.beat, n.column))
    return str(NoteData.from_notes(iter(notes), columns=4))


def _format_bpms(bpms: list) -> str:
    """``[[0, 120], [4, 134.922]]`` → ``0.000=120.000,4.000=134.922``."""
    if not bpms:
        bpms = [[0.0, 120.0]]
    return ",".join(f"{float(b):.3f}={float(v):.3f}" for b, v in bpms)


def build_simfile(meta: dict, charts: list[dict]) -> str:
    """Serialize a song + all its charts to ``.sm`` text.

    ``meta`` keys: ``title``, ``artist``, ``music`` (filename), ``offset``
    (seconds), ``bpms`` (``list[[beat, bpm]]``), optional ``banner``/``jacket``/
    ``background``, optional ``time_shift`` (seconds added to every step time
    before beat conversion — used to undo the baked ML ``timing_offset``).

    ``charts`` items: ``difficulty_name``, ``difficulty_level``, ``steps``,
    optional ``radar_values``.
    """
    sm = SMSimfile.blank()
    sm["TITLE"] = meta.get("title") or "Untitled"
    sm["ARTIST"] = meta.get("artist") or ""
    sm["MUSIC"] = meta.get("music") or ""
    sm["OFFSET"] = f"{float(meta.get('offset', 0.0)):.6f}"
    sm["BPMS"] = _format_bpms(meta.get("bpms") or [])
    if meta.get("banner"):
        sm["BANNER"] = meta["banner"]
    if meta.get("background"):
        sm["BACKGROUND"] = meta["background"]
    if meta.get("jacket"):
        sm["JACKET"] = meta["jacket"]
        sm["CDTITLE"] = meta.get("cdtitle", sm.get("CDTITLE", ""))

    # Inverse-timing engine built from the same BPMS/OFFSET the importer used.
    engine = TimingEngine(TimingData(sm))
    time_shift = float(meta.get("time_shift", 0.0))
    beat_of = lambda t: float(engine.beat_at(t + time_shift))  # noqa: E731

    for ch in charts:
        steps = ch.get("steps") or []
        if not steps:
            continue
        notes_str = _steps_to_notes(steps, beat_of)
        sc = SMChart.blank()
        sc.stepstype = "dance-single"
        sc.difficulty = _difficulty_slot(ch.get("difficulty_name", ""))
        sc.meter = str(int(ch.get("difficulty_level") or 1))
        radar = ch.get("radar_values")
        if radar:
            sc.radarvalues = ",".join(f"{float(v):.3f}" for v in radar)
        sc.notes = notes_str
        sm.charts.append(sc)

    return str(sm)


_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def sanitize_filename(name: str, default: str = "song") -> str:
    """Strip path-unsafe characters; collapse whitespace; never empty."""
    cleaned = _UNSAFE.sub("", (name or "").strip()).strip(". ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:120] or default


def content_disposition(filename: str) -> str:
    """Build an attachment ``Content-Disposition`` value safe for any title.

    HTTP headers are latin-1 only, but titles can contain arbitrary Unicode
    (curly quotes, accents, CJK). Emit an ASCII fallback plus an RFC 5987
    ``filename*`` so clients render the real name.
    """
    ascii_name = filename.encode("ascii", "ignore").decode("ascii").strip() or "song.zip"
    ascii_name = ascii_name.replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def build_song_zip(
    sm_text: str, folder_name: str, assets: dict[str, bytes]
) -> bytes:
    """Bundle ``<folder>/<folder>.sm`` + assets into a zip; return its bytes."""
    folder = sanitize_filename(folder_name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{folder}/{folder}.sm", sm_text)
        for name, data in assets.items():
            if data is not None:
                zf.writestr(f"{folder}/{name}", data)
    return buf.getvalue()
