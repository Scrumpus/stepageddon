"""
StepMania .sm/.ssc file parser using the simfile library.

Parses chart files into structured note data with accurate timing,
handling BPM changes, stops, warps, and offsets via simfile's TimingEngine.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import logging

import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.timing.engine import TimingEngine

logger = logging.getLogger(__name__)

ARROW_NAMES = ['left', 'down', 'up', 'right']

DIFFICULTY_MAP = {
    'beginner': 0,
    'easy': 1,
    'basic': 1,
    'medium': 2,
    'hard': 3,
    'challenge': 4,
    'expert': 4,
    'edit': 4,
}

# Map simfile NoteType to our label strings
NOTE_TYPE_MAP = {
    NoteType.TAP: 'tap',
    NoteType.HOLD_HEAD: 'hold_head',
    NoteType.TAIL: 'hold_tail',
    NoteType.ROLL_HEAD: 'roll_head',
}


@dataclass
class Note:
    time: float
    arrow: int          # 0-3 (L/D/U/R)
    note_type: str      # 'tap', 'hold_head', 'hold_tail', 'roll_head'


@dataclass
class NoteChart:
    game_type: str      # 'dance-single', 'dance-double', etc.
    difficulty: str     # 'Beginner', 'Easy', 'Medium', 'Hard', 'Challenge', 'Edit'
    difficulty_id: int  # 0-4 mapped
    meter: int          # numeric difficulty rating
    notes: List[Note] = field(default_factory=list)


@dataclass
class SmFile:
    title: str = ''
    artist: str = ''
    offset: float = 0.0
    primary_bpm: float = 120.0
    charts: List[NoteChart] = field(default_factory=list)


def parse_sm_file(filepath: str) -> Optional[SmFile]:
    """
    Parse a .sm or .ssc file into structured data.

    Uses simfile library for robust parsing of BPM changes,
    stops, warps, and beat-to-time conversion.

    Returns SmFile with header info and list of NoteCharts (dance-single only),
    or None if parsing fails.
    """
    try:
        sf = simfile.open(filepath)
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None

    timing = TimingData(sf)
    engine = TimingEngine(timing)

    # Extract primary BPM from first BPM entry
    primary_bpm = 120.0
    if timing.bpms and len(timing.bpms) > 0:
        try:
            primary_bpm = float(timing.bpms[0].value)
        except (IndexError, ValueError, TypeError):
            pass

    result = SmFile(
        title=sf.title or '',
        artist=sf.artist or '',
        offset=float(sf.offset) if sf.offset else 0.0,
        primary_bpm=primary_bpm,
    )

    for chart in sf.charts:
        # Only parse dance-single charts
        if chart.stepstype != 'dance-single':
            continue

        diff_key = chart.difficulty.lower().strip()
        difficulty_id = DIFFICULTY_MAP.get(diff_key, 2)

        try:
            meter = int(chart.meter)
        except (ValueError, TypeError):
            meter = 1

        # Parse notes using simfile's NoteData
        notes = []
        try:
            note_data = NoteData(chart)
            for note in note_data:
                # Only keep gameplay-relevant note types
                if note.note_type not in NOTE_TYPE_MAP:
                    continue

                time = engine.time_at(note.beat)
                notes.append(Note(
                    time=time,
                    arrow=note.column,
                    note_type=NOTE_TYPE_MAP[note.note_type],
                ))
        except Exception as e:
            logger.debug(f"Failed to parse notes for {sf.title} {chart.difficulty}: {e}")
            continue

        if not notes:
            continue

        result.charts.append(NoteChart(
            game_type=chart.stepstype,
            difficulty=chart.difficulty,
            difficulty_id=difficulty_id,
            meter=meter,
            notes=notes,
        ))

    if not result.charts:
        logger.debug(f"No dance-single charts found in {filepath}")
        return None

    return result
