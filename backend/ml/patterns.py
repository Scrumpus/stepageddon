"""
DDR pattern catalog — named arrow sequences used in 4-panel dance games.

These patterns are consumed by FootStateArrowAssigner at inference time to
produce idiomatic step charts rather than purely random foot-alternating notes.
All arrow indices follow the canonical ordering used throughout the codebase:

    Direction.LEFT  = 0
    Direction.DOWN  = 1
    Direction.UP    = 2
    Direction.RIGHT = 3

Pattern names follow DDR community convention (ref: ddrcommunity.com basic
patterns guide, stepmania.fandom.com/wiki/Pattern_for_Dance).
"""

from typing import Dict, List, Tuple

from src.charts.schemas import Direction

# ---------------------------------------------------------------------------
# Arrow index → Direction helpers (avoid circular imports with inference.py)
# ---------------------------------------------------------------------------
_ARROW_INDICES: Dict[Direction, int] = {
    Direction.LEFT: 0,
    Direction.DOWN: 1,
    Direction.UP: 2,
    Direction.RIGHT: 3,
}

_ARROW_FROM_INDEX: Dict[int, Direction] = {v: k for k, v in _ARROW_INDICES.items()}


def arrow_index(d: Direction) -> int:
    return _ARROW_INDICES[d]


def arrow_from_index(i: int) -> Direction:
    return _ARROW_FROM_INDEX[i]


# ---------------------------------------------------------------------------
# Stream patterns — 4-note repeating sequences that form the backbone of
# longer runs.  Each pattern is a list of arrow indices.
# ---------------------------------------------------------------------------
STREAM_PATTERNS: List[List[int]] = [
    [0, 2, 1, 3],   # L U D R — standard weave
    [3, 1, 2, 0],   # R D U L — reverse weave
    [0, 1, 2, 3],   # L D U R — staircase up
    [3, 2, 1, 0],   # R U D L — staircase down
    [0, 3, 1, 2],   # L R D U — wide crossover
    [1, 2, 0, 3],   # D U L R — inside-out
    [0, 1, 3, 2],   # L D R U — skip-up variant
    [2, 3, 1, 0],   # U R D L — skip-down variant
    [0, 2, 3, 1],   # L U R D — alternating cross
    [1, 3, 0, 2],   # D R L U — alternating cross reverse
]

# ---------------------------------------------------------------------------
# Staircase patterns — full 4-note runs across all panels in order.
# ---------------------------------------------------------------------------
STAIRCASE_PATTERNS: Dict[str, List[int]] = {
    'up':   [0, 1, 2, 3],   # L D U R
    'down': [3, 2, 1, 0],   # R U D L
    'up_alt':   [1, 2, 3, 0],   # D U R L
    'down_alt': [2, 1, 0, 3],   # U D L R
}

# ---------------------------------------------------------------------------
# Candle patterns — 3-note sequences where the foot on step 1 crosses the
# center panel to reach step 3. Named because a candle on the center panel
# would be knocked over.
#
# Four canonical variants:
#   ULD — Up→Left→Down   (right foot starts on Up, left crosses to Down)
#   DRU — Down→Right→Up  (reverse)
#   DLD — Down→Left→Down (same-foot crosses center twice)
#   URU — Up→Right→Up    (reverse)
# ---------------------------------------------------------------------------
CANDLE_PATTERNS: List[List[int]] = [
    [2, 0, 1],   # U L D
    [1, 3, 2],   # D R U
    [1, 0, 1],   # D L D
    [2, 3, 2],   # U R U
]

# ---------------------------------------------------------------------------
# Spin patterns — 5-note sequences that rotate the player 180° or 360°.
#   L D R U L  — full spin left
#   R D L U R  — full spin right
#   U L D R U  — spin starting from up
#   D R U L D  — spin starting from down
# ---------------------------------------------------------------------------
SPIN_PATTERNS: List[List[int]] = [
    [0, 1, 3, 2, 0],   # L D R U L
    [3, 1, 0, 2, 3],   # R D L U R
    [2, 0, 1, 3, 2],   # U L D R U
    [1, 3, 2, 0, 1],   # D R U L D
]

# ---------------------------------------------------------------------------
# Drill patterns — repeated 1/16th-note alternation on two arrows.
# These are 2-note frames; the assigner repeats them N times.
# ---------------------------------------------------------------------------
DRILL_PATTERNS: List[Tuple[int, int]] = [
    (0, 3),   # L R
    (1, 2),   # D U
    (0, 2),   # L U
    (1, 3),   # D R
]

# ---------------------------------------------------------------------------
# Jump patterns — canonical two-arrow simultaneous hits.
# ---------------------------------------------------------------------------
JUMP_PATTERNS: List[Tuple[Direction, Direction]] = [
    (Direction.LEFT, Direction.RIGHT),
    (Direction.DOWN, Direction.UP),
    (Direction.LEFT, Direction.UP),
    (Direction.DOWN, Direction.RIGHT),
]

# ---------------------------------------------------------------------------
# Gallop template — two quick 1/16th notes on different arrows that form a
# natural alternating pair. These are used when the beat subdivision grid
# places two notes at 1/16th spacing.
# ---------------------------------------------------------------------------
GALLOP_PAIRS: List[Tuple[int, int]] = [
    (0, 1),   # L D
    (1, 3),   # D R
    (3, 2),   # R U
    (2, 0),   # U L
    (1, 0),   # D L (reverse)
    (3, 1),   # R D (reverse)
    (2, 3),   # U R (reverse)
    (0, 2),   # L U (reverse)
]
