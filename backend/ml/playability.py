"""Playability model + validator for generated step charts.

This module is the single source of truth for what makes a 4-panel (dance-single)
chart *physically playable* under natural two-foot footing. It is used two ways:

* **Validation** (`validate_chart`) — given a finished chart, prove that no
  unplayable pattern exists: no jack faster than a per-difficulty threshold, no
  concurrency above 2, no un-recoverable crossover, and no forced double-step.
* **Generation** (Phase 1 Commit 2) — the rewritten arrow assigner imports the
  same `PLAYABILITY_SPEC` and the same foot-flow primitives so the charts it emits
  are playable *by construction* and pass this validator.

Footing is not stored on a `Step` (a step only knows its arrows), so the validator
*infers* the most natural footing via a small dynamic program that minimises
crossovers and forced double-steps between "barrier" events (jumps and holds, which
fix the stance). A chart is flagged only when even the best footing still breaks a
hard rule — i.e. no playable footing exists.

Panels: LEFT, DOWN, UP, RIGHT. Horizontal coordinate (for crossover reasoning):
LEFT=-1, DOWN=0, UP=0, RIGHT=+1. Feet are "crossed" when the left foot is strictly
to the right of the right foot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Import lazily-friendly: these are plain enums with no heavy deps.
from src.charts.schemas import BeatSubdivision, Direction, Step, StepType


# ---------------------------------------------------------------------------
# Per-difficulty playability spec — the single source of truth.
# ---------------------------------------------------------------------------
# jack_min_dt        : minimum seconds between two hits on the SAME panel. A
#                      repeat faster than this is an unhittable jack.
# double_step_min_dt : minimum seconds for the same foot to move between two
#                      DIFFERENT panels (a forced double-step / cross-body reach).
# crossovers_allowed : whether the difficulty may ever place feet in a crossed
#                      stance at all.
# uncross_within     : max consecutive notes the stance may stay crossed before it
#                      must return to a neutral (un-crossed) stance.
# max_run_length     : stamina ceiling — longest run of ~evenly-spaced notes the
#                      difficulty may sustain (mirrors DifficultyConfig.max_stream_length).

@dataclass(frozen=True)
class PlayabilitySpec:
    jack_min_dt: float
    double_step_min_dt: float
    crossovers_allowed: bool
    uncross_within: int
    max_run_length: int


PLAYABILITY_SPEC: Dict[str, PlayabilitySpec] = {
    # Thresholds chosen so a difficulty's own min_gap comfortably clears its
    # jack/double-step floor; tuned during verification.
    'beginner':  PlayabilitySpec(0.30, 0.30, crossovers_allowed=False, uncross_within=0, max_run_length=0),
    'easy':      PlayabilitySpec(0.25, 0.25, crossovers_allowed=False, uncross_within=0, max_run_length=0),
    'medium':    PlayabilitySpec(0.22, 0.20, crossovers_allowed=True,  uncross_within=2, max_run_length=6),
    'hard':      PlayabilitySpec(0.15, 0.14, crossovers_allowed=True,  uncross_within=2, max_run_length=16),
    'challenge': PlayabilitySpec(0.12, 0.11, crossovers_allowed=True,  uncross_within=3, max_run_length=32),
}

MAX_CONCURRENT_ARROWS = 2


def spec_for(difficulty: str) -> PlayabilitySpec:
    return PLAYABILITY_SPEC.get(difficulty, PLAYABILITY_SPEC['medium'])


# ---------------------------------------------------------------------------
# Panel geometry
# ---------------------------------------------------------------------------

PANEL_X: Dict[Direction, int] = {
    Direction.LEFT: -1,
    Direction.DOWN: 0,
    Direction.UP: 0,
    Direction.RIGHT: 1,
}

Foot = str  # 'L' or 'R'


def is_crossed(left_panel: Direction, right_panel: Direction) -> bool:
    """Feet are crossed when the left foot sits strictly right of the right foot."""
    return PANEL_X[left_panel] > PANEL_X[right_panel]


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------

class ViolationType(str, Enum):
    CONCURRENCY = "concurrency"              # >2 arrows on screen at once
    FAST_JACK = "fast_jack"                  # same panel repeated too fast
    DOUBLE_STEP = "double_step"              # one foot forced to two panels too fast
    DISALLOWED_CROSSOVER = "disallowed_crossover"   # crossed stance where not allowed
    UNRESOLVED_CROSSOVER = "unresolved_crossover"    # crossed too many notes in a row
    JUMP_UNREACHABLE = "jump_unreachable"    # jump collides with a held panel / bad arity


@dataclass
class Violation:
    type: ViolationType
    index: int                 # step index in the chart
    time: float                # step time (seconds)
    detail: str = ""
    dt: Optional[float] = None  # relevant interval, when applicable

    def __str__(self) -> str:
        dt = f" dt={self.dt*1000:.0f}ms" if self.dt is not None else ""
        return f"[{self.type.value}] step {self.index} @ {self.time:.3f}s{dt} — {self.detail}"


# ---------------------------------------------------------------------------
# Normalisation — accept Step objects or serialized dicts.
# ---------------------------------------------------------------------------

@dataclass
class _Note:
    index: int
    time: float
    arrows: List[Direction]
    is_hold: bool
    hold_end: float  # time + hold_duration, or == time for taps


_STR_TO_DIR = {d.value: d for d in Direction}


def _to_notes(steps: Sequence[Union[Step, dict]]) -> List[_Note]:
    notes: List[_Note] = []
    for i, s in enumerate(steps):
        if isinstance(s, Step):
            t = float(s.time)
            arrows = list(s.arrows)
            is_hold = s.step_type == StepType.HOLD
            dur = float(s.hold_duration) if (is_hold and s.hold_duration) else 0.0
        else:  # serialized dict
            t = float(s['time'])
            arrows = [a if isinstance(a, Direction) else _STR_TO_DIR[a] for a in s['arrows']]
            is_hold = s.get('type') == 'hold' or s.get('type') == StepType.HOLD
            dur = float(s.get('hold_duration') or 0.0)
        notes.append(_Note(i, t, arrows, is_hold, t + dur))
    notes.sort(key=lambda n: (n.time, n.index))
    return notes


# ---------------------------------------------------------------------------
# Foot-flow inference (min-crossover / min-double-step DP over single runs)
# ---------------------------------------------------------------------------

# Cost weights: forbid double-steps hardest, heavily penalise a crossed run that
# exceeds the difficulty budget (so the DP finds a min-*max*-run footing, not just
# a min-*total*-cross one), then keep crossed stance short, then minor movement.
_W_DOUBLE_STEP = 100.0        # fast same-foot move across panels — unplayable
_W_OVER_BUDGET = 60.0         # crossed run beyond the difficulty budget
_W_SLOW_DOUBLE_STEP = 5.0     # slow same-foot shuffle/footswitch — legal, mildly costly
_W_CROSS_STEP = 4.0
_W_MOVE = 0.1


@dataclass
class _Stance:
    left_panel: Direction = Direction.LEFT
    right_panel: Direction = Direction.RIGHT
    last_foot: Optional[Foot] = 'R'   # so the first single naturally alternates to L
    crossed_run: int = 0

    def key(self) -> Tuple:
        return (self.left_panel, self.right_panel, self.last_foot, self.crossed_run)


def _panel_of(stance: _Stance, foot: Foot) -> Direction:
    return stance.left_panel if foot == 'L' else stance.right_panel


def _apply_single(stance: _Stance, foot: Foot, panel: Direction) -> _Stance:
    left = panel if foot == 'L' else stance.left_panel
    right = panel if foot == 'R' else stance.right_panel
    crossed = is_crossed(left, right)
    return _Stance(left, right, foot, stance.crossed_run + 1 if crossed else 0)


class FootFlowInference:
    """Infer the most natural footing for a chart via a barrier-segmented DP.

    Jumps and holds are *barriers*: they fix both feet (a jump) or pin one foot
    (a hold), so the DP collapses to a single committed stance there and the
    single-note runs between barriers are optimised independently.
    """

    def __init__(self, difficulty: str):
        self.difficulty = difficulty
        self.spec = spec_for(difficulty)

    def run(self, notes: List[_Note]) -> Tuple[List[Optional[Foot]], List[_Stance]]:
        """Return per-note assigned foot (None for jumps) and post-note stance."""
        # Frontier: map stance.key() -> (cost, stance, path_of_feet, path_of_stances)
        start = _Stance()
        frontier: Dict[Tuple, Tuple[float, _Stance, List[Optional[Foot]], List[_Stance]]] = {
            start.key(): (0.0, start, [], [])
        }
        pinned: Optional[Tuple[Foot, Direction, float]] = None  # (foot, panel, until)
        prev_t: Optional[float] = None

        for n in notes:
            # Expire a pin whose hold has ended before this note.
            if pinned is not None and pinned[2] <= n.time + 1e-6:
                pinned = None

            if len(n.arrows) >= 2:
                frontier = self._commit_jump(frontier, n)
                pinned = None  # a jump re-plants both feet
                prev_t = n.time
                continue

            dt = (n.time - prev_t) if prev_t is not None else 1e9
            prev_t = n.time
            panel = n.arrows[0]
            if n.is_hold:
                # A hold is a barrier: commit the best stance, plant the hold foot,
                # and pin it for the hold's duration.
                frontier, foot = self._commit_single_barrier(frontier, n, panel, pinned)
                pinned = (foot, panel, n.hold_end)
                continue

            # Ordinary single: expand the DP frontier.
            frontier = self._expand_single(frontier, n, panel, pinned, dt)

        # Reconstruct the min-cost path.
        best = min(frontier.values(), key=lambda v: v[0])
        return best[2], best[3]

    def _expand_single(self, frontier, n: _Note, panel: Direction, pinned, dt: float):
        new_frontier: Dict[Tuple, Tuple] = {}
        for cost, stance, feet, stances in frontier.values():
            for foot in ('L', 'R'):
                if pinned is not None and pinned[0] == foot and pinned[1] != panel:
                    continue  # this foot is pinned to a different panel by a hold
                step_cost = self._single_cost(stance, foot, panel, dt)
                ns = _apply_single(stance, foot, panel)
                nkey = ns.key()
                total = cost + step_cost
                prev = new_frontier.get(nkey)
                if prev is None or total < prev[0]:
                    new_frontier[nkey] = (
                        total, ns, feet + [foot], stances + [ns],
                    )
        if not new_frontier:  # fully pinned & unreachable — force the free foot
            for cost, stance, feet, stances in frontier.values():
                foot = 'R' if (pinned and pinned[0] == 'L') else 'L'
                ns = _apply_single(stance, foot, panel)
                new_frontier[ns.key()] = (
                    cost + _W_DOUBLE_STEP, ns, feet + [foot], stances + [ns],
                )
                break
        return new_frontier

    def _single_cost(self, stance: _Stance, foot: Foot, panel: Direction, dt: float) -> float:
        cost = 0.0
        old_panel = _panel_of(stance, foot)
        # Double-step: same foot as last note, moving to a different panel. A slow
        # double-step (a legal footswitch/shuffle) is cheap; a fast one is forbidden.
        if stance.last_foot == foot and old_panel != panel:
            cost += _W_DOUBLE_STEP if dt < self.spec.double_step_min_dt else _W_SLOW_DOUBLE_STEP
        ns = _apply_single(stance, foot, panel)
        if is_crossed(ns.left_panel, ns.right_panel):
            cost += _W_CROSS_STEP
            # A crossed run beyond the difficulty budget is much worse than a few
            # extra isolated crossovers — steer the DP toward a spread footing.
            if ns.crossed_run > self.spec.uncross_within:
                cost += _W_OVER_BUDGET
        cost += _W_MOVE * abs(PANEL_X[old_panel] - PANEL_X[panel])
        return cost

    def _commit_single_barrier(self, frontier, n: _Note, panel: Direction, pinned):
        # A hold is naturally taken by the foot that OWNS its panel (LEFT/DOWN -> left,
        # UP/RIGHT -> right), which keeps the hold un-crossed — matching how the
        # generator foots holds. Fall back to the other foot only if the owner is
        # already pinned by another active hold.
        cost0, stance0, feet0, stances0 = min(frontier.values(), key=lambda v: v[0])
        owner = 'L' if panel in (Direction.LEFT, Direction.DOWN) else 'R'
        if pinned is not None and pinned[0] == owner:
            foot = 'R' if owner == 'L' else 'L'
        else:
            foot = owner
        ns = _apply_single(stance0, foot, panel)
        committed = {ns.key(): (cost0, ns, feet0 + [foot], stances0 + [ns])}
        return committed, foot

    def _commit_jump(self, frontier, n: _Note):
        cost0, stance0, feet0, stances0 = min(frontier.values(), key=lambda v: v[0])
        a, b = n.arrows[0], n.arrows[1]
        # Assign lower-x panel to the left foot, higher-x to the right foot.
        if PANEL_X[a] <= PANEL_X[b]:
            left, right = a, b
        else:
            left, right = b, a
        ns = _Stance(left, right, last_foot=None, crossed_run=0)
        return {ns.key(): (cost0, ns, feet0 + [None], stances0 + [ns])}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chart(
    steps: Sequence[Union[Step, dict]],
    difficulty: str,
) -> List[Violation]:
    """Return every playability violation in `steps` for the given difficulty.

    An empty list means the chart is playable under natural two-foot footing.
    """
    spec = spec_for(difficulty)
    notes = _to_notes(steps)
    violations: List[Violation] = []
    if not notes:
        return violations

    # --- Concurrency (footing-independent) ---
    # Sweep active hold intervals; count overlapping arrows at each note onset.
    active_ends: List[float] = []
    for n in notes:
        active_ends = [e for e in active_ends if e > n.time + 1e-6]
        concurrent = len(active_ends) + len(n.arrows)
        if concurrent > MAX_CONCURRENT_ARROWS:
            violations.append(Violation(
                ViolationType.CONCURRENCY, n.index, n.time,
                detail=f"{concurrent} arrows on screen (>{MAX_CONCURRENT_ARROWS})",
            ))
        if n.is_hold:
            for _ in n.arrows:
                active_ends.append(n.hold_end)

    # --- Fast jacks (footing-independent) ---
    # Consecutive SINGLE notes on the same panel closer than jack_min_dt.
    prev_single: Optional[_Note] = None
    for n in notes:
        if len(n.arrows) == 1:
            if prev_single is not None and prev_single.arrows[0] == n.arrows[0]:
                dt = n.time - prev_single.time
                if dt < spec.jack_min_dt:
                    violations.append(Violation(
                        ViolationType.FAST_JACK, n.index, n.time,
                        detail=f"{n.arrows[0].value} repeated at "
                               f"{dt*1000:.0f}ms (< {spec.jack_min_dt*1000:.0f}ms)",
                        dt=dt,
                    ))
            prev_single = n
        else:
            prev_single = None

    # --- Jump reachability (footing-independent) ---
    active_ends = []
    for n in notes:
        active_ends_now = [e for e in active_ends if e > n.time + 1e-6]
        if len(n.arrows) >= 2:
            if len(set(n.arrows)) < len(n.arrows):
                violations.append(Violation(
                    ViolationType.JUMP_UNREACHABLE, n.index, n.time,
                    detail="jump repeats a panel",
                ))
        if n.is_hold:
            active_ends = active_ends_now + [n.hold_end] * len(n.arrows)
        else:
            active_ends = active_ends_now

    # --- Crossovers & forced double-steps (footing-dependent) ---
    inf = FootFlowInference(difficulty)
    feet, stances = inf.run(notes)

    crossed_run = 0
    run_flagged = False
    prev_foot: Optional[Foot] = None
    prev_note: Optional[_Note] = None
    for n, foot, stance in zip(notes, feet, stances):
        crossed = is_crossed(stance.left_panel, stance.right_panel)
        if crossed:
            crossed_run += 1
            if not spec.crossovers_allowed and not run_flagged:
                violations.append(Violation(
                    ViolationType.DISALLOWED_CROSSOVER, n.index, n.time,
                    detail="crossed stance not allowed at this difficulty",
                ))
                run_flagged = True
            elif spec.crossovers_allowed and crossed_run > spec.uncross_within and not run_flagged:
                violations.append(Violation(
                    ViolationType.UNRESOLVED_CROSSOVER, n.index, n.time,
                    detail=f"crossed for {crossed_run} notes "
                           f"(> uncross_within={spec.uncross_within})",
                ))
                run_flagged = True
        else:
            crossed_run = 0
            run_flagged = False

        # Forced double-step: same foot two notes running, different panel, too fast.
        if (foot is not None and prev_foot is not None and foot == prev_foot
                and prev_note is not None):
            old_panel = None  # different-panel check via the inferred stance path
            dt = n.time - prev_note.time
            # If the same foot was reused on a different panel this fast, flag it.
            if dt < spec.double_step_min_dt and n.arrows[0] != prev_note.arrows[0]:
                violations.append(Violation(
                    ViolationType.DOUBLE_STEP, n.index, n.time,
                    detail=f"same foot forced across panels at {dt*1000:.0f}ms",
                    dt=dt,
                ))
        if foot is not None:
            prev_foot = foot
            prev_note = n
        else:  # jump resets alternation
            prev_foot = None
            prev_note = None

    violations.sort(key=lambda v: (v.index, v.type.value))
    return violations


# ---------------------------------------------------------------------------
# Summary helpers (used by the harness)
# ---------------------------------------------------------------------------

@dataclass
class ChartReport:
    difficulty: str
    step_count: int
    violations: List[Violation] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.violations:
            out[v.type.value] = out.get(v.type.value, 0) + 1
        return out

    @property
    def total(self) -> int:
        return len(self.violations)


def report_chart(steps: Sequence[Union[Step, dict]], difficulty: str) -> ChartReport:
    return ChartReport(
        difficulty=difficulty,
        step_count=len(steps),
        violations=validate_chart(steps, difficulty),
    )
