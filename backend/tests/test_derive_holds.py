"""Unit tests for algorithmic hold derivation.

Holds are no longer predicted by the model — :meth:`MLChartGenerator._derive_holds`
promotes taps to holds from the harmonic-sustain envelope, bounded by what the
free foot can cover. These tests pin the three behaviours that motivated the
change.

Runs under pytest, or directly: ``python tests/test_derive_holds.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference import MLChartGenerator, FRAMES_PER_SECOND  # noqa: E402
from src.generation.constants import get_difficulty_config  # noqa: E402

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5s


def _generator() -> MLChartGenerator:
    """A generator with no model loaded — _derive_holds is pure post-processing."""
    return MLChartGenerator.__new__(MLChartGenerator)


def _beats(n: int = 64) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * BEAT


def _events(times, arities=None, pad_to=24, pad_from=40.0):
    """Build a time-sorted event list.

    ``pad_to`` appends distant filler notes so the hold budget
    (``round(hold_rate * len(note_events))``) is at least 1 — a handful of
    events would otherwise round down to zero holds and pass vacuously.
    Filler sits well after ``pad_from`` so it can't win a promotion slot.
    """
    arities = list(arities or [1] * len(times))
    times = list(times)
    n_pad = max(0, pad_to - len(times))
    for k in range(n_pad):
        times.append(pad_from + 0.13 * (k + 1))
        arities.append(1)
    return [
        {'frame': int(round(t * FRAMES_PER_SECOND)), 'time': float(t),
         'type': 'tap', 'confidence': 0.9, 'num_arrows': a}
        for t, a in zip(times, arities)
    ]


def _ringing(n_frames: int, decay_from: float = None) -> np.ndarray:
    """Harmonic envelope: flat 1.0, optionally collapsing to 0 after a time."""
    curve = np.ones(n_frames, dtype=np.float32)
    if decay_from is not None:
        curve[int(round(decay_from * FRAMES_PER_SECOND)):] = 0.0
    return curve


KNOBS = {'hold_rate': 0.5, 'avg_hold_beats': 4.0}


def test_promotes_tap_into_silence():
    """A ringing note with 4 beats of space becomes a hold."""
    gen = _generator()
    n_frames = int(20 * FRAMES_PER_SECOND)
    events = _events([1.0, 1.0 + 4 * BEAT, 1.0 + 8 * BEAT])

    n = gen._derive_holds(
        events, _ringing(n_frames), BPM, _beats(), n_frames,
        get_difficulty_config('hard'), KNOBS,
    )

    assert n >= 1, "expected at least one promotion"
    assert events[0]['type'] == 'hold'
    # The exact cell depends on the sustain-rank draw against HOLD_LENGTH_CDF,
    # so assert the invariants rather than a specific length.
    cfg = get_difficulty_config('hard')
    dur = events[0]['hold_duration']
    assert cfg.min_hold_duration <= dur <= cfg.max_hold_duration, dur
    assert any(abs(dur / BEAT - c) < 1e-6 for c in MLChartGenerator.HOLD_BEAT_CELLS), dur


def test_short_ring_is_rejected_not_inflated():
    """A 50ms ring stays a tap.

    The old sustain path applied ``max(min_hold_duration, ...)``, inflating a
    single-frame flicker into a 0.3-0.8s hold. Rejection is the fix.
    """
    gen = _generator()
    n_frames = int(20 * FRAMES_PER_SECOND)
    events = _events([1.0, 1.0 + 8 * BEAT])
    curve = _ringing(n_frames, decay_from=1.05)  # rings for 50ms

    n = gen._derive_holds(
        events, curve, BPM, _beats(), n_frames,
        get_difficulty_config('hard'), KNOBS,
    )

    assert n == 0, "a 50ms ring must not become a hold"
    assert all(e['type'] == 'tap' for e in events)
    assert 'hold_duration' not in events[0]


def test_trims_before_a_jump():
    """A jump needs both feet, so the hold must end before it."""
    gen = _generator()
    n_frames = int(20 * FRAMES_PER_SECOND)
    # Hold candidate at 1.0, a jump 4 beats later, plenty of ring throughout.
    events = _events([1.0, 1.0 + 4 * BEAT], arities=[1, 2])

    gen._derive_holds(
        events, _ringing(n_frames), BPM, _beats(), n_frames,
        get_difficulty_config('hard'), KNOBS,
    )

    assert events[0]['type'] == 'hold'
    end = events[0]['time'] + events[0]['hold_duration']
    assert end <= events[1]['time'], f"hold ends {end} but jump is at {events[1]['time']}"


def test_beginner_allows_no_notes_under_holds():
    """max_notes_under_hold=0 reduces the rule to 'holds live in rests'."""
    gen = _generator()
    n_frames = int(20 * FRAMES_PER_SECOND)
    # A note one beat after the candidate: fine for hard, fatal for beginner.
    events = _events([1.0, 1.0 + 1 * BEAT, 1.0 + 6 * BEAT])
    curve = _ringing(n_frames)

    beginner_events = [dict(e) for e in events]
    gen._derive_holds(
        beginner_events, curve, BPM, _beats(), n_frames,
        get_difficulty_config('beginner'), KNOBS,
    )
    # Only 0.5 beat of room before the next note → nothing fits.
    assert beginner_events[0]['type'] == 'tap'


def test_hold_under_stream_allowed_on_challenge():
    """Challenge permits a hold with the free foot tapping underneath."""
    gen = _generator()
    n_frames = int(30 * FRAMES_PER_SECOND)
    # Candidate, then 8th notes underneath (0.25s apart — clears jack_min_dt).
    times = [1.0] + [1.0 + (i + 1) * 0.5 * BEAT for i in range(8)]
    events = _events(times)

    gen._derive_holds(
        events, _ringing(n_frames), BPM, _beats(), n_frames,
        get_difficulty_config('challenge'), KNOBS,
    )

    assert events[0]['type'] == 'hold', "challenge should allow hold-under-stream"
    end = events[0]['time'] + events[0]['hold_duration']
    n_under = sum(1 for e in events[1:] if e['time'] < end)
    assert n_under > 0, "expected notes playing under the hold"


def test_durations_are_whole_beat_cells():
    """Tails land on the grid, so .sm export doesn't have to invent a snap."""
    gen = _generator()
    n_frames = int(40 * FRAMES_PER_SECOND)
    events = _events([1.0 + i * 8 * BEAT for i in range(4)])

    gen._derive_holds(
        events, _ringing(n_frames), BPM, _beats(), n_frames,
        get_difficulty_config('hard'), KNOBS,
    )

    holds = [e for e in events if e['type'] == 'hold']
    assert holds, "expected some holds"
    for h in holds:
        beats = h['hold_duration'] / BEAT
        assert any(abs(beats - c) < 1e-6 for c in MLChartGenerator.HOLD_BEAT_CELLS), (
            f"{beats} beats is not a quantization cell"
        )


if __name__ == "__main__":
    test_promotes_tap_into_silence()
    test_short_ring_is_rejected_not_inflated()
    test_trims_before_a_jump()
    test_beginner_allows_no_notes_under_holds()
    test_hold_under_stream_allowed_on_challenge()
    test_durations_are_whole_beat_cells()
    print("OK: all hold-derivation tests passed")
