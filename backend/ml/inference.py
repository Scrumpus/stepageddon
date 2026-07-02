"""
Inference module: generate step charts from audio using the v7 hybrid model.

The model emits three dense per-frame channels (onset / sustain / intensity).
Algorithmic post-processing — driven by selectable, auto-clustered style
profiles — decides arrows, jump vs. tap, hold start/end, and jumpholds.
"""

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import librosa

from ml.model import StepChartModel, ARCH_VERSION
from ml.dataset import DEFAULT_DENSITY_BY_ID, DENSITY_MEAN, DENSITY_STD
from ml.prepare_data import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, FRAMES_PER_SECOND,
    DB_MIN, DB_MAX, DB_RANGE,
    N_FEAT_CHANNELS, N_SPECTRAL_CONTRAST,
)
from ml.patterns import (
    STREAM_PATTERNS, JUMP_PATTERNS,
    CANDLE_PATTERNS, SPIN_PATTERNS,
    DRILL_PATTERNS, GALLOP_PAIRS,
    arrow_from_index,
)
from src.charts.schemas import (
    Chart, Step, StepType, Direction, BeatSubdivision,
)
from src.generation.constants import get_difficulty_config
from ml.dataset import ARROW_CLASS_CODES, ARROW_BOS_CLASS

logger = logging.getLogger(__name__)

ARROW_DIRECTIONS = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]

APP_DIFFICULTY_MAP = {
    'beginner': 0,
    'easy': 1,
    'medium': 2,
    'hard': 3,
    'challenge': 4,
}

@dataclass
class _AudioAnalysis:
    """Per-song features that are reused across every difficulty in a
    multi-difficulty generation. Created once by :meth:`MLChartGenerator._analyze_audio`
    and consumed by :meth:`MLChartGenerator._chart_for_difficulty`."""

    duration: float
    feats_whitened: np.ndarray
    tempo: float
    beat_times: np.ndarray
    timing_offset: float
    rms_norm: np.ndarray
    centroid_norm: np.ndarray
    harm_rms: np.ndarray
    audio_seed: int
    # Dominant chroma pitch class per frame (0=C .. 11=B), or -1 where the frame
    # is not confidently pitched (percussive/quiet). Drives music-aware arrow
    # selection in FootStateArrowAssigner. Shape [T], int16.
    pitch_class: np.ndarray


# Defaults for style-profile post-processing knobs. These are used when a
# profile doesn't explicitly carry the field (e.g. a hand-edited profile, or
# the legacy "auto" path before clustering has run). Conservative values that
# work on all difficulties.
DEFAULT_STYLE_KNOBS: Dict[str, float] = {
    'jump_threshold': 0.05,       # min intensity for a peak to be jump-eligible
    'target_jump_rate': 0.05,     # fraction of peaks to mark as jumps (top-K)
    'jumphold_rate': 0.0,         # 0 disables jumpholds
    'jumphold_rms_thr': 0.7,
    'jack_rate': 0.05,
    'crossover_rate': 0.10,
    'stream_preference': 0.5,
    'candle_rate': 0.05,          # probability of candle pattern injection
    'spin_rate': 0.02,            # probability of spin pattern injection
}


# Music-aware arrow selection: map each of the 12 pitch classes to a panel, in
# three-class blocks that align with the foot regions (LEFT/DOWN = left foot,
# UP/RIGHT = right foot). Ascending pitch therefore drifts left-foot → right-
# foot, mirroring pitch height → pad position. A given note maps to a stable
# panel, so a repeated melodic phrase yields a repeated arrow phrase for free.
_PITCH_CLASS_TO_PANEL = {
    0: Direction.LEFT,  1: Direction.LEFT,  2: Direction.LEFT,     # C  C# D
    3: Direction.DOWN,  4: Direction.DOWN,  5: Direction.DOWN,     # D# E  F
    6: Direction.UP,    7: Direction.UP,    8: Direction.UP,       # F# G  G#
    9: Direction.RIGHT, 10: Direction.RIGHT, 11: Direction.RIGHT,  # A  A# B
}
# Pitch-class "center" of each panel, for nearest-panel fallback (circular).
_PANEL_PITCH_CENTER = {
    Direction.LEFT: 1, Direction.DOWN: 4, Direction.UP: 7, Direction.RIGHT: 10,
}


def _pitch_class_distance(a: int, b: int) -> int:
    """Circular distance between two pitch classes on the 12-tone wheel."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


class FootStateArrowAssigner:
    """
    Assigns arrows to note events using foot-state tracking, a seeded PRNG,
    and selectable style preferences (jack_rate, crossover_rate,
    stream_preference).

    The crossover/stream/jack branches existed in v6; what's new in v7 is
    that the style profile drives the rates directly, replacing the
    energy-only thresholds.
    """

    LEFT_FOOT_PANELS = [Direction.LEFT, Direction.DOWN]
    RIGHT_FOOT_PANELS = [Direction.UP, Direction.RIGHT]

    ALL_PANELS = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]

    # Pattern catalogs imported from ml.patterns
    STREAM_PATTERNS = STREAM_PATTERNS
    JUMP_PATTERNS = JUMP_PATTERNS
    CANDLE_PATTERNS = CANDLE_PATTERNS
    SPIN_PATTERNS = SPIN_PATTERNS

    def __init__(
        self,
        seed: int = 0,
        allowed_patterns: Optional[List[str]] = None,
        jack_rate: float = 0.05,
        crossover_rate: float = 0.10,
        stream_preference: float = 0.5,
        candle_rate: float = 0.05,
        spin_rate: float = 0.02,
    ):
        self.rng = random.Random(seed)
        self.allowed_patterns = set(allowed_patterns or ['single'])
        self.jack_rate = float(np.clip(jack_rate, 0.0, 1.0))
        self.crossover_rate = float(np.clip(crossover_rate, 0.0, 1.0))
        self.stream_preference = float(np.clip(stream_preference, 0.0, 1.0))
        self.candle_rate = float(np.clip(candle_rate, 0.0, 1.0))
        self.spin_rate = float(np.clip(spin_rate, 0.0, 1.0))
        self.reset()

    def reset(self):
        self.last_foot = 'right'
        self.left_pos = Direction.LEFT
        self.right_pos = Direction.RIGHT
        # Per-arrow active hold map: arrow -> end_time. Replaces the v6
        # singleton (held_foot, hold_end_time) so jumpholds and back-to-back
        # holds don't overwrite each other's state.
        self.active_holds: Dict[Direction, float] = {}
        self._step_count = 0
        self._stream_pattern: Optional[List[int]] = None
        self._stream_idx: int = 0
        self._last_arrow: Optional[Direction] = None
        # Pitch class of the previous single tap, for melodic-jack detection.
        self._last_pitch_class: int = -1
        # Consecutive melodic jacks so far, capped so a long repeated-note run
        # eventually breaks to alternation instead of an endless jack.
        self._melodic_jack_run: int = 0
        # Candle state: preloaded sequence of 3 arrow indices (or None).
        self._candle_pattern: Optional[List[int]] = None
        self._candle_idx: int = 0
        # Spin state: preloaded sequence of 5 arrow indices (or None).
        self._spin_pattern: Optional[List[int]] = None
        self._spin_idx: int = 0
        # Recent arrow history for pattern detection (last 4 single-tap arrows).
        self._arrow_history: List[Direction] = []

    def _purge_holds(self, time: float) -> None:
        if not self.active_holds:
            return
        self.active_holds = {
            a: e for a, e in self.active_holds.items() if e > time
        }

    def _held_arrows(self, time: float) -> Set[Direction]:
        self._purge_holds(time)
        return set(self.active_holds.keys())

    def _held_feet(self, time: float) -> Set[str]:
        feet: Set[str] = set()
        for a in self._held_arrows(time):
            if a in self.LEFT_FOOT_PANELS:
                feet.add('left')
            if a in self.RIGHT_FOOT_PANELS:
                feet.add('right')
        return feet

    def _pick_foot(self, time: float) -> str:
        held_feet = self._held_feet(time)
        # Prefer a free foot. If both held (jumphold + tap case), fall back
        # to alternation; the tap is non-claiming so this is harmless.
        if 'left' in held_feet and 'right' not in held_feet:
            return 'right'
        if 'right' in held_feet and 'left' not in held_feet:
            return 'left'
        return 'right' if self.last_foot == 'left' else 'left'

    def _panels_for_foot(self, foot: str) -> List[Direction]:
        return self.LEFT_FOOT_PANELS if foot == 'left' else self.RIGHT_FOOT_PANELS

    def _update_foot(self, foot: str, arrow: Direction):
        if foot == 'left':
            self.left_pos = arrow
        else:
            self.right_pos = arrow
        self.last_foot = foot
        self._last_arrow = arrow

    def _panel_weights(
        self,
        panels: List[Direction],
        current_pos: Direction,
        brightness: float,
        pitch_class: int,
    ) -> List[float]:
        """Weights over the chosen foot's two panels [panels[0], panels[1]].

        Pitch-aware when a confident pitch class is available: strongly prefer
        the panel this note maps to (or, if that panel is on the other foot, the
        nearer of the two by pitch center). Falls back to the original
        brightness heuristic for unpitched frames. A small stickiness term still
        nudges away from the panel the foot is already on.
        """
        if pitch_class is not None and pitch_class >= 0:
            pref = _PITCH_CLASS_TO_PANEL[pitch_class]
            if pref == panels[0]:
                weights = [0.85, 0.15]
            elif pref == panels[1]:
                weights = [0.15, 0.85]
            else:
                d0 = _pitch_class_distance(pitch_class, _PANEL_PITCH_CENTER[panels[0]])
                d1 = _pitch_class_distance(pitch_class, _PANEL_PITCH_CENTER[panels[1]])
                weights = [0.65, 0.35] if d0 <= d1 else [0.35, 0.65]
        elif brightness > 0.6:
            weights = [0.7, 0.3]
        elif brightness < 0.4:
            weights = [0.3, 0.7]
        else:
            weights = [0.5, 0.5]

        if current_pos == panels[0]:
            weights[1] += 0.2
        else:
            weights[0] += 0.2
        return weights

    def assign_single(self, time: float, energy: float = 0.5,
                       brightness: float = 0.5, pitch_class: int = -1) -> Direction:
        # Record this onset's pitch for the next call's melodic-jack check, and
        # keep the previous value to test note repetition below.
        prev_pitch_class = self._last_pitch_class
        self._last_pitch_class = pitch_class

        # Active stream — follow the pattern. Must not emit a stream arrow
        # onto a panel that is still held from a prior hold_start.
        if self._stream_pattern is not None:
            arrow_idx = self._stream_pattern[self._stream_idx]
            self._stream_idx += 1
            if self._stream_idx >= len(self._stream_pattern):
                self._stream_pattern = None
                self._stream_idx = 0
            arrow = self.ALL_PANELS[arrow_idx]
            if arrow in self._held_arrows(time):
                free = [p for p in self.ALL_PANELS if p not in self.active_holds]
                if free:
                    arrow = free[0]
                else:
                    # All four held — abort the stream.
                    self._stream_pattern = None
                    self._stream_idx = 0
            foot = self._pick_foot(time)
            self._update_foot(foot, arrow)
            self._step_count += 1
            return arrow

        # Jack branch: with profile-controlled probability, repeat the last
        # arrow rather than alternating feet. Must not jack onto an arrow
        # that is still held from a prior hold_start.
        held_now = self._held_arrows(time)
        if (
            self._last_arrow is not None
            and self.jack_rate > 0.0
            and self.rng.random() < self.jack_rate
            and not self._held_feet(time)
            and self._last_arrow not in held_now
        ):
            self._step_count += 1
            arrow = self._last_arrow
            # Don't flip feet for a jack — same foot taps again.
            self._update_foot(self.last_foot, arrow)
            self._arrow_history.append(arrow)
            if len(self._arrow_history) > 5:
                self._arrow_history = self._arrow_history[-5:]
            return arrow

        # Melodic-jack branch: a repeated note (same pitch class as the previous
        # tap) is charted as a jack — repeat the same panel — the way human
        # authors handle held/repeated melody notes. Music-driven, not random.
        # Capped so a long repeated-note run breaks to alternation eventually.
        if (
            pitch_class >= 0
            and prev_pitch_class == pitch_class
            and self._melodic_jack_run < 3
            and self._last_arrow is not None
            and not self._held_feet(time)
            and self._last_arrow not in held_now
        ):
            self._step_count += 1
            self._melodic_jack_run += 1
            arrow = self._last_arrow
            self._update_foot(self.last_foot, arrow)
            self._arrow_history.append(arrow)
            if len(self._arrow_history) > 5:
                self._arrow_history = self._arrow_history[-5:]
            return arrow
        self._melodic_jack_run = 0

        # ---- Candle pattern branch ----
        # Follow an active candle sequence if one is in progress.
        if self._candle_pattern is not None:
            arrow_idx = self._candle_pattern[self._candle_idx]
            self._candle_idx += 1
            if self._candle_idx >= len(self._candle_pattern):
                self._candle_pattern = None
                self._candle_idx = 0
            arrow = self.ALL_PANELS[arrow_idx]
            if arrow in held_now:
                arrow = self._fallback_arrow(held_now)
                self._candle_pattern = None
                self._candle_idx = 0
            foot = self._pick_foot(time)
            self._update_foot(foot, arrow)
            self._step_count += 1
            self._arrow_history.append(arrow)
            if len(self._arrow_history) > 5:
                self._arrow_history = self._arrow_history[-5:]
            return arrow

        # ---- Spin pattern branch ----
        if self._spin_pattern is not None:
            arrow_idx = self._spin_pattern[self._spin_idx]
            self._spin_idx += 1
            if self._spin_idx >= len(self._spin_pattern):
                self._spin_pattern = None
                self._spin_idx = 0
            arrow = self.ALL_PANELS[arrow_idx]
            if arrow in held_now:
                arrow = self._fallback_arrow(held_now)
                self._spin_pattern = None
                self._spin_idx = 0
            foot = self._pick_foot(time)
            self._update_foot(foot, arrow)
            self._step_count += 1
            self._arrow_history.append(arrow)
            if len(self._arrow_history) > 5:
                self._arrow_history = self._arrow_history[-5:]
            return arrow

        # ---- Pattern-start detection ----
        # If no active pattern and not holding anything, try to start a candle
        # or spin based on recent arrow history.
        if not held_now and not self._held_feet(time):
            hist = self._arrow_history
            # Try candle start: need at least 2 recent taps to form a prefix.
            if (
                len(hist) >= 2
                and self.candle_rate > 0.0
                and self.rng.random() < self.candle_rate
            ):
                last2 = (self.ALL_PANELS.index(hist[-2]),
                          self.ALL_PANELS.index(hist[-1]))
                # Check if last 2 arrows match the start of any candle pattern.
                match = self._match_pattern_prefix(last2, self.CANDLE_PATTERNS, 2)
                if match is not None:
                    self._candle_pattern = list(match)
                    self._candle_idx = 2  # first two already emitted
                    arrow_idx = self._candle_pattern[self._candle_idx]
                    self._candle_idx += 1
                    if self._candle_idx >= len(self._candle_pattern):
                        self._candle_pattern = None
                        self._candle_idx = 0
                    arrow = self.ALL_PANELS[arrow_idx]
                    foot = self._pick_foot(time)
                    self._update_foot(foot, arrow)
                    self._step_count += 1
                    self._arrow_history.append(arrow)
                    if len(self._arrow_history) > 5:
                        self._arrow_history = self._arrow_history[-5:]
                    return arrow

            # Try spin start: need at least 3 recent taps.
            if (
                len(hist) >= 3
                and self.spin_rate > 0.0
                and self.rng.random() < self.spin_rate
            ):
                last3 = (self.ALL_PANELS.index(hist[-3]),
                          self.ALL_PANELS.index(hist[-2]),
                          self.ALL_PANELS.index(hist[-1]))
                match = self._match_pattern_prefix(last3, self.SPIN_PATTERNS, 3)
                if match is not None:
                    self._spin_pattern = list(match)
                    self._spin_idx = 3
                    arrow_idx = self._spin_pattern[self._spin_idx]
                    self._spin_idx += 1
                    if self._spin_idx >= len(self._spin_pattern):
                        self._spin_pattern = None
                        self._spin_idx = 0
                    arrow = self.ALL_PANELS[arrow_idx]
                    foot = self._pick_foot(time)
                    self._update_foot(foot, arrow)
                    self._step_count += 1
                    self._arrow_history.append(arrow)
                    if len(self._arrow_history) > 5:
                        self._arrow_history = self._arrow_history[-5:]
                    return arrow

        foot = self._pick_foot(time)
        panels = self._panels_for_foot(foot)
        current_pos = self.left_pos if foot == 'left' else self.right_pos
        self._step_count += 1

        # Crossover branch: pick from the OTHER foot's panels.
        if (
            'crossover' in self.allowed_patterns
            and self.crossover_rate > 0.0
            and self.rng.random() < self.crossover_rate
        ):
            cross_panels = self._panels_for_foot(
                'right' if foot == 'left' else 'left'
            )
            arrow = self.rng.choice(cross_panels)
        else:
            weights = self._panel_weights(panels, current_pos, brightness, pitch_class)
            total = weights[0] + weights[1]
            arrow = panels[0] if self.rng.random() < weights[0] / total else panels[1]

        # Guard: if the chosen arrow is still held from a prior hold_start,
        # swap to any free panel (same foot first, then the other foot).
        if arrow in self._held_arrows(time):
            same_foot = [p for p in panels if p not in self.active_holds]
            if same_foot:
                arrow = same_foot[0]
            else:
                free = [p for p in self.ALL_PANELS if p not in self.active_holds]
                if free:
                    arrow = free[0]
                    foot = 'left' if arrow in self.LEFT_FOOT_PANELS else 'right'

        self._update_foot(foot, arrow)
        self._arrow_history.append(arrow)
        if len(self._arrow_history) > 5:
            self._arrow_history = self._arrow_history[-5:]
        return arrow

    def _fallback_arrow(self, held_now: Set[Direction]) -> Direction:
        """Return a safe fallback arrow when pattern collides with holds."""
        free = [p for p in self.ALL_PANELS if p not in held_now]
        if free:
            return free[0]
        # All four held — abort patterns, just pick anything.
        return Direction.LEFT

    @staticmethod
    def _match_pattern_prefix(
        prefix: tuple,
        patterns: List[List[int]],
        min_match: int,
    ) -> Optional[List[int]]:
        """Check if `prefix` matches the first `min_match` elements of any pattern.
        Returns the first matching full pattern, or None."""
        for pat in patterns:
            if list(prefix) == pat[:min_match]:
                return pat
        return None

    def assign_jump(self, time: float, energy: float = 0.5,
                     brightness: float = 0.5, pitch_class: int = -1) -> List[Direction]:
        if self._held_feet(time):
            return [self.assign_single(time, energy, brightness, pitch_class)]

        if energy > 0.7:
            candidates = [self.JUMP_PATTERNS[0], self.JUMP_PATTERNS[2],
                          self.JUMP_PATTERNS[3]]
        elif energy < 0.3:
            candidates = [self.JUMP_PATTERNS[1]]
        else:
            candidates = list(self.JUMP_PATTERNS)

        held_now = self._held_arrows(time)
        # Shuffle so repeated collisions don't always fall back the same way.
        self.rng.shuffle(candidates)
        pattern = None
        for c in candidates:
            if c[0] not in held_now and c[1] not in held_now:
                pattern = c
                break
        # If every jump pattern collides with active holds, emit a tap.
        if pattern is None:
            return [self.assign_single(time, energy, brightness, pitch_class)]

        self._step_count += 1
        self.left_pos = pattern[0]
        self.right_pos = pattern[1]
        self.last_foot = 'right'
        self._last_arrow = pattern[1]
        return [pattern[0], pattern[1]]

    def start_hold(self, time: float, duration: float,
                    energy: float = 0.5, brightness: float = 0.5,
                    pitch_class: int = -1) -> Direction:
        self._purge_holds(time)
        arrow = self.assign_single(time, energy, brightness, pitch_class)
        # Same-arrow collision guard: if assign_single landed on an already-
        # held arrow (e.g. via crossover or jack with stale state), swap to
        # any free panel. With the concurrency cap upstream, a free panel is
        # guaranteed to exist.
        if arrow in self.active_holds:
            free = [p for p in self.ALL_PANELS if p not in self.active_holds]
            if free:
                arrow = free[0]
                foot = 'left' if arrow in self.LEFT_FOOT_PANELS else 'right'
                self._update_foot(foot, arrow)
        self.active_holds[arrow] = time + duration
        return arrow

    def maybe_start_stream(self, energy: float, remaining_events: int,
                            max_stream_length: int, time: float = 0.0) -> bool:
        if 'stream' not in self.allowed_patterns or max_stream_length == 0:
            return False
        if self._stream_pattern is not None:
            return False
        if self._held_feet(time):
            return False

        # stream_preference is the per-profile bias; energy modulates it so
        # high-energy sections still dominate the streamed parts.
        stream_chance = max(
            0.0, 0.5 * self.stream_preference + 0.5 * (energy - 0.5),
        )
        stream_chance = min(stream_chance, 0.95)
        if self.rng.random() >= stream_chance:
            return False

        pattern = list(self.rng.choice(self.STREAM_PATTERNS))
        max_len = min(max_stream_length, remaining_events, len(pattern) * 3)
        stream_len = self.rng.randint(len(pattern), max(len(pattern), max_len))

        full_pattern = []
        while len(full_pattern) < stream_len:
            full_pattern.extend(pattern)
        self._stream_pattern = full_pattern[:stream_len]
        self._stream_idx = 0
        return True


# ---------------------------------------------------------------------------
# Style profile resolution
# ---------------------------------------------------------------------------

def _profile_knobs(profile: Dict) -> Dict[str, float]:
    """Derive runtime knobs from a profile centroid's raw_means.

    Profiles ship raw stat means; we map those to the post-processing
    parameters that drive jump-vs-tap classification, hold cadence, and the
    arrow-assigner branches. Values are clamped to defensible ranges.
    """
    rm = profile.get('raw_means', {})
    knobs = dict(DEFAULT_STYLE_KNOBS)

    # Top-K jump selection: the profile's observed jump_rate tells us what
    # fraction of peaks should become jumps. We use the raw rate as a target
    # and keep a small absolute intensity floor to filter out near-zero noise.
    jump_rate = float(rm.get('jump_rate', 0.05))
    knobs['target_jump_rate'] = float(np.clip(jump_rate, 0.0, 0.40))
    knobs['jump_threshold'] = 0.05

    # Jumphold: enable when both holds and jumps are common; rate scales with
    # min(hold_rate, jump_rate).
    hold_rate = float(rm.get('hold_rate', 0.04))
    knobs['jumphold_rate'] = float(np.clip(min(hold_rate, jump_rate) * 1.5, 0.0, 0.4))

    knobs['jack_rate'] = float(np.clip(rm.get('jack_rate', 0.05), 0.0, 0.5))
    knobs['crossover_rate'] = float(np.clip(rm.get('crossover_rate', 0.10), 0.0, 0.5))
    # stream_preference rolls stream_density into the assigner's branch
    # probability — high values produce visibly more stream patterns.
    knobs['stream_preference'] = float(np.clip(
        rm.get('stream_density', 0.20) * 2.0, 0.0, 1.0,
    ))
    # Candle rate: scale with crossover_rate (candles are a crossover variant)
    # and stream_density (candles appear in dense passages).
    candle_raw = float(rm.get('crossover_rate', 0.05)) * 0.5
    knobs['candle_rate'] = float(np.clip(candle_raw, 0.0, 0.15))
    # Spin rate: rare; only in high-crossover, high-density profiles.
    spin_raw = float(rm.get('crossover_rate', 0.05)) * 0.2
    knobs['spin_rate'] = float(np.clip(spin_raw, 0.0, 0.08))
    return knobs


class MLChartGenerator:
    """Generate step charts using the v7 hybrid model + style profiles."""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        chunk_frames: int = 500,
        overlap_frames: int = 100,
        confidence_threshold: Optional[float] = None,
        min_note_gap: float = 0.05,
        snap_to_beats: bool = True,
        sustain_threshold_up: float = 0.5,
        sustain_threshold_down: float = 0.4,
        intensity_threshold_base: float = 0.5,
        style: str = 'auto',
        style_profiles_path: Optional[str] = None,
        min_first_step_time: float = 0.25,
        min_last_step_buffer: float = 0.0,
        timing_trim: float = 0.0,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self._confidence_threshold_override = confidence_threshold
        self.confidence_threshold = (
            float(confidence_threshold) if confidence_threshold is not None else 0.3
        )
        self.min_note_gap = min_note_gap
        self.snap_to_beats = snap_to_beats
        self.sustain_threshold_up = float(sustain_threshold_up)
        self.sustain_threshold_down = float(sustain_threshold_down)
        self.intensity_threshold_base = float(intensity_threshold_base)
        self.min_first_step_time = float(min_first_step_time)
        self.min_last_step_buffer = float(min_last_step_buffer)
        # Manual timing trim (seconds) added on top of the per-song measured
        # offset (see _analyze_audio). Negative pulls notes earlier. Normally 0;
        # exists only for fine-tuning beyond the auto-calibrated value.
        self.timing_trim = float(timing_trim)
        self.style = str(style)

        self.model: Optional[StepChartModel] = None
        # Set at load time: True only for checkpoints trained with the learned
        # step-selection head (arch_version >= 9). Gates learned vs assigner arrows.
        self.has_arrow_head: bool = False
        self.default_density_by_id = DEFAULT_DENSITY_BY_ID.clone()
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.n_in_channels: int = N_FEAT_CHANNELS

        # Style profile catalog. None means "no profiles loaded" — `style`
        # is then ignored and DEFAULT_STYLE_KNOBS are used directly.
        self.style_profiles: Optional[List[Dict]] = None
        self._style_profiles_by_name: Dict[str, Dict] = {}

        if model_path:
            self.load_model(model_path)
        if style_profiles_path:
            self._load_style_profiles(style_profiles_path)

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        arch_version = int(checkpoint.get('arch_version', 1))
        if arch_version > ARCH_VERSION:
            raise ValueError(
                f"Checkpoint at {model_path} is arch_version={arch_version}, newer "
                f"than this code's {ARCH_VERSION}. Update model.py."
            )
        # Older checkpoints load with strict=False; the learned arrow head only
        # activates for checkpoints trained with it (arch_version >= 9). Anything
        # older falls back to the algorithmic FootStateArrowAssigner.
        self.has_arrow_head = arch_version >= 9
        if arch_version < ARCH_VERSION:
            logger.warning(
                f"Checkpoint arch_version={arch_version} < {ARCH_VERSION}; loading "
                f"tolerantly. Learned arrow head "
                f"{'ENABLED' if self.has_arrow_head else 'DISABLED (assigner fallback)'}."
            )

        n_in_channels = int(checkpoint.get('n_in_channels', N_FEAT_CHANNELS))
        args = checkpoint.get('args', {})
        self.model = StepChartModel(
            n_in_channels=n_in_channels,
            hidden_dim=args.get('hidden_dim', 160),
            n_heads=args.get('n_heads', 4),
            n_transformer_layers=args.get('n_layers', 3),
            n_difficulties=5,
        ).to(self.device)
        self.n_in_channels = n_in_channels

        state_dict = checkpoint.get('ema_state_dict') or checkpoint['model_state_dict']
        if checkpoint.get('ema_state_dict') is not None:
            logger.info("Loading EMA weights from checkpoint.")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Checkpoint missing keys (zero-initialized): {missing}")
        if unexpected:
            logger.warning(f"Checkpoint had unexpected keys (ignored): {unexpected}")
        self.model.eval()

        ckpt_default = checkpoint.get('default_density_by_id')
        if ckpt_default is not None:
            self.default_density_by_id = torch.as_tensor(
                ckpt_default, dtype=torch.float32
            )
            logger.info(
                f"Using checkpoint default densities: "
                f"{self.default_density_by_id.tolist()}"
            )

        ckpt_thr = checkpoint.get('best_onset_threshold')
        if self._confidence_threshold_override is not None:
            logger.info(
                f"Onset threshold = {self.confidence_threshold:.3f} (caller override)"
            )
        elif ckpt_thr is not None:
            self.confidence_threshold = float(ckpt_thr)
            logger.info(
                f"Onset threshold = {self.confidence_threshold:.3f} "
                f"(from checkpoint best_onset_threshold)"
            )

        ckpt_sustain_up = checkpoint.get('best_sustain_threshold')
        if ckpt_sustain_up is not None:
            self.sustain_threshold_up = float(ckpt_sustain_up)
            self.sustain_threshold_down = max(
                0.0, float(ckpt_sustain_up) - 0.1,
            )
            logger.info(
                f"Sustain thresholds: up={self.sustain_threshold_up:.3f} "
                f"down={self.sustain_threshold_down:.3f}"
            )

        if 'feat_mean' in checkpoint and 'feat_std' in checkpoint:
            fm = np.asarray(checkpoint['feat_mean'], dtype=np.float32)
            fs = np.asarray(checkpoint['feat_std'], dtype=np.float32)
            assert fm.shape == (n_in_channels,), (
                f"checkpoint feat_mean has shape {fm.shape}, expected "
                f"({n_in_channels},)"
            )
            self.feat_mean = fm
            self.feat_std = np.where(fs < 1e-6, 1.0, fs).astype(np.float32)
        else:
            raise ValueError(
                f"v7 checkpoint at {model_path} is missing feat_mean / feat_std. "
                f"Re-train via the new train.py."
            )

        logger.info(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    def _load_style_profiles(self, path: str) -> None:
        with open(path, 'r') as f:
            payload = json.load(f)
        profiles = payload.get('profiles', [])
        if not profiles:
            logger.warning(f"Style profile file {path} contained no profiles.")
            return
        self.style_profiles = profiles
        self._style_profiles_by_name = {p['name']: p for p in profiles}
        names = [p['name'] for p in profiles]
        logger.info(f"Loaded {len(profiles)} style profiles: {names}")

    def _resolve_style(
        self,
        style: str,
        onset_density: float,
        intensity_mean: float,
    ) -> Tuple[Dict[str, float], str]:
        """Pick a profile and return (knobs, resolved_name).

        - If style profiles haven't been loaded, fall back to defaults.
        - If `style == 'auto'`, pick the profile whose mean_density is closest
          to the predicted onset density (with intensity_mean breaking ties).
        - Otherwise look up the named profile; raise on miss.
        """
        if not self.style_profiles:
            return dict(DEFAULT_STYLE_KNOBS), 'default'

        if style == 'auto':
            best_name = None
            best_score = math.inf
            for p in self.style_profiles:
                rm = p['raw_means']
                density_diff = abs(rm.get('mean_density', 2.0) - onset_density)
                jump_diff = abs(rm.get('jump_rate', 0.05) - intensity_mean)
                score = density_diff + 0.5 * jump_diff
                if score < best_score:
                    best_score = score
                    best_name = p['name']
            chosen = self._style_profiles_by_name[best_name]
            knobs = _profile_knobs(chosen)
            logger.info(
                f"Resolved style='auto' → '{best_name}' "
                f"(onset_density={onset_density:.2f}, intensity_mean={intensity_mean:.3f})"
            )
            return knobs, best_name

        if style not in self._style_profiles_by_name:
            available = ', '.join(self._style_profiles_by_name.keys())
            raise ValueError(
                f"Unknown style '{style}'. Available: {available} (or 'auto')."
            )
        chosen = self._style_profiles_by_name[style]
        knobs = _profile_knobs(chosen)
        logger.info(f"Resolved style='{style}' → knobs={knobs}")
        return knobs, style

    # ------------------------------------------------------------------
    # Audio feature extraction
    # ------------------------------------------------------------------

    def _extract_feats(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute the v7 feats[T, n_in_channels] tensor from raw audio.

        Mirrors prepare_data.extract_audio_features but returns float32 ready
        for whitening (the dataset path stored float16 + applied per-song
        z-norm to side channels; we replicate the per-song z-norm here).
        """
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
        np.clip(mel_db, DB_MIN, DB_MAX, out=mel_db)
        mel_scaled = (mel_db - DB_MIN) / DB_RANGE
        mel_t = mel_scaled.T.astype(np.float32)
        n_frames = mel_t.shape[0]

        try:
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=HOP_LENGTH,
            )
        except Exception:
            onset_env = np.zeros(n_frames, dtype=np.float32)
        onset_env = self._align_length(np.asarray(onset_env, dtype=np.float32), n_frames)

        try:
            contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT,
            )
        except Exception:
            contrast = np.zeros((N_SPECTRAL_CONTRAST, n_frames), dtype=np.float32)
        contrast = np.asarray(contrast, dtype=np.float32)
        if contrast.shape[0] != N_SPECTRAL_CONTRAST:
            if contrast.shape[0] < N_SPECTRAL_CONTRAST:
                pad = np.zeros(
                    (N_SPECTRAL_CONTRAST - contrast.shape[0], contrast.shape[1]),
                    dtype=np.float32,
                )
                contrast = np.concatenate([contrast, pad], axis=0)
            else:
                contrast = contrast[:N_SPECTRAL_CONTRAST]
        contrast_t = self._align_length(contrast.T, n_frames)

        onset_z = self._per_song_znorm(onset_env)
        contrast_z = self._per_song_znorm(contrast_t)

        feats = np.concatenate([mel_t, onset_z, contrast_z], axis=1).astype(np.float32)
        assert feats.shape == (n_frames, self.n_in_channels), (
            f"feats shape {feats.shape} != expected ({n_frames}, {self.n_in_channels})"
        )
        return feats

    @staticmethod
    def _align_length(arr: np.ndarray, n_frames: int) -> np.ndarray:
        if arr.ndim == 1:
            T = arr.shape[0]
            if T == n_frames:
                return arr
            if T > n_frames:
                return arr[:n_frames]
            out = np.zeros(n_frames, dtype=arr.dtype)
            out[:T] = arr
            return out
        T = arr.shape[0]
        if T == n_frames:
            return arr
        if T > n_frames:
            return arr[:n_frames]
        pad = np.zeros((n_frames - T,) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    @staticmethod
    def _per_song_znorm(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, None]
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return ((x - mean) / std).astype(np.float32)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _analyze_audio(self, audio_path: str) -> "_AudioAnalysis":
        """Load audio and compute everything that's shared across difficulties.

        Extracted from :meth:`generate_from_audio` so the multi-difficulty
        path can run feature extraction / beat tracking / RMS / HPSS once
        and reuse them across N inference passes.
        """
        if self.feat_mean is None or self.feat_std is None:
            raise RuntimeError("Model loaded without feat_mean/feat_std. Retrain.")

        logger.info(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        feats = self._extract_feats(y, sr)
        feats_whitened = (feats - self.feat_mean) / self.feat_std

        # Shared onset-strength envelope: drives both the beat grid and the
        # per-song timing-offset measurement below.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

        # Global tempo scalar (still used for difficulty conditioning and the
        # frontend receptor pulse, which both want a single BPM).
        tempo, bt_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
        )
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        tempo = float(tempo)

        # Predominant Local Pulse: phase-accurate under tempo variation, unlike
        # beat_track's single-tempo DP (which drifts out of phase mid-song and
        # made on-beat notes snap to 8th/16th cells → wrong subdivision colors).
        # Constrain the tempo window around the detected tempo to avoid octave
        # (half/double) errors.
        pulse = librosa.beat.plp(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
            tempo_min=max(30.0, tempo * 0.7), tempo_max=min(300.0, tempo * 1.5),
        )
        plp_frames = np.flatnonzero(librosa.util.localmax(pulse))
        beat_times = librosa.frames_to_time(plp_frames, sr=sr, hop_length=HOP_LENGTH)

        # Fallback: if PLP produced an implausible grid (too few beats vs.
        # duration*tempo), revert to the beat_track grid so steady-tempo songs
        # are never made worse.
        expected_beats = duration * tempo / 60.0
        if len(beat_times) < 0.5 * expected_beats:
            beat_times = librosa.frames_to_time(
                bt_frames, sr=sr, hop_length=HOP_LENGTH,
            )

        # Per-song timing offset: measure how far each detected onset's envelope
        # peak sits after the true transient. onset_detect(backtrack=True) snaps
        # each onset back to the preceding local energy minimum; the median gap is
        # the systematic latency to compensate by pulling notes earlier.
        peaks = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
            backtrack=False, units='time',
        )
        backtracked = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH,
            backtrack=True, units='time',
        )
        if len(peaks) and len(peaks) == len(backtracked):
            lag = float(np.median(peaks - backtracked))
            timing_offset = -max(0.0, min(lag, 0.080))  # clamp to (-80ms, 0]
        else:
            timing_offset = -0.030  # old hardcoded default as a safe fallback
        logger.info(
            "Measured timing offset: %.1f ms (tempo=%.1f, beats=%d)",
            timing_offset * 1000.0, tempo, len(beat_times),
        )

        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        rms_norm = rms / (rms.max() + 1e-8)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=HOP_LENGTH,
        )[0]
        centroid_norm = centroid / (centroid.max() + 1e-8)

        # Harmonic-energy envelope feeds only the hold-note decay heuristic in
        # post-processing (it is not a model input), so an approximate curve is
        # fine. Full-res librosa.effects.hpss costs ~6s on a 3-min track because
        # it median-filters the hop=512 STFT and reconstructs a waveform we
        # immediately collapse back to per-frame energy. Running HPSS on a
        # 4x-coarser STFT, taking energy straight from the harmonic magnitude
        # spectrogram (no inverse STFT), and interpolating back to the frame
        # grid is ~4x faster (~1.5s) with negligible impact on the decay check.
        try:
            harm_mag = librosa.decompose.hpss(
                np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH * 4))
            )[0]
            harm_energy = np.sqrt(np.mean(harm_mag ** 2, axis=0))
            harm_rms = np.interp(
                np.linspace(0.0, 1.0, feats.shape[0]),
                np.linspace(0.0, 1.0, harm_energy.shape[0]),
                harm_energy,
            )
        except Exception:
            harm_rms = rms.copy()
        harm_rms = harm_rms / (harm_rms.max() + 1e-8)
        harm_rms = self._align_length(harm_rms, feats.shape[0])
        rms_norm = self._align_length(rms_norm, feats.shape[0])
        centroid_norm = self._align_length(centroid_norm, feats.shape[0])

        # Dominant pitch class per frame for music-aware arrow selection. Chroma
        # sums the spectrum into 12 pitch classes; the argmax is the frame's
        # strongest note. Frames whose top class isn't clearly dominant (flat
        # chroma ⇒ percussive/noisy) are marked -1 so the assigner ignores them.
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
            chroma_sum = chroma.sum(axis=0) + 1e-8
            top_class = chroma.argmax(axis=0).astype(np.int16)
            top_frac = chroma.max(axis=0) / chroma_sum       # dominance in [1/12, 1]
            pitch_class = np.where(top_frac >= 0.18, top_class, -1).astype(np.int16)
            # Fit to the feats grid, padding any shortfall with -1 (not 0=C).
            fitted = np.full(feats.shape[0], -1, dtype=np.int16)
            n = min(pitch_class.shape[0], feats.shape[0])
            fitted[:n] = pitch_class[:n]
            pitch_class = fitted
        except Exception:
            pitch_class = np.full(feats.shape[0], -1, dtype=np.int16)

        audio_seed = int(
            hashlib.sha256(feats[:min(4096, feats.shape[0])].tobytes()
                           ).hexdigest()[:8],
            16,
        )

        return _AudioAnalysis(
            duration=duration,
            feats_whitened=feats_whitened,
            tempo=tempo,
            beat_times=beat_times,
            timing_offset=timing_offset,
            rms_norm=rms_norm,
            centroid_norm=centroid_norm,
            harm_rms=harm_rms,
            audio_seed=audio_seed,
            pitch_class=pitch_class,
        )

    def _chart_from_signals(
        self,
        analysis: "_AudioAnalysis",
        difficulty: str,
        onset_p: np.ndarray,
        sustain_p: np.ndarray,
        intensity: np.ndarray,
        style: Optional[str] = None,
        arrow_features: Optional[np.ndarray] = None,
    ) -> Chart:
        """Build a Chart from already-computed per-difficulty signals.

        Shared by the single-difficulty and batched-multi paths so
        post-processing only lives in one place.
        """
        diff_config = get_difficulty_config(difficulty)

        T = onset_p.shape[0]
        onset_density_pred = float(onset_p.sum() / max(T / FRAMES_PER_SECOND, 1.0))
        intensity_mean_pred = (
            float(intensity[intensity > 0.0].mean()) if (intensity > 0.0).any() else 0.0
        )
        resolved_style = style if style is not None else self.style
        knobs, resolved_name = self._resolve_style(
            resolved_style, onset_density_pred, intensity_mean_pred,
        )

        # Effective offset baked into emitted step times: per-song measured
        # latency plus the manual trim.
        timing_offset = analysis.timing_offset + self.timing_trim

        steps = self._postprocess(
            onset_p, sustain_p, intensity,
            analysis.tempo, analysis.beat_times, analysis.duration, diff_config,
            energy_curve=analysis.rms_norm,
            brightness_curve=analysis.centroid_norm,
            harmonic_curve=analysis.harm_rms,
            audio_seed=analysis.audio_seed,
            style_knobs=knobs,
            timing_offset=timing_offset,
            arrow_features=arrow_features,
            pitch_class_curve=analysis.pitch_class,
        )

        chart = Chart(
            steps=steps,
            difficulty=difficulty,
            tempo=analysis.tempo,
            duration=analysis.duration,
            timing_offset=round(timing_offset, 4),
        )
        logger.info(
            f"Generated {len(steps)} {difficulty} steps "
            f"({len(chart.get_taps())} taps, {len(chart.get_holds())} holds) "
            f"style='{resolved_name}'"
        )
        return chart

    def _chart_for_difficulty(
        self,
        analysis: "_AudioAnalysis",
        difficulty: str,
        target_density: Optional[float] = None,
        style: Optional[str] = None,
    ) -> Chart:
        """Run inference + post-processing for one difficulty over a shared
        :class:`_AudioAnalysis`."""
        difficulty_id = APP_DIFFICULTY_MAP.get(difficulty, 2)
        if target_density is None:
            target_density = float(self.default_density_by_id[difficulty_id])
        logger.info(
            f"Running inference (difficulty={difficulty}, id={difficulty_id}, "
            f"density={target_density:.2f} steps/sec)..."
        )

        onset_p, sustain_p, intensity = self._predict_chunked(
            analysis.feats_whitened, difficulty_id, target_density,
        )
        arrow_features = None
        if self.has_arrow_head:
            arrow_features = self._encode_features_chunked(
                analysis.feats_whitened, difficulty_id, target_density,
            )
        return self._chart_from_signals(
            analysis, difficulty, onset_p, sustain_p, intensity, style=style,
            arrow_features=arrow_features,
        )

    def generate_from_audio(
        self,
        audio_path: str,
        difficulty: str = 'medium',
        target_density: Optional[float] = None,
        style: Optional[str] = None,
    ) -> Chart:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        analysis = self._analyze_audio(audio_path)
        return self._chart_for_difficulty(
            analysis, difficulty, target_density=target_density, style=style,
        )

    def generate_all_difficulties(
        self,
        audio_path: str,
        style: Optional[str] = None,
        difficulties: Optional[List[str]] = None,
    ) -> Dict[str, Chart]:
        """Generate a chart for every difficulty over one shared audio analysis.

        Audio load + feature extraction + beat/RMS/HPSS analysis runs once,
        AND model inference runs once with all difficulties batched into a
        single forward pass per chunk. Only post-processing (NMS, beat snap,
        arrow assignment) is serial per difficulty.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        names = difficulties or list(APP_DIFFICULTY_MAP.keys())
        analysis = self._analyze_audio(audio_path)

        difficulty_ids = [APP_DIFFICULTY_MAP.get(n, 2) for n in names]
        target_densities = [
            float(self.default_density_by_id[did]) for did in difficulty_ids
        ]
        logger.info(
            f"Running batched inference over {len(names)} difficulties: "
            f"{list(zip(names, [round(d, 2) for d in target_densities]))}"
        )
        onset_batch, sustain_batch, intensity_batch = self._predict_chunked_multi(
            analysis.feats_whitened, difficulty_ids, target_densities,
        )

        charts: Dict[str, Chart] = {}
        for i, name in enumerate(names):
            charts[name] = self._chart_from_signals(
                analysis, name,
                onset_batch[i], sustain_batch[i], intensity_batch[i],
                style=style,
            )
        return charts

    @torch.no_grad()
    def _predict_chunked(
        self,
        feats: np.ndarray,
        difficulty_id: int,
        target_density: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model on overlapping chunks; return (onset_p, sustain_p, intensity).

        Each output is shape [T] float32. onset_p and sustain_p are sigmoids;
        intensity is the raw regression output (clipped to [0, 1]).
        """
        T = feats.shape[0]
        onset_sum = np.zeros(T, dtype=np.float32)
        sustain_sum = np.zeros(T, dtype=np.float32)
        intensity_sum = np.zeros(T, dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)

        stride = self.chunk_frames - self.overlap_frames
        density_norm = (target_density - DENSITY_MEAN) / DENSITY_STD

        for start in range(0, T, stride):
            end = min(start + self.chunk_frames, T)
            chunk = feats[start:end]

            if chunk.shape[0] < self.chunk_frames:
                pad = np.zeros(
                    (self.chunk_frames - chunk.shape[0], self.n_in_channels),
                    dtype=np.float32,
                )
                chunk = np.concatenate([chunk, pad], axis=0)

            feats_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            diff_tensor = torch.tensor(
                [difficulty_id], dtype=torch.long, device=self.device,
            )
            density_tensor = torch.tensor(
                [density_norm], dtype=torch.float32, device=self.device,
            )
            start_seconds = float(start) / FRAMES_PER_SECOND
            remaining_seconds = max(0.0, float(T - end) / FRAMES_PER_SECOND)
            start_seconds_tensor = torch.tensor(
                [start_seconds], dtype=torch.float32, device=self.device,
            )
            remaining_seconds_tensor = torch.tensor(
                [remaining_seconds], dtype=torch.float32, device=self.device,
            )

            onset_logits, sustain_logits, intensity_pred, _arrow = self.model(
                feats_tensor, diff_tensor, density_tensor,
                start_seconds_tensor, remaining_seconds_tensor,
            )
            onset_p = torch.sigmoid(onset_logits.float()).cpu().numpy()[0, :, 0]
            sustain_p = torch.sigmoid(sustain_logits.float()).cpu().numpy()[0, :, 0]
            intensity_v = intensity_pred.float().cpu().numpy()[0, :, 0]

            valid_len = min(end - start, self.chunk_frames)
            onset_sum[start:start + valid_len] += onset_p[:valid_len]
            sustain_sum[start:start + valid_len] += sustain_p[:valid_len]
            intensity_sum[start:start + valid_len] += intensity_v[:valid_len]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)
        onset_avg = onset_sum / counts
        sustain_avg = sustain_sum / counts
        intensity_avg = np.clip(intensity_sum / counts, 0.0, 1.0)
        return onset_avg, sustain_avg, intensity_avg

    @torch.no_grad()
    def _encode_features_chunked(
        self,
        feats: np.ndarray,
        difficulty_id: int,
        target_density: float,
    ) -> np.ndarray:
        """Backbone hidden features [T, H], stitched over overlapping chunks.

        Mirrors :meth:`_predict_chunked`'s chunking but returns `model.encode`
        output (averaged across overlaps), which the learned arrow head consumes
        at onset frames. Only called when `self.has_arrow_head`.
        """
        T = feats.shape[0]
        H = int(self.model.hidden_dim)
        feat_sum = np.zeros((T, H), dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)

        stride = self.chunk_frames - self.overlap_frames
        density_norm = (target_density - DENSITY_MEAN) / DENSITY_STD

        for start in range(0, T, stride):
            end = min(start + self.chunk_frames, T)
            chunk = feats[start:end]
            if chunk.shape[0] < self.chunk_frames:
                pad = np.zeros(
                    (self.chunk_frames - chunk.shape[0], self.n_in_channels),
                    dtype=np.float32,
                )
                chunk = np.concatenate([chunk, pad], axis=0)

            feats_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            diff_tensor = torch.tensor([difficulty_id], dtype=torch.long, device=self.device)
            density_tensor = torch.tensor([density_norm], dtype=torch.float32, device=self.device)
            start_seconds = float(start) / FRAMES_PER_SECOND
            remaining_seconds = max(0.0, float(T - end) / FRAMES_PER_SECOND)
            ss_tensor = torch.tensor([start_seconds], dtype=torch.float32, device=self.device)
            rs_tensor = torch.tensor([remaining_seconds], dtype=torch.float32, device=self.device)

            feats_h = self.model.encode(
                feats_tensor, diff_tensor, density_tensor, ss_tensor, rs_tensor,
            )
            fh = feats_h.float().cpu().numpy()[0]  # [chunk_frames, H]

            valid_len = min(end - start, self.chunk_frames)
            feat_sum[start:start + valid_len] += fh[:valid_len]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)[:, None]
        return feat_sum / counts

    @staticmethod
    def _class_to_dirs(cls: int) -> List[Direction]:
        code = ARROW_CLASS_CODES[cls]
        return [ARROW_DIRECTIONS[i] for i in range(4) if code & (1 << i)]

    @torch.no_grad()
    def _decode_arrows(
        self,
        features: np.ndarray,
        note_events: List[dict],
    ) -> List[List[Direction]]:
        """Autoregressively decode a panel set for each note event.

        Greedy over the arrow head's logits, feeding the previous chosen class
        back in (as at train time). A legality mask enforces: (a) the desired
        arity — the event's tap/jump decision from the intensity head is
        preserved — (b) no arrow that is still held, and (c) the 2-arrow
        concurrency cap. Holds register into a local active-hold map so later
        events see them. Returns one Direction list per event, aligned to
        `note_events`. Only called when `self.has_arrow_head`.
        """
        T = features.shape[0]
        feats_t = torch.from_numpy(features.astype(np.float32)).to(self.device)
        n_classes = len(ARROW_CLASS_CODES)

        active_holds: Dict[Direction, float] = {}
        prev_class = ARROW_BOS_CLASS
        last_time: Optional[float] = None
        out: List[List[Direction]] = []

        for ev in note_events:
            t = float(ev['time'])
            frame = max(0, min(int(round(t * FRAMES_PER_SECOND)), T - 1))
            active_holds = {a: e for a, e in active_holds.items() if e > t}
            held = set(active_holds.keys())
            free_slots = max(0, 2 - len(held))
            target_arity = 2 if int(ev.get('num_arrows', 1)) >= 2 else 1

            dt = 0.0 if last_time is None else min(2.0, t - last_time)
            h = feats_t[frame]
            pc = torch.tensor(prev_class, dtype=torch.long, device=self.device)
            dtt = torch.tensor(dt, dtype=torch.float32, device=self.device)
            logits = self.model.arrow_head(h, pc, dtt)  # [n_classes]

            mask = torch.ones(n_classes, dtype=torch.bool, device=self.device)  # True = forbid
            for c in range(n_classes):
                dirs = self._class_to_dirs(c)
                if len(dirs) != target_arity:
                    continue
                if any(d in held for d in dirs):
                    continue
                if len(dirs) > free_slots:
                    continue
                mask[c] = False

            if bool((~mask).any()):
                cls = int(torch.argmax(logits.masked_fill(mask, float('-inf'))))
            else:
                # No legal class at the desired arity: relax to any non-held single.
                singles = [c for c in range(n_classes)
                           if len(self._class_to_dirs(c)) == 1
                           and self._class_to_dirs(c)[0] not in held]
                cls = singles[int(torch.argmax(logits[singles]))] if singles else int(torch.argmax(logits))

            dirs = self._class_to_dirs(cls)
            if ev.get('type') == 'hold':
                end_time = t + float(ev.get('hold_duration', 0.0))
                for d in dirs:
                    active_holds[d] = end_time
            out.append(dirs)
            prev_class = cls
            last_time = t
        return out

    @torch.no_grad()
    def _predict_chunked_multi(
        self,
        feats: np.ndarray,
        difficulty_ids: List[int],
        target_densities: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched-multi version of :meth:`_predict_chunked`.

        Runs all difficulties as one forward pass per chunk: the shared audio
        features are broadcast across the batch dimension, while difficulty
        and density vary per row. Returns three arrays of shape ``[B, T]``,
        one row per requested difficulty.
        """
        B = len(difficulty_ids)
        if B == 0:
            raise ValueError("difficulty_ids must be non-empty")
        if len(target_densities) != B:
            raise ValueError(
                f"target_densities length {len(target_densities)} != "
                f"difficulty_ids length {B}"
            )

        T = feats.shape[0]
        onset_sum = np.zeros((B, T), dtype=np.float32)
        sustain_sum = np.zeros((B, T), dtype=np.float32)
        intensity_sum = np.zeros((B, T), dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)

        stride = self.chunk_frames - self.overlap_frames
        density_norms = [
            (float(d) - DENSITY_MEAN) / DENSITY_STD for d in target_densities
        ]

        # Conditioning tensors are per-chunk-constant — build once.
        diff_tensor = torch.tensor(
            difficulty_ids, dtype=torch.long, device=self.device,
        )
        density_tensor = torch.tensor(
            density_norms, dtype=torch.float32, device=self.device,
        )

        for start in range(0, T, stride):
            end = min(start + self.chunk_frames, T)
            chunk = feats[start:end]

            if chunk.shape[0] < self.chunk_frames:
                pad = np.zeros(
                    (self.chunk_frames - chunk.shape[0], self.n_in_channels),
                    dtype=np.float32,
                )
                chunk = np.concatenate([chunk, pad], axis=0)

            # Expand the single chunk to a [B, chunk_frames, C] batch. The
            # underlying storage is shared; `.contiguous()` materializes a real
            # batched tensor so the model's BatchNorm sees independent rows.
            feats_tensor = (
                torch.from_numpy(chunk)
                .to(self.device)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .contiguous()
            )

            start_seconds = float(start) / FRAMES_PER_SECOND
            remaining_seconds = max(0.0, float(T - end) / FRAMES_PER_SECOND)
            start_seconds_tensor = torch.full(
                (B,), start_seconds, dtype=torch.float32, device=self.device,
            )
            remaining_seconds_tensor = torch.full(
                (B,), remaining_seconds, dtype=torch.float32, device=self.device,
            )

            onset_logits, sustain_logits, intensity_pred, _arrow = self.model(
                feats_tensor, diff_tensor, density_tensor,
                start_seconds_tensor, remaining_seconds_tensor,
            )
            onset_p = torch.sigmoid(onset_logits.float()).cpu().numpy()[:, :, 0]
            sustain_p = torch.sigmoid(sustain_logits.float()).cpu().numpy()[:, :, 0]
            intensity_v = intensity_pred.float().cpu().numpy()[:, :, 0]

            valid_len = min(end - start, self.chunk_frames)
            onset_sum[:, start:start + valid_len] += onset_p[:, :valid_len]
            sustain_sum[:, start:start + valid_len] += sustain_p[:, :valid_len]
            intensity_sum[:, start:start + valid_len] += intensity_v[:, :valid_len]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)  # broadcast across the B axis
        onset_avg = onset_sum / counts
        sustain_avg = sustain_sum / counts
        intensity_avg = np.clip(intensity_sum / counts, 0.0, 1.0)
        return onset_avg, sustain_avg, intensity_avg

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _find_hold_end(
        self,
        sustain_p: np.ndarray,
        harmonic_curve: np.ndarray,
        start_frame: int,
        max_frames: int,
    ) -> int:
        """Walk forward from start_frame until sustain falls below the down
        threshold (with hysteresis) OR the harmonic energy decays to 10% of
        the value at start_frame. Returns the final in-hold frame index.
        """
        T = sustain_p.shape[0]
        end_cap = min(T, start_frame + max_frames)
        if start_frame >= T - 1:
            return start_frame
        start_harm = float(harmonic_curve[start_frame]) if harmonic_curve is not None else 1.0
        decay_floor = max(start_harm * 0.10, 1e-3)
        f = start_frame + 1
        while f < end_cap:
            if sustain_p[f] < self.sustain_threshold_down:
                return f - 1
            if harmonic_curve is not None and harmonic_curve[f] < decay_floor:
                return f - 1
            f += 1
        return end_cap - 1

    def _postprocess(
        self,
        onset_p: np.ndarray,
        sustain_p: np.ndarray,
        intensity: np.ndarray,
        tempo: float,
        beat_times: np.ndarray,
        duration: float,
        diff_config,
        energy_curve: np.ndarray,
        brightness_curve: np.ndarray,
        harmonic_curve: np.ndarray,
        audio_seed: int,
        style_knobs: Dict[str, float],
        timing_offset: float = 0.0,
        arrow_features: Optional[np.ndarray] = None,
        pitch_class_curve: Optional[np.ndarray] = None,
    ) -> List[Step]:
        """
        Detect WHEN (NMS on onset_p) → classify type (jump/hold/tap) →
        assign arrows: the learned arrow head when `arrow_features` is provided
        (checkpoint has one), else FootStateArrowAssigner driven by style knobs.

        ``timing_offset`` (seconds) is the per-song calibration baked into every
        emitted step time; negative pulls notes earlier.
        """
        T = onset_p.shape[0]
        logger.info(
            f"[postprocess debug] onset_p stats: max={onset_p.max():.4f} "
            f"mean={onset_p.mean():.4f} p95={np.percentile(onset_p, 95):.4f}"
        )

        min_gap = max(diff_config.min_gap, self.min_note_gap)
        nms_window_frames = max(1, int(round(min_gap * FRAMES_PER_SECOND)))
        jumps_allowed = 'jump' in diff_config.allowed_patterns
        holds_allowed = diff_config.hold_percentage > 0

        first_allowed = max(0.0, self.min_first_step_time)
        last_allowed = duration - max(0.0, self.min_last_step_buffer)

        # NMS peak picking on onset_p.
        peaks: List[Tuple[int, float, float]] = []
        for frame in range(T):
            p = float(onset_p[frame])
            if p < max(self.confidence_threshold, 1e-4):
                continue
            lo = max(0, frame - nms_window_frames)
            hi = min(T, frame + nms_window_frames + 1)
            if p < onset_p[lo:hi].max():
                continue
            t = frame / FRAMES_PER_SECOND
            if t < first_allowed or t > last_allowed:
                continue
            peaks.append((frame, t, p))

        # Classify each peak.
        jump_threshold = float(style_knobs.get('jump_threshold', self.intensity_threshold_base))
        target_jump_rate = float(style_knobs.get('target_jump_rate', 0.0))
        jumphold_rate = float(style_knobs.get('jumphold_rate', 0.0))
        jumphold_rms_thr = float(style_knobs.get('jumphold_rms_thr', 0.7))

        max_hold_frames = max(1, int(round(diff_config.max_hold_duration * FRAMES_PER_SECOND)))
        rng = random.Random(audio_seed ^ 0xA5A5A5A5)

        # Top-K jump selection: rank peaks by predicted intensity, mark the top
        # `target_jump_rate * N` (above the absolute floor) as jumps. Robust
        # to the regression head's absolute scale — MSE on the sparse-jump
        # target compresses predictions toward 0, so an absolute threshold
        # would silence all jumps even though the relative ranking is sound.
        if peaks and jumps_allowed and target_jump_rate > 0.0:
            peak_intens = np.fromiter(
                (float(intensity[f]) for f, _, _ in peaks),
                dtype=np.float32, count=len(peaks),
            )
            eligible = np.where(peak_intens >= jump_threshold)[0]
            target_k = int(round(target_jump_rate * len(peaks)))
            if eligible.size > 0 and target_k > 0:
                top = eligible[np.argsort(-peak_intens[eligible])[:target_k]]
                jump_peak_indices: Set[int] = {int(i) for i in top}
            else:
                jump_peak_indices = set()
            logger.info(
                f"[jump debug] peaks={len(peaks)} intensity max={peak_intens.max():.3f} "
                f"p90={np.percentile(peak_intens, 90):.3f} floor={jump_threshold:.3f} "
                f"target_rate={target_jump_rate:.3f} → selected={len(jump_peak_indices)}"
            )
        else:
            jump_peak_indices = set()
            logger.info(
                f"[jump debug] peaks={len(peaks)} jumps_allowed={jumps_allowed} "
                f"target_rate={target_jump_rate:.3f} → selected=0"
            )

        note_events = []
        for peak_idx, (frame, t, confidence) in enumerate(peaks):
            is_hold_start = (
                holds_allowed
                and sustain_p[frame] >= self.sustain_threshold_up
            )
            is_jump = peak_idx in jump_peak_indices

            if is_hold_start:
                end_frame = self._find_hold_end(
                    sustain_p, harmonic_curve, frame, max_hold_frames,
                )
                hold_seconds = max(
                    diff_config.min_hold_duration,
                    min(
                        diff_config.max_hold_duration,
                        (end_frame - frame) / FRAMES_PER_SECOND,
                    ),
                )
                # Jumphold heuristic: rare; only when both intensity is high
                # and the audio energy spikes hard at hold onset.
                rms_at_frame = float(energy_curve[frame]) if energy_curve is not None else 0.0
                wants_jumphold = (
                    is_jump
                    and rms_at_frame > jumphold_rms_thr
                    and jumphold_rate > 0.0
                    and rng.random() < jumphold_rate
                )
                note_events.append({
                    'frame': frame, 'time': t, 'type': 'hold',
                    'confidence': confidence,
                    'hold_duration': hold_seconds,
                    'num_arrows': 2 if wants_jumphold else 1,
                })
            elif is_jump:
                note_events.append({
                    'frame': frame, 'time': t, 'type': 'tap',
                    'confidence': confidence, 'num_arrows': 2,
                })
            else:
                note_events.append({
                    'frame': frame, 'time': t, 'type': 'tap',
                    'confidence': confidence, 'num_arrows': 1,
                })

        # Density cap — keep top-N by confidence (holds are protected).
        avg_density = (diff_config.min_density + diff_config.max_density) / 2.0
        target_notes = max(1, int(round(avg_density * duration)))
        logger.info(
            f"[postprocess debug] peaks={len(peaks)} note_events={len(note_events)} "
            f"target_notes={target_notes} (density={avg_density:.2f}/s, dur={duration:.1f}s)"
        )
        if len(note_events) > target_notes:
            holds = [e for e in note_events if e['type'] == 'hold']
            taps = [e for e in note_events if e['type'] == 'tap']
            taps.sort(key=lambda e: e['confidence'], reverse=True)
            remaining = max(0, target_notes - len(holds))
            note_events = holds + taps[:remaining]

        # Beat-snap: anchor each event to the nearest librosa beat, enumerate
        # candidate cells {4th, 8th, 16th, 12th} within that beat, and pick
        # the one with the smallest effective distance (real distance plus a
        # priority penalty 4th < 8th < 16th < 12th). If the chosen cell time
        # collides with a recently-snapped event's cell, escalate to the next
        # candidate so the two events spread across subdivisions instead of
        # collapsing onto the same grid line.
        if self.snap_to_beats and len(beat_times) > 1:
            bts = np.asarray(beat_times, dtype=np.float64)
            intervals = np.diff(bts)
            local_interval = np.empty_like(bts)
            local_interval[0] = intervals[0]
            local_interval[-1] = intervals[-1]
            if intervals.size > 1:
                local_interval[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])

            # (fraction-within-beat, subdivision label, priority index).
            BEAT_CELLS = [
                (0.0,       BeatSubdivision.QUARTER,    0),
                (0.5,       BeatSubdivision.EIGHTH,     1),
                (0.25,      BeatSubdivision.SIXTEENTH,  2),
                (0.75,      BeatSubdivision.SIXTEENTH,  2),
                (1.0 / 3.0, BeatSubdivision.TWELFTH,    3),
                (2.0 / 3.0, BeatSubdivision.TWELFTH,    3),
            ]
            # Penalty per priority step, scaled by local 16th cell width.
            # 0.20 keeps a clean triplet (real_dist≈0) winning over a far
            # 16th, while letting marginal-16th events stay binary.
            PRIORITY_EPS_FRAC = 0.20
            COLLISION_TOL = 1e-3  # seconds

            note_events.sort(key=lambda e: e['time'])
            recent_claims: List[float] = []

            for event in note_events:
                t = float(event['time'])
                idx = int(np.searchsorted(bts, t))
                beat_idxs: List[int] = []
                if idx > 0:
                    beat_idxs.append(idx - 1)
                if idx < bts.size:
                    beat_idxs.append(idx)
                if not beat_idxs:
                    continue

                # Drop claims older than one beat from the current event so
                # collision-avoidance stays a local effect.
                step_16_local = float(local_interval[beat_idxs[0]]) / 4.0
                cap = 0.5 * step_16_local
                cutoff = t - float(local_interval[beat_idxs[0]])
                recent_claims = [c for c in recent_claims if c >= cutoff]

                # Build candidate list across nearby beats.
                cands: List[Tuple[float, float, BeatSubdivision]] = []
                for c in beat_idxs:
                    beat_t = float(bts[c])
                    interval = float(local_interval[c])
                    eps = (interval / 4.0) * PRIORITY_EPS_FRAC
                    for frac, sub, prio in BEAT_CELLS:
                        cell_t = beat_t + frac * interval
                        real = abs(t - cell_t)
                        if real > cap:
                            continue
                        eff = real + eps * prio
                        cands.append((eff, cell_t, sub))

                if not cands:
                    continue

                cands.sort(key=lambda x: x[0])

                chosen: Optional[Tuple[float, float, BeatSubdivision]] = None
                for cand in cands:
                    cell_t = cand[1]
                    if any(abs(cell_t - r) < COLLISION_TOL for r in recent_claims):
                        continue
                    chosen = cand
                    break
                if chosen is None:
                    chosen = cands[0]  # all in-range cells claimed; accept dup

                event['time'] = float(chosen[1])
                event['subdivision'] = chosen[2]
                recent_claims.append(chosen[1])

        # Sort and enforce min-gap.
        note_events.sort(key=lambda e: e['time'])
        filtered = []
        last_time = -1.0
        for event in note_events:
            if event['time'] - last_time >= min_gap:
                filtered.append(event)
                last_time = event['time']
        note_events = filtered

        # Concurrency cap: at any instant the chart shows at most
        # MAX_CONCURRENT_ARROWS arrows (active hold bodies + new taps/jumps).
        # A jump/jumphold counts as 2. If a new event would exceed the cap,
        # demote arity (jump→single, jumphold→single hold). If even a single
        # arrow won't fit (holds already saturate), drop the event.
        MAX_CONCURRENT_ARROWS = 2
        active_until: List[float] = []
        kept: List[dict] = []
        for event in note_events:
            active_until = [end for end in active_until if end > event['time']]
            free_slots = MAX_CONCURRENT_ARROWS - len(active_until)
            if free_slots <= 0:
                continue  # 2 holds already on screen → drop tap/jump/hold
            arity = int(event.get('num_arrows', 1))
            if arity > free_slots:
                arity = free_slots
                event['num_arrows'] = arity
            if event['type'] == 'hold':
                end_time = event['time'] + event['hold_duration']
                for _ in range(arity):
                    active_until.append(end_time)
            kept.append(event)
        note_events = kept

        # Learned step-selection: when the checkpoint has a trained arrow head,
        # decode a panel set per event up front. The assigner below is still
        # constructed — it tracks hold state for the safety net — but the arrow
        # *choice* comes from the model. Older (headless) checkpoints leave this
        # None and fall back to the assigner's algorithmic choice.
        learned_arrows: Optional[List[List[Direction]]] = None
        if self.has_arrow_head and arrow_features is not None and note_events:
            learned_arrows = self._decode_arrows(arrow_features, note_events)

        # Arrow assignment.
        assigner = FootStateArrowAssigner(
            seed=audio_seed,
            allowed_patterns=diff_config.allowed_patterns,
            jack_rate=style_knobs.get('jack_rate', DEFAULT_STYLE_KNOBS['jack_rate']),
            crossover_rate=style_knobs.get('crossover_rate', DEFAULT_STYLE_KNOBS['crossover_rate']),
            stream_preference=style_knobs.get('stream_preference', DEFAULT_STYLE_KNOBS['stream_preference']),
            candle_rate=style_knobs.get('candle_rate', DEFAULT_STYLE_KNOBS['candle_rate']),
            spin_rate=style_knobs.get('spin_rate', DEFAULT_STYLE_KNOBS['spin_rate']),
        )
        steps: List[Step] = []
        max_stream_length = max(1, getattr(diff_config, 'max_stream_length', 0))

        for i, event in enumerate(note_events):
            t = round(event['time'], 3)
            # Emitted (player-facing) time carries the global timing
            # calibration; `t` stays on the analyzed grid so energy lookups and
            # subdivision derivation reference the true musical position.
            emit_t = round(max(0.0, event['time'] + timing_offset), 3)
            subdivision = event.get('subdivision') or self._subdivision_from_grid(
                t, tempo, beat_times,
            )
            frame_idx = int(round(t * FRAMES_PER_SECOND))
            frame_idx = max(0, min(frame_idx, T - 1))

            if energy_curve is not None and len(energy_curve) > 0:
                energy = float(energy_curve[min(frame_idx, len(energy_curve) - 1)])
            else:
                energy = 0.5
            if brightness_curve is not None and len(brightness_curve) > 0:
                brightness = float(brightness_curve[
                    min(frame_idx, len(brightness_curve) - 1)
                ])
            else:
                brightness = 0.5
            if pitch_class_curve is not None and len(pitch_class_curve) > 0:
                pitch_class = int(pitch_class_curve[
                    min(frame_idx, len(pitch_class_curve) - 1)
                ])
            else:
                pitch_class = -1

            target_arity = 2 if event.get('num_arrows', 1) >= 2 else 1

            # Learned-arrow branch: model already chose the panel set (legality
            # and concurrency enforced during decode). Register holds for the
            # safety net and emit; skip the assigner's algorithmic choice.
            if learned_arrows is not None:
                arrows = list(learned_arrows[i])
                if event['type'] == 'hold' and arrows:
                    end_time = t + event['hold_duration']
                    for a in arrows:
                        assigner.active_holds[a] = end_time
                    steps.append(Step(
                        time=emit_t, arrows=arrows, step_type=StepType.HOLD,
                        hold_duration=round(event['hold_duration'], 3),
                        beat_subdivision=subdivision,
                    ))
                elif arrows:
                    held_now = assigner._held_arrows(t)
                    if any(a in held_now for a in arrows):
                        continue
                    steps.append(Step(
                        time=emit_t, arrows=arrows, step_type=StepType.TAP,
                        beat_subdivision=subdivision,
                    ))
                continue

            if max_stream_length > 0:
                assigner.maybe_start_stream(
                    energy=energy,
                    remaining_events=len(note_events) - i,
                    max_stream_length=max_stream_length,
                    time=t,
                )

            if event['type'] == 'hold':
                if target_arity >= 2:
                    # Jumphold: assign two arrows, register both as active holds.
                    arrows = assigner.assign_jump(t, energy, brightness, pitch_class)
                    if arrows:
                        end_time = t + event['hold_duration']
                        for a in arrows:
                            assigner.active_holds[a] = end_time
                else:
                    arrows = [assigner.start_hold(
                        t, event['hold_duration'], energy, brightness, pitch_class,
                    )]
                steps.append(Step(
                    time=emit_t,
                    arrows=arrows,
                    step_type=StepType.HOLD,
                    hold_duration=round(event['hold_duration'], 3),
                    beat_subdivision=subdivision,
                ))
            else:
                if target_arity >= 2:
                    arrows = assigner.assign_jump(t, energy, brightness, pitch_class)
                else:
                    arrows = [assigner.assign_single(t, energy, brightness, pitch_class)]
                # Safety net: drop the step if any assigned arrow is still
                # held (e.g. all four panels occupied by jumpholds).
                held_now = assigner._held_arrows(t)
                if any(a in held_now for a in arrows):
                    continue
                steps.append(Step(
                    time=emit_t,
                    arrows=arrows,
                    step_type=StepType.TAP,
                    beat_subdivision=subdivision,
                ))

        return steps

    def _subdivision_from_grid(
        self,
        time: float,
        tempo: float,
        beat_times: np.ndarray,
    ) -> BeatSubdivision:
        """Positional fallback for events that didn't snap to a beat-anchored
        grid cell (too far from any beat or beat tracking unavailable)."""
        if len(beat_times) == 0 or tempo <= 0:
            return BeatSubdivision.QUARTER

        # Anchor to the nearest preceding beat and use the local beat spacing so
        # the position stays correct under tempo changes (a global 60/tempo
        # interval drifts out of phase on dynamic-tempo songs and mis-colors
        # otherwise on-beat events).
        bts = np.asarray(beat_times, dtype=np.float64)
        idx = int(np.searchsorted(bts, time)) - 1
        idx = max(0, min(idx, bts.size - 1))
        anchor = float(bts[idx])
        if idx + 1 < bts.size:
            beat_interval = float(bts[idx + 1] - bts[idx])
        elif idx > 0:
            beat_interval = float(bts[idx] - bts[idx - 1])
        else:
            beat_interval = 60.0 / tempo
        if beat_interval <= 0:
            beat_interval = 60.0 / tempo
        beat_position = ((time - anchor) % beat_interval) / beat_interval

        # Triplet positions (1/3, 2/3) win when within 1/24 of a beat. The
        # 1/24 band sits exactly halfway between the 16th grid (1/4, 3/4)
        # and the 12th grid (1/3, 2/3).
        TRIPLET_BAND = 1.0 / 24.0
        if (
            abs(beat_position - 1.0 / 3.0) < TRIPLET_BAND
            or abs(beat_position - 2.0 / 3.0) < TRIPLET_BAND
        ):
            return BeatSubdivision.TWELFTH

        if beat_position < 0.125 or beat_position > 0.875:
            return BeatSubdivision.QUARTER
        if abs(beat_position - 0.5) < 0.125:
            return BeatSubdivision.EIGHTH
        return BeatSubdivision.SIXTEENTH
