"""
Inference module: generate step charts from audio using the v7 hybrid model.

The model emits three dense per-frame channels (onset / intensity / direction).
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
from src.charts.schemas import (
    Chart, Step, StepType, Direction, BeatSubdivision,
)
from src.generation.constants import get_difficulty_config
from ml.playability import (
    PANEL_X as _PANEL_X,
    spec_for as playability_spec,
)

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


# Defaults for style-profile post-processing knobs. These are used when a
# profile doesn't explicitly carry the field (e.g. a hand-edited profile, or
# the legacy "auto" path before clustering has run). Conservative values that
# work on all difficulties.
DEFAULT_STYLE_KNOBS: Dict[str, float] = {
    'jump_threshold': 0.05,       # min intensity for a peak to be jump-eligible
    'target_jump_rate': 0.05,     # fraction of peaks to mark as jumps (top-K)
    'jumphold_rate': 0.0,         # 0 disables jumpholds
    'jumphold_rms_thr': 0.7,
    'hold_rate': 0.04,            # fraction of notes promoted to holds (top-K)
    'avg_hold_beats': 2.0,        # corpus-typical hold length, in beats
    'jack_rate': 0.05,
    'crossover_rate': 0.10,
    'stream_preference': 0.5,
}


class FootFlowAssigner:
    """Whole-timeline, foot-flow arrow assignment.

    Consumes the *finalized* ``note_events`` timeline (times, tap/hold/jump type,
    and beat subdivisions are already decided by ``_postprocess``) and chooses the
    arrows. Unlike the v6/v7 online assigner, it:

    * Picks the FOOT before the panel, so the foot model and the panel it lands on
      can never disagree (kills the desync -> forced-double-step bug).
    * Keeps each foot on its own panels by default (left->LEFT/DOWN, right->UP/RIGHT),
      which is un-crossed and double-step-free *by construction*.
    * Only jacks when the local interval clears the per-difficulty ``jack_min_dt``.
    * Only crosses over when allowed and spacing-safe, and lets the surrounding
      alternation keep the crossed run short (validated by ``ml.playability``;
      the harness reports any run that exceeds the difficulty's budget).
    * Derives streams from the runs already present in the timeline -- no invented
      pattern injection -- and layers musical texture on top: bar-fingerprint motif
      repetition (identical rhythmic figures step identically), strong-beat emphasis
      (outer panels on strong beats), and per-section contrast.

    Determinism: panel textures come from a stable integer fingerprint (not Python's
    randomized ``hash``); a single ``random.Random(seed)`` breaks the few remaining
    ties. A given song regenerates identically.
    """

    OUTER = {'L': Direction.LEFT, 'R': Direction.RIGHT}
    INNER = {'L': Direction.DOWN, 'R': Direction.UP}
    OWN_PANELS = {
        'L': (Direction.LEFT, Direction.DOWN),
        'R': (Direction.UP, Direction.RIGHT),
    }
    ALL_PANELS = (Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT)
    # Column index of each panel, matching the model's arrow head (L/D/U/R).
    COL = {Direction.LEFT: 0, Direction.DOWN: 1, Direction.UP: 2, Direction.RIGHT: 3}
    # Physically sane two-panel jump pairs (all un-crossed).
    JUMP_PATTERNS = [
        (Direction.LEFT, Direction.RIGHT),
        (Direction.DOWN, Direction.UP),
        (Direction.LEFT, Direction.UP),
        (Direction.DOWN, Direction.RIGHT),
    ]
    _SUB_CODE = {
        BeatSubdivision.QUARTER: 0,
        BeatSubdivision.EIGHTH: 1,
        BeatSubdivision.TWELFTH: 2,
        BeatSubdivision.SIXTEENTH: 3,
    }

    def __init__(
        self,
        difficulty: str,
        seed: int = 0,
        allowed_patterns: Optional[List[str]] = None,
        jack_rate: float = 0.05,
        crossover_rate: float = 0.10,
        section_aware: bool = True,
    ):
        self.difficulty = difficulty
        self.spec = playability_spec(difficulty)
        self.rng = random.Random(seed)
        self.allowed = set(allowed_patterns or ['single'])
        self.jack_rate = float(np.clip(jack_rate, 0.0, 1.0))
        self.crossover_rate = float(np.clip(crossover_rate, 0.0, 1.0))
        self.section_aware = section_aware

        self._jacks_ok = difficulty in ('medium', 'hard', 'challenge')
        self._crossovers_ok = (
            'crossover' in self.allowed and self.spec.crossovers_allowed
        )
        self._jumps_ok = 'jump' in self.allowed

    # -- foot / stance state -------------------------------------------------

    def _reset(self):
        self.left_panel = Direction.LEFT
        self.right_panel = Direction.RIGHT
        self.last_foot: Optional[str] = 'R'   # first single alternates to L
        self.last_arrow: Optional[Direction] = None
        self.prev_single_time: Optional[float] = None
        # active_holds: panel -> (end_time, foot)
        self.active_holds: Dict[Direction, Tuple[float, str]] = {}
        self.crossed = False
        self.crossed_len = 0
        self.arrow_probs: Optional[np.ndarray] = None  # [T,4] learned direction (v8)
        self._cur_pref: Optional[np.ndarray] = None    # 4-vector for the current event

    def _purge_holds(self, t: float):
        if self.active_holds:
            self.active_holds = {
                p: (e, f) for p, (e, f) in self.active_holds.items() if e > t + 1e-6
            }

    def _pinned_feet(self, t: float) -> set:
        self._purge_holds(t)
        return {f for (_, f) in self.active_holds.values()}

    def _place(self, foot: str, panel: Direction):
        if foot == 'L':
            self.left_panel = panel
        else:
            self.right_panel = panel
        self.last_foot = foot
        self.last_arrow = panel
        self.crossed = _PANEL_X[self.left_panel] > _PANEL_X[self.right_panel]
        self.crossed_len = self.crossed_len + 1 if self.crossed else 0

    # -- public entry point --------------------------------------------------

    def assign(
        self,
        note_events: List[dict],
        energy_curve: Optional[np.ndarray],
        brightness_curve: Optional[np.ndarray],
        beat_times: np.ndarray,
        arrow_probs: Optional[np.ndarray] = None,
    ) -> List[List[Direction]]:
        """Return the arrow list for each event, aligned with ``note_events``.

        ``arrow_probs`` ([T,4] per-frame L/D/U/R probabilities from the v8 learned
        direction head, or None) is a *preference*: it steers panel choice within
        the foot the flow model picked, and jump-pair choice — the playability
        constraints still decide the foot and veto anything ungettable.
        """
        self._reset()
        self.arrow_probs = arrow_probs
        bts = np.asarray(beat_times, dtype=np.float64) if beat_times is not None else np.array([])
        sections = self._detect_sections(note_events, energy_curve)
        bars, textures, strongs = self._rhythm_features(note_events, bts)

        out: List[List[Direction]] = []
        cur_section = -1
        for i, ev in enumerate(note_events):
            t = float(ev['time'])
            if self.section_aware and sections[i] != cur_section:
                cur_section = sections[i]  # section contrast: flip base texture
            energy = self._sample(energy_curve, t)
            num = int(ev.get('num_arrows', 1))
            texture = textures[i] ^ (cur_section & 1)
            strong = strongs[i]
            self._cur_pref = self._pref_at(t)

            if ev['type'] == 'hold':
                arrows = self._assign_hold(ev, t, num, energy, strong, texture)
            elif num >= 2 and self._jumps_ok:
                arrows = self._assign_jump(t, energy, strong)
            else:
                a = self._assign_single(t, energy, strong, texture)
                arrows = [a] if a is not None else []
            out.append(arrows)
        return out

    # -- single-note assignment ---------------------------------------------

    def _free_foot(self, t: float) -> Optional[str]:
        pinned = self._pinned_feet(t)
        if 'L' in pinned and 'R' in pinned:
            return None  # both feet committed to holds
        if 'L' in pinned:
            return 'R'
        if 'R' in pinned:
            return 'L'
        return 'L' if self.last_foot == 'R' else 'R'  # both free -> alternate

    def _assign_single(self, t, energy, strong, texture) -> Optional[Direction]:
        dt = (t - self.prev_single_time) if self.prev_single_time is not None else 1e9
        foot = self._free_foot(t)
        if foot is None:
            return None  # both feet holding; concurrency cap should prevent this
        held = set(self.active_holds)

        # 1) Deliberate jack: same foot repeats the same panel. Only when spacing is
        #    physically repeatable and nothing is being held.
        if (
            self._jacks_ok
            and self.last_arrow is not None
            and self.last_foot is not None
            and dt >= self.spec.jack_min_dt
            and not self.active_holds
            and self.last_arrow not in held
            and self.rng.random() < self.jack_rate
        ):
            self._place(self.last_foot, self.last_arrow)
            self.prev_single_time = t
            return self.last_arrow

        # 2) Bounded crossover: step the free foot onto the opposite side. The next
        #    alternation naturally returns, so the crossed run stays short. Guarded
        #    against colliding with the other foot's current panel or a hold.
        if (
            self._crossovers_ok
            and not self.crossed
            and dt >= self.spec.jack_min_dt
            and self.rng.random() < self.crossover_rate
        ):
            other_panel = self.right_panel if foot == 'L' else self.left_panel
            cross_panel = self.OUTER['R' if foot == 'L' else 'L']
            if cross_panel not in held and cross_panel != other_panel:
                self._place(foot, cross_panel)
                self.prev_single_time = t
                return cross_panel

        # Panels to steer away from: the other foot's current panel (collision) and,
        # when notes are too close to jack, the panel we just hit (accidental jack).
        other_panel = self.right_panel if foot == 'L' else self.left_panel
        avoid = {other_panel}
        if self.last_arrow is not None and dt < self.spec.jack_min_dt:
            avoid.add(self.last_arrow)

        # 3) If the crossed run has reached the difficulty's budget, force the
        #    crossed foot home so the run ends within uncross_within. The crossed
        #    foot is also the one alternation wants next, so this isn't a double-step.
        if self.crossed and self.crossed_len >= self.spec.uncross_within:
            uf, upanel = self._forced_uncross(held)
            self._place(uf, upanel)
            self.prev_single_time = t
            return upanel

        # 4) Normal musical placement on the foot's own panels (also un-crosses
        #    gradually while a crossover is in flight).
        panel = self._pick_own_panel(foot, strong, held, texture, avoid)
        self._place(foot, panel)
        self.prev_single_time = t
        return panel

    def _forced_uncross(self, held) -> Tuple[str, Direction]:
        """Return (foot, panel) that is guaranteed to leave an un-crossed stance:
        pull the left foot to LEFT (min x) or the right foot to RIGHT (max x)."""
        if (Direction.LEFT not in held and self.right_panel != Direction.LEFT):
            return 'L', Direction.LEFT
        if (Direction.RIGHT not in held and self.left_panel != Direction.RIGHT):
            return 'R', Direction.RIGHT
        # Degenerate fallback: any free own-side panel.
        if Direction.DOWN not in held:
            return 'L', Direction.DOWN
        return 'R', Direction.UP

    def _pick_own_panel(self, foot, strong, held, texture, avoid=frozenset()) -> Direction:
        outer, inner = self.OUTER[foot], self.INNER[foot]
        if self._cur_pref is not None:
            # Learned direction proposes: pick the foot's panel the model prefers.
            want_outer = float(self._cur_pref[self.COL[outer]]) >= float(self._cur_pref[self.COL[inner]])
        else:
            # Strong beats emphasize the outer (side) arrow; otherwise the
            # deterministic per-rhythm texture bit chooses, so identical figures
            # step identically.
            want_outer = strong or bool(texture)
        first = outer if want_outer else inner
        second = inner if want_outer else outer
        for cand in (first, second):        # prefer own panels, honoring avoid
            if cand not in held and cand not in avoid:
                return cand
        for cand in (first, second):        # relax avoid before leaving own panels
            if cand not in held:
                return cand
        for cand in self.ALL_PANELS:         # both own panels held (rare)
            if cand not in held and cand not in avoid:
                return cand
        for cand in self.ALL_PANELS:
            if cand not in held:
                return cand
        return first

    # -- hold assignment -----------------------------------------------------

    def _assign_hold(self, ev, t, num, energy, strong, texture):
        """Place a derived hold, demoting it to a tap rather than dropping it if
        the stance can't accommodate one.

        ``ev`` is mutated on demotion so ``_postprocess`` emits the right
        ``StepType`` — a hold whose arrows were reassigned as a tap must not
        keep its ``hold_duration``.
        """
        if num >= 2 and self._jumps_ok:
            # _assign_jump has already committed the stance, so whatever it
            # returns is the answer — it may have degraded to a single arrow if
            # a foot was pinned. Register a hold on each arrow it did place.
            arrows = self._assign_jump(t, energy, strong)
            if not arrows:
                self._demote_to_tap(ev)
                return []
            end = t + float(ev['hold_duration'])
            for a in arrows:
                foot = 'L' if a in self.OWN_PANELS['L'] else 'R'
                self.active_holds[a] = (end, foot)
            return arrows
        foot = self._free_foot(t)
        if foot is None:
            # Both feet are committed to holds; the concurrency cap should have
            # prevented this event from surviving at all.
            self._demote_to_tap(ev)
            return []
        held = set(self.active_holds)
        # Same jack/collision avoidance as singles: don't start a hold on the panel
        # we just tapped (quick re-press) or on the other foot's current panel.
        dt = (t - self.prev_single_time) if self.prev_single_time is not None else 1e9
        other_panel = self.right_panel if foot == 'L' else self.left_panel
        avoid = {other_panel}
        if self.last_arrow is not None and dt < self.spec.jack_min_dt:
            avoid.add(self.last_arrow)
        panel = self._pick_own_panel(foot, strong, held, texture, avoid)
        self._place(foot, panel)
        self.prev_single_time = t
        if panel in held:
            # _pick_own_panel exhausted its options and fell back to an occupied
            # panel. Starting a second hold there would clobber the active one's
            # end time, so emit a plain tap instead.
            self._demote_to_tap(ev)
            return [panel]
        self.active_holds[panel] = (t + float(ev['hold_duration']), foot)
        return [panel]

    @staticmethod
    def _demote_to_tap(ev: dict) -> None:
        """Turn a hold event back into a tap, in place."""
        ev['type'] = 'tap'
        ev.pop('hold_duration', None)

    # -- jump assignment -----------------------------------------------------

    def _assign_jump(self, t, energy, strong) -> List[Direction]:
        if self._pinned_feet(t):
            a = self._assign_single(t, energy, strong, 0)  # a foot is holding -> tap
            return [a] if a is not None else []
        held = set(self.active_holds)
        if energy > 0.7:
            candidates = [self.JUMP_PATTERNS[0], self.JUMP_PATTERNS[2], self.JUMP_PATTERNS[3]]
        elif energy < 0.3:
            candidates = [self.JUMP_PATTERNS[1]]
        else:
            candidates = list(self.JUMP_PATTERNS)
        if strong:
            candidates = [self.JUMP_PATTERNS[0]] + candidates  # emphasize L/R on strong beats
        if self._cur_pref is not None:
            # Learned direction proposes the pair: rank by summed column preference.
            candidates = sorted(
                candidates,
                key=lambda p: -(float(self._cur_pref[self.COL[p[0]]])
                                + float(self._cur_pref[self.COL[p[1]]])),
            )
        else:
            self.rng.shuffle(candidates)
        for a, b in candidates:
            if a not in held and b not in held:
                self.left_panel, self.right_panel = a, b
                self.last_foot = None  # either foot may lead the next note
                self.last_arrow = b
                self.crossed = False
                return [a, b]
        a = self._assign_single(t, energy, strong, 0)  # all collide -> tap fallback
        return [a] if a is not None else []

    # -- rhythm / musical helpers -------------------------------------------

    def _sample(self, curve: Optional[np.ndarray], t: float) -> float:
        if curve is None or len(curve) == 0:
            return 0.5
        idx = int(round(t * FRAMES_PER_SECOND))
        idx = max(0, min(idx, len(curve) - 1))
        return float(curve[idx])

    def _pref_at(self, t: float) -> Optional[np.ndarray]:
        """The learned [L,D,U,R] preference vector at time ``t``, or None."""
        if self.arrow_probs is None or len(self.arrow_probs) == 0:
            return None
        idx = int(round(t * FRAMES_PER_SECOND))
        idx = max(0, min(idx, len(self.arrow_probs) - 1))
        return self.arrow_probs[idx]

    def _rhythm_features(self, note_events, bts):
        """Per-event (bar index, texture bit, strong-beat flag).

        The texture bit is a stable function of the bar's rhythmic fingerprint and
        the note's position in the bar, so two bars with the same rhythm produce the
        same outer/inner footing texture -- motif repetition without storing state.
        """
        n = len(note_events)
        bars = [0] * n
        strongs = [False] * n
        interval = float(np.median(np.diff(bts))) if bts.size > 1 else 0.5
        for i, ev in enumerate(note_events):
            t = float(ev['time'])
            if bts.size >= 2:
                bi = int(np.searchsorted(bts, t))
                bars[i] = max(0, (bi - 1) // 4)
                nearest = min((abs(t - bts[j]) for j in (bi - 1, bi) if 0 <= j < bts.size),
                              default=1e9)
                strongs[i] = nearest < 0.15 * interval
            sub = ev.get('subdivision')
            if sub is not None and not strongs[i]:
                strongs[i] = (sub == BeatSubdivision.QUARTER)

        # Fingerprint each bar from its notes' subdivisions, then derive a texture
        # bit per note position via a stable integer mix (NOT Python's hash()).
        from itertools import groupby
        textures = [0] * n
        idx = 0
        for _, grp in groupby(range(n), key=lambda k: bars[k]):
            members = list(grp)
            fp = 1469598103  # FNV-ish seed
            for k in members:
                code = self._SUB_CODE.get(note_events[k].get('subdivision'), 0)
                fp = (fp * 16777619 + code + 1) & 0xFFFFFFFF
            for pos, k in enumerate(members):
                mix = (fp * 2654435761 + pos * 40503) & 0xFFFFFFFF
                textures[k] = (mix >> 16) & 1
        return bars, textures, strongs

    def _detect_sections(self, note_events, energy_curve) -> List[int]:
        """Coarse section id per event from energy-curve novelty. Cheap and bounded;
        only used to contrast footing texture between sections."""
        n = len(note_events)
        if (not self.section_aware or energy_curve is None
                or len(energy_curve) == 0 or n == 0):
            return [0] * n
        e = np.asarray(energy_curve, dtype=np.float64)
        win = max(1, int(2.0 * FRAMES_PER_SECOND))  # ~2s smoothing
        sm = np.convolve(e, np.ones(win) / win, mode='same')
        step = max(1, int(4.0 * FRAMES_PER_SECOND))
        boundaries = []
        last_val = sm[0] if len(sm) else 0.0
        for f in range(0, len(sm), step):
            if abs(sm[f] - last_val) > 0.25:
                boundaries.append(f / FRAMES_PER_SECOND)
                last_val = sm[f]
        if not boundaries:
            return [0] * n
        return [int(np.searchsorted(boundaries, float(ev['time']))) for ev in note_events]


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

    # Hold derivation (see _derive_holds): the corpus hold_rate sets how many
    # notes get promoted, and avg_hold_beats biases the quantized length toward
    # what this style actually writes.
    knobs['hold_rate'] = float(np.clip(hold_rate, 0.0, 0.5))
    knobs['avg_hold_beats'] = float(np.clip(rm.get('avg_hold_beats', 2.0), 0.5, 8.0))

    knobs['jack_rate'] = float(np.clip(rm.get('jack_rate', 0.05), 0.0, 0.5))
    knobs['crossover_rate'] = float(np.clip(rm.get('crossover_rate', 0.10), 0.0, 0.5))
    # stream_preference rolls stream_density into the assigner's branch
    # probability — high values produce visibly more stream patterns.
    knobs['stream_preference'] = float(np.clip(
        rm.get('stream_density', 0.20) * 2.0, 0.0, 1.0,
    ))
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
        self.intensity_threshold_base = float(intensity_threshold_base)
        self.min_first_step_time = float(min_first_step_time)
        self.min_last_step_buffer = float(min_last_step_buffer)
        # Manual timing trim (seconds) added on top of the per-song measured
        # offset (see _analyze_audio). Negative pulls notes earlier. Normally 0;
        # exists only for fine-tuning beyond the auto-calibrated value.
        self.timing_trim = float(timing_trim)
        self.style = str(style)

        self.model: Optional[StepChartModel] = None
        self.use_learned_arrows: bool = False
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
        # v7 (onset/sustain/intensity), v8 (+ learned direction head) and v9
        # (sustain head removed) are all loadable. v7/v8 checkpoints carry
        # sustain_head weights that this architecture no longer has; they land
        # in `unexpected` below and are ignored, which is exactly right — holds
        # are derived algorithmically now, so the head's output is unused. The
        # learned direction is only consumed when the checkpoint actually
        # carries a trained arrow head.
        if arch_version not in (7, 8, 9):
            raise ValueError(
                f"Checkpoint at {model_path} is arch_version={arch_version}; "
                f"expected 7, 8 or 9. Retrain with current model.py."
            )
        self.use_learned_arrows = arch_version >= 8

        n_in_channels = int(checkpoint.get('n_in_channels', N_FEAT_CHANNELS))
        args = checkpoint.get('args', {})
        self.model = StepChartModel(
            n_in_channels=n_in_channels,
            hidden_dim=args.get('hidden_dim', 160),
            n_heads=args.get('n_heads', 4),
            n_transformer_layers=args.get('n_layers', 3),
            n_difficulties=5,
            with_arrow_head=self.use_learned_arrows,
        ).to(self.device)
        self.n_in_channels = n_in_channels

        state_dict = checkpoint.get('ema_state_dict') or checkpoint['model_state_dict']
        if checkpoint.get('ema_state_dict') is not None:
            logger.info("Loading EMA weights from checkpoint.")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Checkpoint missing keys (zero-initialized): {missing}")
        if unexpected:
            sustain_keys = [k for k in unexpected if k.startswith('sustain_head.')]
            other = [k for k in unexpected if k not in sustain_keys]
            if sustain_keys:
                logger.info(
                    f"Ignoring {len(sustain_keys)} sustain_head weights from this "
                    f"arch_version={arch_version} checkpoint (holds are derived "
                    f"algorithmically since v9)."
                )
            if other:
                logger.warning(f"Checkpoint had unexpected keys (ignored): {other}")
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
        )

    def _chart_from_signals(
        self,
        analysis: "_AudioAnalysis",
        difficulty: str,
        onset_p: np.ndarray,
        intensity: np.ndarray,
        style: Optional[str] = None,
        arrow_probs: Optional[np.ndarray] = None,
    ) -> Chart:
        """Build a Chart from already-computed per-difficulty signals.

        Shared by the single-difficulty and batched-multi paths so
        post-processing only lives in one place. ``arrow_probs`` ([T,4], v8 only)
        is the learned per-column direction preference, or None for v7.
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
            onset_p, intensity,
            analysis.tempo, analysis.beat_times, analysis.duration, diff_config,
            energy_curve=analysis.rms_norm,
            brightness_curve=analysis.centroid_norm,
            harmonic_curve=analysis.harm_rms,
            audio_seed=analysis.audio_seed,
            style_knobs=knobs,
            timing_offset=timing_offset,
            arrow_probs=arrow_probs,
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

        onset_p, intensity, arrow_probs = self._predict_chunked(
            analysis.feats_whitened, difficulty_id, target_density,
        )
        return self._chart_from_signals(
            analysis, difficulty, onset_p, intensity, style=style,
            arrow_probs=arrow_probs,
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
        onset_batch, intensity_batch, arrow_batch = self._predict_chunked_multi(
            analysis.feats_whitened, difficulty_ids, target_densities,
        )

        charts: Dict[str, Chart] = {}
        for i, name in enumerate(names):
            charts[name] = self._chart_from_signals(
                analysis, name,
                onset_batch[i], intensity_batch[i],
                style=style,
                arrow_probs=(arrow_batch[i] if arrow_batch is not None else None),
            )
        return charts

    @torch.no_grad()
    def _predict_chunked(
        self,
        feats: np.ndarray,
        difficulty_id: int,
        target_density: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model on overlapping chunks; return (onset_p, intensity).

        Each output is shape [T] float32. onset_p is a sigmoid;
        intensity is the raw regression output (clipped to [0, 1]).
        """
        T = feats.shape[0]
        onset_sum = np.zeros(T, dtype=np.float32)
        intensity_sum = np.zeros(T, dtype=np.float32)
        arrow_sum = np.zeros((T, 4), dtype=np.float32) if self.use_learned_arrows else None
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

            onset_logits, intensity_pred, arrow_logits = self.model(
                feats_tensor, diff_tensor, density_tensor,
                start_seconds_tensor, remaining_seconds_tensor,
            )
            onset_p = torch.sigmoid(onset_logits.float()).cpu().numpy()[0, :, 0]
            intensity_v = intensity_pred.float().cpu().numpy()[0, :, 0]

            valid_len = min(end - start, self.chunk_frames)
            onset_sum[start:start + valid_len] += onset_p[:valid_len]
            intensity_sum[start:start + valid_len] += intensity_v[:valid_len]
            if arrow_sum is not None and arrow_logits is not None:
                arrow_p = torch.sigmoid(arrow_logits.float()).cpu().numpy()[0]  # [T,4]
                arrow_sum[start:start + valid_len, :] += arrow_p[:valid_len, :]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)
        onset_avg = onset_sum / counts
        intensity_avg = np.clip(intensity_sum / counts, 0.0, 1.0)
        arrow_avg = (arrow_sum / counts[:, None]) if arrow_sum is not None else None
        return onset_avg, intensity_avg, arrow_avg

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
        intensity_sum = np.zeros((B, T), dtype=np.float32)
        arrow_sum = np.zeros((B, T, 4), dtype=np.float32) if self.use_learned_arrows else None
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

            onset_logits, intensity_pred, arrow_logits = self.model(
                feats_tensor, diff_tensor, density_tensor,
                start_seconds_tensor, remaining_seconds_tensor,
            )
            onset_p = torch.sigmoid(onset_logits.float()).cpu().numpy()[:, :, 0]
            intensity_v = intensity_pred.float().cpu().numpy()[:, :, 0]

            valid_len = min(end - start, self.chunk_frames)
            onset_sum[:, start:start + valid_len] += onset_p[:, :valid_len]
            intensity_sum[:, start:start + valid_len] += intensity_v[:, :valid_len]
            if arrow_sum is not None and arrow_logits is not None:
                arrow_p = torch.sigmoid(arrow_logits.float()).cpu().numpy()  # [B,T,4]
                arrow_sum[:, start:start + valid_len, :] += arrow_p[:, :valid_len, :]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)  # broadcast across the B axis
        onset_avg = onset_sum / counts
        intensity_avg = np.clip(intensity_sum / counts, 0.0, 1.0)
        arrow_avg = (arrow_sum / counts[None, :, None]) if arrow_sum is not None else None
        return onset_avg, intensity_avg, arrow_avg

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    # Hold lengths are quantized to these beat counts so tails land on the same
    # grid the heads were snapped to.
    HOLD_BEAT_CELLS = (4.0, 3.0, 2.0, 1.5, 1.0, 0.5)
    # Cumulative distribution of hold lengths measured over the training corpus
    # (n=3242): one beat is far and away the mode, with a long right tail.
    # Sampling a cell from this by sustain rank reproduces that shape; mapping a
    # continuous target length onto the nearest cell does not, because the cells
    # are unevenly spaced and the 2-beat cell would swallow half the range.
    HOLD_LENGTH_CDF = (
        (0.5, 0.07), (1.0, 0.54), (1.5, 0.61),
        (2.0, 0.81), (3.0, 0.96), (4.0, 1.00),
    )
    # Median corpus hold length in beats. Dividing a style's typical length by
    # this gives 1.0 for a corpus-average profile, so the bias below is neutral
    # unless the style really does write longer or shorter holds.
    HOLD_CORPUS_TYPICAL_BEATS = 1.24
    # Harmonic energy below this fraction of its value at the onset counts as
    # the note having decayed away.
    HOLD_DECAY_FLOOR = 0.10
    # A hold tail stops this many beats short of the next note, so the foot has
    # time to release and move.
    HOLD_TAIL_LEAD_BEATS = 0.5
    # Style profiles report avg_hold_beats as a MEAN over a strongly
    # right-skewed distribution (corpus: mode 1.0, median 1.24, mean 1.94
    # beats). Preferring the mean would systematically overshoot by ~1 beat, so
    # convert to a typical length with the corpus median/mean ratio.
    HOLD_TYPICAL_OVER_MEAN = 0.65
    # Lookahead for scoring how well a note sustains, in beats.
    HOLD_RETENTION_BEATS = 1.0

    def _audio_hold_end(
        self,
        harmonic_curve: Optional[np.ndarray],
        start_frame: int,
        max_frames: int,
        n_frames: int,
    ) -> int:
        """Walk forward from ``start_frame`` until the harmonic energy decays to
        :attr:`HOLD_DECAY_FLOOR` of its value there. Returns the last frame the
        note is still ringing.

        This is the audio half of the old ``_find_hold_end``; the model-sustain
        half is gone (holds are no longer predicted). Where the note stops
        ringing is what *proposes* a hold — foot feasibility then trims it.
        """
        end_cap = min(n_frames, start_frame + max_frames)
        if start_frame >= n_frames - 1 or harmonic_curve is None:
            return start_frame
        start_harm = float(harmonic_curve[start_frame])
        decay_floor = max(start_harm * self.HOLD_DECAY_FLOOR, 1e-3)
        f = start_frame + 1
        while f < end_cap:
            if harmonic_curve[f] < decay_floor:
                return f - 1
            f += 1
        return end_cap - 1

    @staticmethod
    def _percentile_ranks(values: List[float]) -> List[float]:
        """Map each value to its percentile rank in [0, 1], ties sharing a rank."""
        n = len(values)
        if n == 0:
            return []
        if n == 1:
            return [0.5]
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 / (n - 1)
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    def _sustain_retention(
        self,
        harmonic_curve: Optional[np.ndarray],
        start_frame: int,
        beat_interval: float,
        n_frames: int,
    ) -> float:
        """Ratio of harmonic energy a beat after the onset to energy at it.

        High for a sustained pad or held vocal, near zero for a staccato stab.
        Used only to rank hold candidates against each other, so it has to
        *vary* — unlike the walk-until-decay length, which pins to its cap on
        most material. Deliberately not clipped to 1.0: harmonic energy often
        holds steady or rises after an onset, and clipping there collapses
        almost every note onto the same value, leaving the ranking with no
        information to order by.
        """
        if harmonic_curve is None or start_frame >= n_frames - 1:
            return 0.0
        start = float(harmonic_curve[start_frame])
        if start <= 1e-6:
            return 0.0
        look = min(n_frames - 1, start_frame + max(1, int(round(
            self.HOLD_RETENTION_BEATS * beat_interval * FRAMES_PER_SECOND
        ))))
        window = harmonic_curve[start_frame + 1:look + 1]
        if window.size == 0:
            return 0.0
        return float(np.mean(window)) / start

    def _derive_holds(
        self,
        note_events: List[dict],
        harmonic_curve: Optional[np.ndarray],
        tempo: float,
        beat_times: np.ndarray,
        n_frames: int,
        diff_config,
        style_knobs: Dict[str, float],
    ) -> int:
        """Promote single taps to holds in place. Returns the number promoted.

        Audio proposes, feet veto. A note whose harmonic energy keeps ringing is
        a hold *candidate*; how long it may actually last is bounded by what the
        free foot can still cover, because a hold pins one foot and every note
        underneath it is played by the other one.

        Runs after beat-snapping so tail times land on the grid, and before the
        concurrency cap so derived holds are subject to it like any other event.
        """
        if not note_events or diff_config.hold_percentage <= 0:
            return 0

        target_rate = min(
            float(style_knobs.get('hold_rate', 0.04)),
            float(diff_config.hold_percentage),
        )
        target_k = int(round(target_rate * len(note_events)))
        if target_k <= 0:
            return 0

        bts = np.asarray(beat_times, dtype=np.float64)
        # Quantize against the musical beat, NOT the spacing between entries in
        # ``beat_times``. Those are PLP pulse maxima — irregular, and often a
        # different count than the song has beats — so using them would give
        # holds a jittery, non-musical length.
        if tempo and tempo > 0:
            beat_interval = 60.0 / float(tempo)
        elif bts.size > 1:
            beat_interval = float(np.median(np.diff(bts)))
        else:
            beat_interval = 0.5
        max_notes_under = int(getattr(diff_config, 'max_notes_under_hold', 0))
        spec = playability_spec(diff_config.name)
        jack_min_dt = float(spec.jack_min_dt)
        # A hold pins its foot for the whole span, so that foot has to *arrive*
        # cleanly. Spacing that is fine for a tap can be a forced double-step
        # once the note becomes a hold, hence a lead-in gap covering both the
        # same-panel (jack) and cross-panel (double-step) floors.
        min_lead_in = max(jack_min_dt, float(spec.double_step_min_dt))
        pref_beats = (
            float(style_knobs.get('avg_hold_beats', 2.0)) * self.HOLD_TYPICAL_OVER_MEAN
        )

        candidates: List[Tuple[float, int, float]] = []  # (score, index, budget)
        for i, ev in enumerate(note_events):
            if ev.get('type') != 'tap' or int(ev.get('num_arrows', 1)) != 1:
                continue  # jumps need both feet; only singles become holds

            t = float(ev['time'])

            # 0. The foot taking the hold must be able to get there in time.
            if i > 0 and t - float(note_events[i - 1]['time']) < min_lead_in:
                continue

            # 1. Propose: how long does the audio keep ringing here?
            start_frame = int(round(t * FRAMES_PER_SECOND))
            max_frames = max(1, int(round(diff_config.max_hold_duration * FRAMES_PER_SECOND)))
            end_frame = self._audio_hold_end(
                harmonic_curve, start_frame, max_frames, n_frames,
            )
            audio_dur = (end_frame - start_frame) / FRAMES_PER_SECOND
            if audio_dur <= 0.0:
                continue

            # How much of the onset's harmonic energy is still there a beat
            # later. audio_dur alone saturates at the cap on most material, and
            # tied scores collapse the ranking into "whatever comes first in the
            # song" — this actually separates ringing notes from staccato ones.
            score = self._sustain_retention(
                harmonic_curve, start_frame, beat_interval, n_frames,
            )

            # 2. Trim: how far can the hold run before the free foot can't cope?
            feasible_dur, blocked_by_jump = self._free_foot_window(
                note_events, i, max_notes_under, jack_min_dt,
            )
            if blocked_by_jump:
                # A jump needs the pinned foot back, so leave it room to release
                # and travel. Notes taken by the free foot need no such lead —
                # a tap coinciding with a tail is played by the other foot.
                feasible_dur -= self.HOLD_TAIL_LEAD_BEATS * beat_interval
            if feasible_dur <= 0.0:
                continue

            # 3. Quantize to a beat cell so the tail lands on the grid. A
            #    candidate whose budget can't fit a cell of at least
            #    min_hold_duration is rejected, never floored — the old path
            #    applied max(min_hold_duration, ...), inflating a single-frame
            #    flicker into a 0.3-0.8s hold.
            #    The exact length is settled after ranking (below), since it
            #    depends on this note's sustain *relative to the song*.
            budget = min(audio_dur, feasible_dur, diff_config.max_hold_duration)
            if self._quantize_hold(
                budget, beat_interval, pref_beats, diff_config.min_hold_duration,
            ) is None:
                continue  # window admits no cell at all

            candidates.append((score, i, budget))

        if not candidates:
            logger.info("[hold debug] no viable hold candidates")
            return 0

        # 5. Budget: split the song into as many spans as we want holds and take
        #    the best-ringing candidate in each. Picking the global top-K
        #    instead would bunch every hold into the intro — sparse, sustained
        #    openings beat the rest of the song on both sustain and foot room,
        #    so they win every slot.
        t_first = float(note_events[0]['time'])
        t_last = float(note_events[-1]['time'])
        span = max(t_last - t_first, 1e-6) / target_k
        best_per_bucket: Dict[int, Tuple[float, int, float]] = {}
        for cand in candidates:
            t = float(note_events[cand[1]]['time'])
            bucket = min(target_k - 1, int((t - t_first) / span))
            if bucket not in best_per_bucket or cand[0] > best_per_bucket[bucket][0]:
                best_per_bucket[bucket] = cand

        min_spacing = 4.0 * beat_interval  # roughly one bar apart
        selected: List[Tuple[float, int, float]] = []
        promoted_times: List[float] = []
        for _bucket, cand in sorted(best_per_bucket.items()):
            t = float(note_events[cand[1]]['time'])
            if any(abs(t - p) < min_spacing for p in promoted_times):
                continue
            selected.append(cand)
            promoted_times.append(t)

        # Length is set from each hold's sustain ranked against the *other holds
        # in this chart*. The raw retention ratio sits near 1.0 on most material,
        # and bucket selection already skims the top scorer from every span, so
        # both the absolute value and a corpus-wide rank would bunch at the top
        # and give every hold the same length. Ranking within the selected set
        # spreads them around the style's typical hold instead.
        ranks = self._percentile_ranks([c[0] for c in selected])
        n_promoted = 0
        for (_score, idx, budget), rank in zip(selected, ranks):
            dur = self._quantize_hold(
                budget, beat_interval, pref_beats,
                diff_config.min_hold_duration, rank,
            )
            if dur is None:
                continue  # unreachable: candidacy already proved a cell fits
            note_events[idx]['type'] = 'hold'
            note_events[idx]['hold_duration'] = dur
            n_promoted += 1

        logger.info(
            f"[hold debug] candidates={len(candidates)} target_k={target_k} "
            f"promoted={n_promoted} (rate={target_rate:.3f}, "
            f"max_under={max_notes_under}, jack_min_dt={jack_min_dt:.3f})"
        )
        return n_promoted

    @staticmethod
    def _free_foot_window(
        note_events: List[dict],
        start_idx: int,
        max_notes_under: int,
        jack_min_dt: float,
    ) -> Tuple[float, bool]:
        """Seconds a hold starting at ``start_idx`` may run before the free foot
        runs out of road, and whether a jump is what stopped it.

        While one foot is pinned by the hold, *every* note in its span is hit by
        the other foot, so this needs no knowledge of the eventual footing —
        which is what lets hold derivation run before arrow assignment. The span
        ends at the first note that:

        * is a jump (needs both feet), or
        * follows the previous in-span note by less than ``jack_min_dt`` (one
          foot cannot repeat that fast), or
        * exceeds the difficulty's ``max_notes_under_hold`` budget.
        """
        t0 = float(note_events[start_idx]['time'])
        prev_t = t0
        n_under = 0
        for j in range(start_idx + 1, len(note_events)):
            nxt = note_events[j]
            t = float(nxt['time'])
            if int(nxt.get('num_arrows', 1)) >= 2:
                return t - t0, True
            if n_under >= max_notes_under or t - prev_t < jack_min_dt:
                return t - t0, False
            n_under += 1
            prev_t = t
        return float('inf'), False

    def _quantize_hold(
        self,
        budget_seconds: float,
        beat_interval: float,
        pref_beats: float,
        min_seconds: float,
        rank: float = 0.5,
    ) -> Optional[float]:
        """Choose a hold length in whole beats, or None if the window admits none.

        Quantizing to whole beats is what puts tails on the grid — the old path
        left ``hold_duration`` in raw frames while the head was beat-snapped, so
        every tail landed off-grid and ``sm_export`` had to re-snap it.

        ``rank`` is the note's sustain percentile among the chart's holds; it
        indexes :attr:`HOLD_LENGTH_CDF`, so the more a note rings the longer its
        hold, and the lengths across a chart reproduce the corpus distribution.
        ``pref_beats`` biases that draw for styles that write unusually long or
        short holds. The result is then clamped into
        ``[min_seconds, budget_seconds]``; if no cell survives, the caller drops
        the candidate rather than flooring it to the minimum.
        """
        if beat_interval <= 0:
            return None
        fitting = [
            c for c in self.HOLD_BEAT_CELLS
            if min_seconds <= c * beat_interval <= budget_seconds
        ]
        if not fitting:
            return None

        bias = pref_beats / self.HOLD_CORPUS_TYPICAL_BEATS if pref_beats > 0 else 1.0
        draw = float(np.clip(rank * bias, 0.0, 1.0))
        target = self.HOLD_LENGTH_CDF[-1][0]
        for cell, cumulative in self.HOLD_LENGTH_CDF:
            if draw <= cumulative:
                target = cell
                break
        # Clamp into the playable window, preferring the shorter side on a tie —
        # overshooting pins a foot for longer and suppresses other notes.
        return min(fitting, key=lambda c: (abs(c - target), c)) * beat_interval

    def _postprocess(
        self,
        onset_p: np.ndarray,
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
        arrow_probs: Optional[np.ndarray] = None,
    ) -> List[Step]:
        """
        Detect WHEN (NMS on onset_p) → mark jumps (top-K on intensity) → snap to
        the beat grid → derive holds (audio sustain, foot-feasibility bounded) →
        assign arrows via FootFlowAssigner (whole-timeline foot-flow pass).

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

        # Every peak starts as a tap. Holds are derived further down, after beat
        # snapping, so their tails land on the grid (see _derive_holds).
        note_events = []
        for peak_idx, (frame, t, confidence) in enumerate(peaks):
            note_events.append({
                'frame': frame, 'time': t, 'type': 'tap',
                'confidence': confidence,
                'num_arrows': 2 if peak_idx in jump_peak_indices else 1,
            })

        # Density cap — keep top-N by confidence.
        avg_density = (diff_config.min_density + diff_config.max_density) / 2.0
        target_notes = max(1, int(round(avg_density * duration)))
        logger.info(
            f"[postprocess debug] peaks={len(peaks)} note_events={len(note_events)} "
            f"target_notes={target_notes} (density={avg_density:.2f}/s, dur={duration:.1f}s)"
        )
        if len(note_events) > target_notes:
            note_events.sort(key=lambda e: e['confidence'], reverse=True)
            note_events = note_events[:target_notes]

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

        # Hold derivation. Runs on the finalized, beat-snapped timeline so a
        # hold's length can be measured in whole beats (grid-aligned tails) and
        # its foot-feasibility judged against the notes that actually survived.
        self._derive_holds(
            note_events, harmonic_curve, tempo, beat_times, T,
            diff_config, style_knobs,
        )

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

        # Arrow assignment — one whole-timeline foot-flow pass. The assigner sees
        # the finalized timeline (times, type, arity, subdivision already decided
        # above) so every panel choice has full lookahead to each note's real
        # inter-onset interval, and every step is a natural next step from the
        # current stance (playable by construction; verified by ml.playability).
        assigner = FootFlowAssigner(
            difficulty=diff_config.name,
            seed=audio_seed,
            allowed_patterns=diff_config.allowed_patterns,
            jack_rate=style_knobs.get('jack_rate', DEFAULT_STYLE_KNOBS['jack_rate']),
            crossover_rate=style_knobs.get('crossover_rate', DEFAULT_STYLE_KNOBS['crossover_rate']),
        )
        arrows_per_event = assigner.assign(
            note_events, energy_curve, brightness_curve, beat_times,
            arrow_probs=arrow_probs,
        )

        steps: List[Step] = []
        n_unassignable = 0
        for event, arrows in zip(note_events, arrows_per_event):
            if not arrows:
                n_unassignable += 1
                continue
            emit_t = round(max(0.0, event['time'] + timing_offset), 3)
            subdivision = event.get('subdivision') or self._subdivision_from_grid(
                round(event['time'], 3), tempo, beat_times,
            )
            # `type` may have been demoted hold→tap during assignment, so it is
            # read after assign(), not before.
            if event['type'] == 'hold':
                steps.append(Step(
                    time=emit_t,
                    arrows=arrows,
                    step_type=StepType.HOLD,
                    hold_duration=round(event['hold_duration'], 3),
                    beat_subdivision=subdivision,
                ))
            else:
                steps.append(Step(
                    time=emit_t,
                    arrows=arrows,
                    step_type=StepType.TAP,
                    beat_subdivision=subdivision,
                ))

        n_holds = sum(1 for s in steps if s.step_type == StepType.HOLD)
        logger.info(
            f"[postprocess debug] emitted {len(steps)} steps "
            f"({n_holds} holds, {n_unassignable} unassignable)"
        )
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
