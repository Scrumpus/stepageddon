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
from modules.step_generator.schemas import (
    Chart, Step, StepType, Direction, BeatSubdivision,
)
from modules.step_generator.difficulty import get_difficulty_config

logger = logging.getLogger(__name__)

ARROW_DIRECTIONS = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]

APP_DIFFICULTY_MAP = {
    'beginner': 0,
    'easy': 1,
    'medium': 2,
    'hard': 3,
    'challenge': 4,
}

# Defaults for style-profile post-processing knobs. These are used when a
# profile doesn't explicitly carry the field (e.g. a hand-edited profile, or
# the legacy "auto" path before clustering has run). Conservative values that
# work on all difficulties.
DEFAULT_STYLE_KNOBS: Dict[str, float] = {
    'jump_threshold': 0.5,        # intensity ≥ this → jump
    'jumphold_rate': 0.0,         # 0 disables jumpholds
    'jumphold_rms_thr': 0.7,
    'jack_rate': 0.05,
    'crossover_rate': 0.10,
    'stream_preference': 0.5,
}


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

    STREAM_PATTERNS = [
        [0, 2, 1, 3],  # L U D R — standard weave
        [3, 1, 2, 0],  # R D U L — reverse weave
        [0, 1, 2, 3],  # L D U R — staircase up
        [3, 2, 1, 0],  # R U D L — staircase down
        [0, 3, 1, 2],  # L R D U — wide crossover
        [1, 2, 0, 3],  # D U L R — inside-out
    ]

    JUMP_PATTERNS = [
        (Direction.LEFT, Direction.RIGHT),
        (Direction.DOWN, Direction.UP),
        (Direction.LEFT, Direction.UP),
        (Direction.DOWN, Direction.RIGHT),
    ]

    def __init__(
        self,
        seed: int = 0,
        allowed_patterns: Optional[List[str]] = None,
        jack_rate: float = 0.05,
        crossover_rate: float = 0.10,
        stream_preference: float = 0.5,
    ):
        self.rng = random.Random(seed)
        self.allowed_patterns = set(allowed_patterns or ['single'])
        self.jack_rate = float(np.clip(jack_rate, 0.0, 1.0))
        self.crossover_rate = float(np.clip(crossover_rate, 0.0, 1.0))
        self.stream_preference = float(np.clip(stream_preference, 0.0, 1.0))
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

    def assign_single(self, time: float, energy: float = 0.5,
                       brightness: float = 0.5) -> Direction:
        # Active stream — follow the pattern.
        if self._stream_pattern is not None:
            arrow_idx = self._stream_pattern[self._stream_idx]
            self._stream_idx += 1
            if self._stream_idx >= len(self._stream_pattern):
                self._stream_pattern = None
                self._stream_idx = 0
            arrow = self.ALL_PANELS[arrow_idx]
            foot = self._pick_foot(time)
            self._update_foot(foot, arrow)
            self._step_count += 1
            return arrow

        # Jack branch: with profile-controlled probability, repeat the last
        # arrow rather than alternating feet.
        if (
            self._last_arrow is not None
            and self.jack_rate > 0.0
            and self.rng.random() < self.jack_rate
            and not self._held_feet(time)
        ):
            self._step_count += 1
            arrow = self._last_arrow
            # Don't flip feet for a jack — same foot taps again.
            self._update_foot(self.last_foot, arrow)
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
            if brightness > 0.6:
                weights = [0.7, 0.3]
            elif brightness < 0.4:
                weights = [0.3, 0.7]
            else:
                weights = [0.5, 0.5]

            if current_pos == panels[0]:
                weights[1] += 0.2
            else:
                weights[0] += 0.2

            total = weights[0] + weights[1]
            arrow = panels[0] if self.rng.random() < weights[0] / total else panels[1]

        self._update_foot(foot, arrow)
        return arrow

    def assign_jump(self, time: float, energy: float = 0.5,
                     brightness: float = 0.5) -> List[Direction]:
        if self._held_feet(time):
            return [self.assign_single(time, energy, brightness)]

        if energy > 0.7:
            candidates = [self.JUMP_PATTERNS[0], self.JUMP_PATTERNS[2],
                          self.JUMP_PATTERNS[3]]
        elif energy < 0.3:
            candidates = [self.JUMP_PATTERNS[1]]
        else:
            candidates = list(self.JUMP_PATTERNS)

        pattern = self.rng.choice(candidates)
        self._step_count += 1
        self.left_pos = pattern[0]
        self.right_pos = pattern[1]
        self.last_foot = 'right'
        self._last_arrow = pattern[1]
        return [pattern[0], pattern[1]]

    def start_hold(self, time: float, duration: float,
                    energy: float = 0.5, brightness: float = 0.5) -> Direction:
        self._purge_holds(time)
        arrow = self.assign_single(time, energy, brightness)
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

    # Jump threshold: profiles with a high observed jump_rate need a *lower*
    # intensity threshold, since jumps are common in their style. Map the
    # standardized jump_rate range [0..0.2] linearly to threshold [0.6..0.35].
    jump_rate = float(rm.get('jump_rate', 0.05))
    knobs['jump_threshold'] = float(np.clip(0.60 - 1.5 * jump_rate, 0.30, 0.65))

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
        self.style = str(style)

        self.model: Optional[StepChartModel] = None
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
        if arch_version != ARCH_VERSION:
            raise ValueError(
                f"Checkpoint at {model_path} is arch_version={arch_version}; "
                f"expected {ARCH_VERSION}. Retrain with current model.py."
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

    def generate_from_audio(
        self,
        audio_path: str,
        difficulty: str = 'medium',
        target_density: Optional[float] = None,
        style: Optional[str] = None,
    ) -> Chart:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        difficulty_id = APP_DIFFICULTY_MAP.get(difficulty, 2)

        if target_density is None:
            target_density = float(self.default_density_by_id[difficulty_id])
        logger.info(f"Target density: {target_density:.2f} steps/sec")

        logger.info(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        feats = self._extract_feats(y, sr)
        if self.feat_mean is None or self.feat_std is None:
            raise RuntimeError("Model loaded without feat_mean/feat_std. Retrain.")
        feats_whitened = (feats - self.feat_mean) / self.feat_std

        # Tempo / beats for snapping.
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # RMS / spectral centroid for the FootStateArrowAssigner heuristics
        # and for the jumphold gate.
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        rms_norm = rms / (rms.max() + 1e-8)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=HOP_LENGTH,
        )[0]
        centroid_norm = centroid / (centroid.max() + 1e-8)

        # HPSS harmonic energy curve — used to detect hold sustain decay.
        try:
            y_harm = librosa.effects.hpss(y)[0]
            harm_rms = librosa.feature.rms(y=y_harm, hop_length=HOP_LENGTH)[0]
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

        logger.info(f"Running inference (difficulty={difficulty}, id={difficulty_id})...")
        onset_p, sustain_p, intensity = self._predict_chunked(
            feats_whitened, difficulty_id, target_density,
        )

        diff_config = get_difficulty_config(difficulty)
        logger.info("Post-processing predictions...")

        # Resolve style profile based on the predicted dense signals.
        T = onset_p.shape[0]
        onset_density_pred = float(onset_p.sum() / max(T / FRAMES_PER_SECOND, 1.0))
        intensity_mean_pred = float(intensity[intensity > 0.0].mean()) if (intensity > 0.0).any() else 0.0
        resolved_style = style if style is not None else self.style
        knobs, resolved_name = self._resolve_style(
            resolved_style, onset_density_pred, intensity_mean_pred,
        )

        steps = self._postprocess(
            onset_p, sustain_p, intensity,
            tempo, beat_times, duration, diff_config,
            energy_curve=rms_norm, brightness_curve=centroid_norm,
            harmonic_curve=harm_rms,
            audio_seed=audio_seed,
            style_knobs=knobs,
        )

        chart = Chart(
            steps=steps,
            difficulty=difficulty,
            tempo=tempo,
            duration=duration,
        )

        logger.info(
            f"Generated {len(steps)} steps "
            f"({len(chart.get_taps())} taps, {len(chart.get_holds())} holds) "
            f"style='{resolved_name}'"
        )
        return chart

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

            onset_logits, sustain_logits, intensity_pred = self.model(
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
    ) -> List[Step]:
        """
        Detect WHEN (NMS on onset_p) → classify type (jump/hold/tap) →
        assign arrows via FootStateArrowAssigner driven by style knobs.
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
        jumphold_rate = float(style_knobs.get('jumphold_rate', 0.0))
        jumphold_rms_thr = float(style_knobs.get('jumphold_rms_thr', 0.7))

        max_hold_frames = max(1, int(round(diff_config.max_hold_duration * FRAMES_PER_SECOND)))
        rng = random.Random(audio_seed ^ 0xA5A5A5A5)

        note_events = []
        for frame, t, confidence in peaks:
            is_hold_start = (
                holds_allowed
                and sustain_p[frame] >= self.sustain_threshold_up
            )
            intensity_val = float(intensity[frame])
            is_jump = jumps_allowed and (intensity_val >= jump_threshold)

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

        # Beat-snap: anchor each event to the nearest librosa beat and try
        # both binary (16th) and ternary (12th) grids local to that beat.
        # Pick whichever fits closer. A small bias against the triplet grid
        # keeps incidental near-12th positions from getting mislabeled in
        # otherwise-binary songs.
        if self.snap_to_beats and len(beat_times) > 1:
            bts = np.asarray(beat_times, dtype=np.float64)
            intervals = np.diff(bts)
            local_interval = np.empty_like(bts)
            local_interval[0] = intervals[0]
            local_interval[-1] = intervals[-1]
            if intervals.size > 1:
                local_interval[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])

            # Slot → subdivision lookup per grid divisor.
            sub_by_slot_4 = (
                BeatSubdivision.QUARTER, BeatSubdivision.SIXTEENTH,
                BeatSubdivision.EIGHTH, BeatSubdivision.SIXTEENTH,
            )
            sub_by_slot_3 = (
                BeatSubdivision.QUARTER, BeatSubdivision.TWELFTH,
                BeatSubdivision.TWELFTH,
            )

            for event in note_events:
                t = float(event['time'])
                idx = int(np.searchsorted(bts, t))
                candidates = []
                if idx > 0:
                    candidates.append(idx - 1)
                if idx < bts.size:
                    candidates.append(idx)

                best_snapped = t
                best_sub: Optional[BeatSubdivision] = None
                best_dist = float('inf')
                best_step_16 = float(local_interval[candidates[0]]) / 4.0
                for c in candidates:
                    beat_t = float(bts[c])
                    interval = float(local_interval[c])
                    step16 = interval / 4.0
                    step12 = interval / 3.0

                    slot16 = int(round((t - beat_t) / step16))
                    snap16 = beat_t + slot16 * step16
                    dist16 = abs(t - snap16)
                    sub16 = sub_by_slot_4[slot16 % 4]

                    slot12 = int(round((t - beat_t) / step12))
                    snap12 = beat_t + slot12 * step12
                    dist12 = abs(t - snap12)
                    sub12 = sub_by_slot_3[slot12 % 3]

                    # Prefer 16th unless 12th is meaningfully closer.
                    triplet_bias = 0.20 * step16
                    if dist16 <= dist12 + triplet_bias:
                        snapped, sub, dist, step_ref = snap16, sub16, dist16, step16
                    else:
                        snapped, sub, dist, step_ref = snap12, sub12, dist12, step16

                    if dist < best_dist:
                        best_dist = dist
                        best_snapped = snapped
                        best_sub = sub
                        best_step_16 = step_ref

                # Cap snap radius at half a 16th of the local interval; beyond
                # that, leave the event unsnapped — the subdivision fallback
                # will guess from its raw position.
                if best_dist <= 0.5 * best_step_16 and best_sub is not None:
                    event['time'] = float(best_snapped)
                    event['subdivision'] = best_sub

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

        # Arrow assignment.
        assigner = FootStateArrowAssigner(
            seed=audio_seed,
            allowed_patterns=diff_config.allowed_patterns,
            jack_rate=style_knobs.get('jack_rate', DEFAULT_STYLE_KNOBS['jack_rate']),
            crossover_rate=style_knobs.get('crossover_rate', DEFAULT_STYLE_KNOBS['crossover_rate']),
            stream_preference=style_knobs.get('stream_preference', DEFAULT_STYLE_KNOBS['stream_preference']),
        )
        steps: List[Step] = []
        max_stream_length = max(1, getattr(diff_config, 'max_stream_length', 0))

        for i, event in enumerate(note_events):
            t = round(event['time'], 3)
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

            if max_stream_length > 0:
                assigner.maybe_start_stream(
                    energy=energy,
                    remaining_events=len(note_events) - i,
                    max_stream_length=max_stream_length,
                    time=t,
                )

            target_arity = 2 if event.get('num_arrows', 1) >= 2 else 1

            if event['type'] == 'hold':
                if target_arity >= 2:
                    # Jumphold: assign two arrows, register both as active holds.
                    arrows = assigner.assign_jump(t, energy, brightness)
                    if arrows:
                        end_time = t + event['hold_duration']
                        for a in arrows:
                            assigner.active_holds[a] = end_time
                else:
                    arrows = [assigner.start_hold(
                        t, event['hold_duration'], energy, brightness,
                    )]
                steps.append(Step(
                    time=t,
                    arrows=arrows,
                    step_type=StepType.HOLD,
                    hold_duration=round(event['hold_duration'], 3),
                    beat_subdivision=subdivision,
                ))
            else:
                if target_arity >= 2:
                    arrows = assigner.assign_jump(t, energy, brightness)
                else:
                    arrows = [assigner.assign_single(t, energy, brightness)]
                steps.append(Step(
                    time=t,
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

        beat_interval = 60.0 / tempo
        anchor = float(beat_times[0])
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
