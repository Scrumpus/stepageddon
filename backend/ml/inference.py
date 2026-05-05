"""
Inference module: generate step charts from audio using trained model.

Handles chunked processing for arbitrary-length songs, post-processing
(beat-snapping, hold cleanup, minimum gap), and conversion to the
existing Chart/Step schema.
"""

import hashlib
import logging
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import librosa

from ml.model import StepChartModel
from ml.dataset import DEFAULT_DENSITY_BY_ID, DENSITY_MEAN, DENSITY_STD
from ml.prepare_data import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, FRAMES_PER_SECOND,
    DB_MIN, DB_MAX, DB_RANGE,
)
from modules.step_generator.schemas import (
    Chart, Step, StepType, Direction, BeatSubdivision,
)
from modules.step_generator.difficulty import get_difficulty_config

logger = logging.getLogger(__name__)

ARROW_DIRECTIONS = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]

DEFAULT_TYPE_PRIOR_FALLBACK = np.array([0.88, 0.08, 0.04], dtype=np.float32)

APP_DIFFICULTY_MAP = {
    'beginner': 0,
    'easy': 1,
    'medium': 2,
    'hard': 3,
    'challenge': 4,
}
class FootStateArrowAssigner:
    """
    Assigns arrows to note events using foot-state tracking, a seeded PRNG,
    expanded pattern vocabulary, and audio-feature-driven selection.
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
        (Direction.LEFT, Direction.RIGHT),   # wide
        (Direction.DOWN, Direction.UP),       # center
        (Direction.LEFT, Direction.UP),       # left-leaning
        (Direction.DOWN, Direction.RIGHT),    # right-leaning
    ]

    CANDLE_PATTERNS = [
        (Direction.LEFT,  [Direction.DOWN, Direction.UP, Direction.RIGHT]),
        (Direction.RIGHT, [Direction.UP, Direction.DOWN, Direction.LEFT]),
        (Direction.DOWN,  [Direction.LEFT, Direction.UP, Direction.RIGHT]),
        (Direction.UP,    [Direction.RIGHT, Direction.DOWN, Direction.LEFT]),
    ]

    def __init__(self, seed: int = 0, allowed_patterns: Optional[List[str]] = None):
        self.rng = random.Random(seed)
        self.allowed_patterns = set(allowed_patterns or ['single'])
        self.reset()

    def reset(self):
        self.last_foot = 'right'  # so first step uses left foot
        self.left_pos = Direction.LEFT
        self.right_pos = Direction.RIGHT
        self.held_foot: Optional[str] = None
        self.hold_end_time: float = 0.0
        self._step_count = 0
        # Stream state: when active, we follow a stream pattern
        self._stream_pattern: Optional[List[int]] = None
        self._stream_idx: int = 0

    def _pick_foot(self, time: float) -> str:
        """Pick which foot moves next, respecting holds."""
        if self.held_foot and time >= self.hold_end_time:
            self.held_foot = None
        if self.held_foot == 'left':
            return 'right'
        elif self.held_foot == 'right':
            return 'left'
        elif self.last_foot == 'left':
            return 'right'
        else:
            return 'left'

    def _panels_for_foot(self, foot: str) -> List[Direction]:
        return self.LEFT_FOOT_PANELS if foot == 'left' else self.RIGHT_FOOT_PANELS

    def _update_foot(self, foot: str, arrow: Direction):
        if foot == 'left':
            self.left_pos = arrow
        else:
            self.right_pos = arrow
        self.last_foot = foot

    def assign_single(self, time: float, energy: float = 0.5,
                       brightness: float = 0.5) -> Direction:
        """
        Assign one arrow for a single tap or hold_start.

        Audio features influence panel selection:
        - energy: 0-1, higher = more likely to pick movement-heavy panels
        - brightness: 0-1, higher = favor outer panels (L/R), lower = inner (D/U)
        """
        # If we're in an active stream, follow the pattern
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

        foot = self._pick_foot(time)
        panels = self._panels_for_foot(foot)
        current_pos = self.left_pos if foot == 'left' else self.right_pos

        self._step_count += 1

        # Crossover chance: high energy can push foot to non-natural panel
        if ('crossover' in self.allowed_patterns
                and energy > 0.7 and self.rng.random() < (energy - 0.5) * 0.6):
            # Pick from the OTHER foot's panels (crossover)
            cross_panels = self._panels_for_foot(
                'right' if foot == 'left' else 'left'
            )
            arrow = self.rng.choice(cross_panels)
        else:
            # Normal panel selection, weighted by brightness
            # High brightness → favor outer panel (index 0 = L or U)
            # Low brightness → favor inner panel (index 1 = D or R)
            if brightness > 0.6:
                weights = [0.7, 0.3]
            elif brightness < 0.4:
                weights = [0.3, 0.7]
            else:
                weights = [0.5, 0.5]

            # Bias toward switching panels (avoid staying on same one)
            if current_pos == panels[0]:
                weights[1] += 0.2
            else:
                weights[0] += 0.2

            # Weighted random choice
            total = weights[0] + weights[1]
            arrow = panels[0] if self.rng.random() < weights[0] / total else panels[1]

        self._update_foot(foot, arrow)
        return arrow

    def assign_jump(self, time: float, energy: float = 0.5,
                     brightness: float = 0.5) -> List[Direction]:
        """Assign two arrows for a jump (one per foot)."""
        if self.held_foot and time >= self.hold_end_time:
            self.held_foot = None

        if self.held_foot:
            return [self.assign_single(time, energy, brightness)]

        # High energy → favor wide jumps, low energy → center jumps
        if energy > 0.7:
            candidates = [self.JUMP_PATTERNS[0], self.JUMP_PATTERNS[2],
                          self.JUMP_PATTERNS[3]]  # wide patterns
        elif energy < 0.3:
            candidates = [self.JUMP_PATTERNS[1]]  # center only
        else:
            candidates = list(self.JUMP_PATTERNS)

        pattern = self.rng.choice(candidates)
        self._step_count += 1
        self.left_pos = pattern[0]
        self.right_pos = pattern[1]
        self.last_foot = 'right'
        return [pattern[0], pattern[1]]

    def start_hold(self, time: float, duration: float,
                    energy: float = 0.5, brightness: float = 0.5) -> Direction:
        """Assign arrow for a hold start and lock that foot."""
        arrow = self.assign_single(time, energy, brightness)
        self.held_foot = self.last_foot
        self.hold_end_time = time + duration
        return arrow

    def maybe_start_stream(self, energy: float, remaining_events: int,
                            max_stream_length: int) -> bool:
        """
        Decide whether to enter a stream pattern based on energy.
        Returns True if a stream was started.
        """
        if 'stream' not in self.allowed_patterns or max_stream_length == 0:
            return False
        if self._stream_pattern is not None:
            return False  # already in a stream
        if self.held_foot is not None:
            return False  # can't stream during a hold

        # Higher energy → higher chance of entering a stream
        stream_chance = max(0.0, (energy - 0.5) * 0.8)
        if self.rng.random() >= stream_chance:
            return False

        # Pick pattern and length based on energy
        pattern = list(self.rng.choice(self.STREAM_PATTERNS))
        max_len = min(max_stream_length, remaining_events, len(pattern) * 3)
        stream_len = self.rng.randint(len(pattern), max(len(pattern), max_len))

        # Tile the pattern to fill the stream length
        full_pattern = []
        while len(full_pattern) < stream_len:
            full_pattern.extend(pattern)
        self._stream_pattern = full_pattern[:stream_len]
        self._stream_idx = 0
        return True


class MLChartGenerator:
    """Generate step charts using the trained neural network."""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        chunk_frames: int = 500,
        overlap_frames: int = 100,
        confidence_threshold: Optional[float] = None,
        min_note_gap: float = 0.05,
        snap_to_beats: bool = True,
        type_logit_adjust: float = 1.0,
        jump_bias: float = 0.0,
        hold_bias: float = 0.0,
        type_prior_override: Optional[List[float]] = None,
        min_first_step_time: float = 0.25,
        min_last_step_buffer: float = 0.0,
    ):
        """
        Args (the new ones for type rebalancing):
            type_logit_adjust: Strength of prior-based logit adjustment on
                the type head, applied as `logits -= type_logit_adjust * log(prior)`.
                0.0 disables it (raw model output, very tap-biased).
                1.0 fully removes the empirical prior, giving a uniform
                posterior over {tap, jump, hold_start}.
                In practice 0.6–1.0 produces noticeably more jumps/holds
                without breaking tap-dominated sections. Requires the
                checkpoint to carry `type_prior`; ignored otherwise.
            jump_bias: Additive logit bias applied to the jump class only.
                Positive values encourage more jumps. Useful at inference
                time to dial the jump rate per song without retraining.
            hold_bias: Same as jump_bias for the hold_start class.
            type_prior_override: Optional 3-vector P(class|onset) over
                {tap, jump, hold_start} used for logit adjustment. Takes
                precedence over the checkpoint's stored prior. When neither
                is provided, falls back to DEFAULT_TYPE_PRIOR_FALLBACK so
                the rebalancing still works on legacy checkpoints.
            min_first_step_time: Hard guardrail (seconds) — peaks before
                this timestamp are dropped. Belt-and-suspenders alongside
                the model's learned position conditioning so the very
                first step never lands at t≈0 even if the model fires.
            min_last_step_buffer: Hard guardrail (seconds) — peaks within
                this many seconds of the song's end are dropped. 0.0
                disables (the default; opt-in for callers who want it).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        # confidence_threshold may be None at construction; load_model will
        # populate it from the checkpoint's saved best_threshold (calibrated
        # by the validation F1 sweep). Falls back to 0.3 if neither the
        # constructor nor the checkpoint provides a value.
        self._confidence_threshold_override = confidence_threshold
        self.confidence_threshold = (
            float(confidence_threshold) if confidence_threshold is not None else 0.3
        )
        self.min_note_gap = min_note_gap
        self.snap_to_beats = snap_to_beats
        self.min_first_step_time = float(min_first_step_time)
        self.min_last_step_buffer = float(min_last_step_buffer)
        self.type_logit_adjust = float(type_logit_adjust)
        # Track whether the caller pinned the biases explicitly. Only
        # caller-provided values take precedence over the per-class
        # calibration written into the checkpoint by training.
        self._jump_bias_override = jump_bias != 0.0
        self._hold_bias_override = hold_bias != 0.0
        self.jump_bias = float(jump_bias)
        self.hold_bias = float(hold_bias)

        # Empirical P(class|onset) over {tap, jump, hold_start} on the
        # training corpus. Resolution order at load time:
        #   1. type_prior_override constructor arg (caller-supplied)
        #   2. checkpoint['type_prior'] (saved by retrained models)
        #   3. DEFAULT_TYPE_PRIOR_FALLBACK (heuristic; covers legacy ckpts)
        self.type_prior: Optional[np.ndarray] = None
        self._type_prior_override = (
            np.asarray(type_prior_override, dtype=np.float32)
            if type_prior_override is not None else None
        )

        self.model = None
        # Per-arrow decision thresholds for the arrow head. Populated from the
        # checkpoint if it was saved with one (arch_version=2). When None,
        # `_choose_arrows` falls back to the scalar `confidence_threshold`.
        self.arrow_thresholds: Optional[np.ndarray] = None
        # Per-difficulty inference-time default density (steps/sec). Populated
        # from the checkpoint if it was saved with one, else falls back to the
        # hardcoded preset midpoints.
        self.default_density_by_id = DEFAULT_DENSITY_BY_ID.clone()
        # Unit of the duration head's output. 'beats' for new checkpoints
        # (BPM-factored target); 'seconds' for legacy checkpoints. Affects
        # the post-decode conversion in generate_from_audio.
        self.duration_unit: str = 'seconds'
        # Mel whitening stats. Populated from the checkpoint if present (new
        # training runs); when absent, mel_mean/mel_std stay None and we fall
        # back to the legacy per-file min-max normalization so legacy
        # checkpoints continue to work without retraining.
        self.mel_mean: Optional[float] = None
        self.mel_std: Optional[float] = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model hyperparameters from checkpoint
        args = checkpoint.get('args', {})
        self.model = StepChartModel(
            n_mels=80,
            hidden_dim=args.get('hidden_dim', 256),
            n_heads=args.get('n_heads', 8),
            n_transformer_layers=args.get('n_layers', 4),
            n_difficulties=5,
        ).to(self.device)

        # Prefer EMA weights if present (the current training script saves them
        # alongside raw weights and selects best by val metrics on EMA).
        state_dict = checkpoint.get('ema_state_dict') or checkpoint['model_state_dict']
        if checkpoint.get('ema_state_dict') is not None:
            logger.info("Loading EMA weights from checkpoint.")
        # strict=False: tolerate minor schema drift between training runs.
        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            logger.warning(f"Checkpoint missing keys (zero-initialized): {missing}")
        if unexpected:
            logger.warning(f"Checkpoint had unexpected keys (ignored): {unexpected}")
        self.model.eval()

        # Pull data-driven default densities if the checkpoint stored them.
        ckpt_default = checkpoint.get('default_density_by_id')
        if ckpt_default is not None:
            self.default_density_by_id = torch.as_tensor(
                ckpt_default, dtype=torch.float32
            )
            logger.info(
                f"Using checkpoint default densities: "
                f"{self.default_density_by_id.tolist()}"
            )

        # Resolve the type prior used for logit adjustment.
        # Override > checkpoint > heuristic fallback.
        if self._type_prior_override is not None:
            tp = self._type_prior_override.astype(np.float64)
            source = "override"
        elif checkpoint.get('type_prior') is not None:
            tp = np.asarray(checkpoint['type_prior'], dtype=np.float64)
            source = "checkpoint"
        else:
            tp = DEFAULT_TYPE_PRIOR_FALLBACK.astype(np.float64)
            source = "fallback (legacy checkpoint, retrain to override)"
        tp = tp / max(tp.sum(), 1e-9)
        self.type_prior = tp.astype(np.float32)
        logger.info(
            f"Type prior [{source}]: tap={tp[0]:.3f} "
            f"jump={tp[1]:.3f} hold_start={tp[2]:.3f}"
        )

        # Pull per-class calibration biases from the checkpoint (saved by
        # training's post-hoc bias sweep that maximizes macro-F1 on val).
        # Caller-supplied biases override these — that's what the explicit-
        # override flags are for.
        ckpt_jump = checkpoint.get('best_jump_bias')
        ckpt_hold = checkpoint.get('best_hold_bias')
        if not self._jump_bias_override and ckpt_jump is not None:
            self.jump_bias = float(ckpt_jump)
            logger.info(f"Jump bias = {self.jump_bias:+.3f} (from checkpoint)")
        if not self._hold_bias_override and ckpt_hold is not None:
            self.hold_bias = float(ckpt_hold)
            logger.info(f"Hold bias = {self.hold_bias:+.3f} (from checkpoint)")

        # Pull the calibrated decision threshold if the checkpoint stored
        # one. Caller-supplied confidence_threshold always wins.
        ckpt_thr = checkpoint.get('best_threshold')
        if self._confidence_threshold_override is not None:
            logger.info(
                f"Onset threshold = {self.confidence_threshold:.3f} (caller override)"
            )
        elif ckpt_thr is not None:
            self.confidence_threshold = float(ckpt_thr)
            logger.info(
                f"Onset threshold = {self.confidence_threshold:.3f} "
                f"(from checkpoint best_threshold)"
            )
        else:
            logger.warning(
                f"Checkpoint has no best_threshold; using default "
                f"{self.confidence_threshold:.3f}. Retrain to calibrate."
            )

        # Per-arrow thresholds. arch_version=2 checkpoints save these from the
        # validation per-arrow F1 sweep. Caller-supplied confidence_threshold
        # disables the per-arrow path (uniform threshold across arrows).
        ckpt_arrow_thr = checkpoint.get('best_arrow_thresholds')
        if (
            ckpt_arrow_thr is not None
            and self._confidence_threshold_override is None
        ):
            self.arrow_thresholds = np.asarray(ckpt_arrow_thr, dtype=np.float32)
            logger.info(
                f"Per-arrow thresholds [L,D,U,R] = {self.arrow_thresholds.tolist()} "
                f"(from checkpoint)"
            )
        else:
            logger.info(
                "Per-arrow thresholds: using uniform confidence_threshold "
                f"({self.confidence_threshold:.3f}) for all 4 arrows."
            )

        # Duration unit marker. New checkpoints train on beats; legacy on
        # seconds. Inference converts beats → seconds using the song tempo.
        self.duration_unit = str(checkpoint.get('duration_unit', 'seconds'))
        if self.duration_unit not in ('beats', 'seconds'):
            logger.warning(
                f"Unknown duration_unit={self.duration_unit!r}; assuming 'seconds'."
            )
            self.duration_unit = 'seconds'
        logger.info(f"Duration head unit: {self.duration_unit}")

        # Pull mel whitening stats if present. Their presence is also the
        # signal that this checkpoint was trained on the v2 fixed-dB pipeline,
        # so the audio path in generate_from_audio switches accordingly.
        if 'mel_mean' in checkpoint and 'mel_std' in checkpoint:
            self.mel_mean = float(checkpoint['mel_mean'])
            self.mel_std = float(checkpoint['mel_std'])
            logger.info(
                f"Using checkpoint mel whitening stats: "
                f"mean={self.mel_mean:.4f}, std={self.mel_std:.4f}"
            )
        else:
            logger.warning(
                "Checkpoint has no mel_mean/mel_std; falling back to legacy "
                "per-file min-max normalization. Retrain with the updated "
                "prepare_data.py + train.py to get cross-song-consistent input."
            )

        logger.info(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    def generate_from_audio(
        self,
        audio_path: str,
        difficulty: str = 'medium',
        target_density: Optional[float] = None,
    ) -> Chart:
        """
        Generate a step chart from an audio file.

        Args:
            audio_path: Path to audio file
            difficulty: 'beginner', 'easy', 'medium', 'hard', or 'challenge'
            target_density: Desired chart density in steps/sec. If None, uses
                the per-difficulty default (from the checkpoint if available,
                else the hardcoded preset midpoint). Callers can override this
                per-song — e.g. scale by tempo so a 180 BPM song gets a denser
                chart than a 90 BPM song at the same nominal difficulty.

        Returns:
            Chart object with generated steps
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Map app difficulty to model difficulty_id
        difficulty_id = APP_DIFFICULTY_MAP.get(difficulty, 2)

        # Resolve density: explicit override → checkpoint default → preset fallback
        if target_density is None:
            target_density = float(self.default_density_by_id[difficulty_id])
        logger.info(f"Target density: {target_density:.2f} steps/sec")

        # Load and process audio
        logger.info(f"Loading audio from {audio_path}...")
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Extract mel spectrogram. Two pipelines:
        #   - v2 (checkpoint has mel_mean/mel_std): fixed dB scale → [0,1] →
        #     whiten with stored stats. Matches prepare_data.py + dataset.py.
        #   - legacy: per-file min-max, kept so old checkpoints still work.
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
        )
        if self.mel_mean is not None and self.mel_std is not None:
            mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
            np.clip(mel_db, DB_MIN, DB_MAX, out=mel_db)
            mel_scaled = (mel_db - DB_MIN) / DB_RANGE  # [0, 1]
            mel_scaled = (mel_scaled - self.mel_mean) / self.mel_std
            mel_frames = mel_scaled.T.astype(np.float32)
        else:
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
            mel_frames = mel_db.T.astype(np.float32)  # [T, N_MELS]

        # Detect tempo and beats for post-processing
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Extract per-frame audio features for arrow assignment
        # RMS energy (same hop as mel so frames align)
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        rms_norm = rms / (rms.max() + 1e-8)  # normalize to [0, 1]

        # Spectral centroid → brightness proxy, normalized to [0, 1]
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=HOP_LENGTH,
        )[0]
        centroid_norm = centroid / (centroid.max() + 1e-8)

        # Derive a deterministic seed from the audio content
        audio_seed = int(
            hashlib.sha256(mel_frames[:min(4096, len(mel_frames))].tobytes()
                           ).hexdigest()[:8],
            16,
        )

        # Run model inference in chunks
        logger.info(f"Running inference (difficulty={difficulty}, id={difficulty_id})...")
        features, arrow_probs0, type_probs, duration_pred = self._predict_chunked(
            mel_frames, difficulty_id, target_density
        )

        # Convert beats → seconds for new (BPM-factored) checkpoints. Legacy
        # checkpoints already emit seconds, so this is a no-op for them.
        if self.duration_unit == 'beats':
            duration_pred = duration_pred * (60.0 / max(tempo, 1.0))

        # Post-process into steps using difficulty preset
        diff_config = get_difficulty_config(difficulty)
        logger.info("Post-processing predictions...")
        steps = self._postprocess(
            features, arrow_probs0, type_probs, duration_pred,
            tempo, beat_times, duration, diff_config,
            energy_curve=rms_norm, brightness_curve=centroid_norm,
            audio_seed=audio_seed,
        )

        chart = Chart(
            steps=steps,
            difficulty=difficulty,
            tempo=tempo,
            duration=duration,
        )

        logger.info(f"Generated {len(steps)} steps "
                     f"({len(chart.get_taps())} taps, {len(chart.get_holds())} holds)")

        return chart

    @torch.no_grad()
    def _predict_chunked(
        self,
        mel_frames: np.ndarray,
        difficulty_id: int,
        target_density: float,
    ):
        """
        Run model on overlapping chunks and merge predictions.

        Returns:
            features:      [T, H] float32 backbone features (kept for the
                arrow head's autoregressive second pass).
            arrow_probs0:  [T, 4] float32 per-arrow sigmoid under prev_arrow=0.
                Used for NMS peak picking (which frames are onsets).
            type_probs:    [T, 3] float32 softmax over {tap, jump, hold_start}
            duration_pred: [T] float32 predicted hold duration in seconds
                (decoded from log-space via StepChartModel.decode_duration)
        """
        T = mel_frames.shape[0]
        H = self.model.hidden_dim
        feat_sum = np.zeros((T, H), dtype=np.float32)
        arrow_sum = np.zeros((T, 4), dtype=np.float32)
        type_sum = np.zeros((T, 3), dtype=np.float32)
        dur_sum = np.zeros(T, dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)

        stride = self.chunk_frames - self.overlap_frames
        density_norm = (target_density - DENSITY_MEAN) / DENSITY_STD

        # Build per-class additive logit bias for the type head. Same logic as
        # before — prior correction + explicit jump/hold bias in logit space.
        type_logit_bias = np.zeros(3, dtype=np.float32)
        if self.type_prior is not None and self.type_logit_adjust != 0.0:
            log_prior = np.log(np.clip(self.type_prior, 1e-6, 1.0))
            type_logit_bias -= self.type_logit_adjust * log_prior
        type_logit_bias[1] += self.jump_bias
        type_logit_bias[2] += self.hold_bias
        if np.any(type_logit_bias != 0.0):
            logger.info(
                f"Type logit bias: tap={type_logit_bias[0]:+.3f} "
                f"jump={type_logit_bias[1]:+.3f} "
                f"hold_start={type_logit_bias[2]:+.3f}"
            )

        for start in range(0, T, stride):
            end = min(start + self.chunk_frames, T)
            chunk = mel_frames[start:end]

            if chunk.shape[0] < self.chunk_frames:
                pad = np.zeros((self.chunk_frames - chunk.shape[0], N_MELS), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            mel_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            diff_tensor = torch.tensor([difficulty_id], dtype=torch.long, device=self.device)
            density_tensor = torch.tensor(
                [density_norm], dtype=torch.float32, device=self.device
            )
            # Per-chunk song-position scalars matching what the dataset
            # produced at training time (start_seconds and remaining_seconds).
            start_seconds = float(start) / FRAMES_PER_SECOND
            remaining_seconds = max(0.0, float(T - end) / FRAMES_PER_SECOND)
            start_seconds_tensor = torch.tensor(
                [start_seconds], dtype=torch.float32, device=self.device,
            )
            remaining_seconds_tensor = torch.tensor(
                [remaining_seconds], dtype=torch.float32, device=self.device,
            )

            features = self.model.encode(
                mel_tensor, diff_tensor, density_tensor,
                start_seconds_tensor, remaining_seconds_tensor,
            )
            arrow_logits = self.model.apply_arrow_head(features, prev_arrow=None)
            type_logits = self.model.apply_type_head(features)
            dur_pred = self.model.apply_duration_head(features)

            features_np = features.float().cpu().numpy()[0]                      # [T_chunk, H]
            arrow_p = torch.sigmoid(arrow_logits.float()).cpu().numpy()[0]       # [T_chunk, 4]
            type_logits_adj = type_logits.float().cpu().numpy()[0] + type_logit_bias[None, :]
            type_logits_adj -= type_logits_adj.max(axis=-1, keepdims=True)
            exp = np.exp(type_logits_adj)
            type_p = exp / exp.sum(axis=-1, keepdims=True)                       # [T_chunk, 3]
            dur_p = dur_pred.float().cpu().numpy()[0, :, 0]                      # [T_chunk]

            valid_len = min(end - start, self.chunk_frames)
            feat_sum[start:start + valid_len] += features_np[:valid_len]
            arrow_sum[start:start + valid_len] += arrow_p[:valid_len]
            type_sum[start:start + valid_len] += type_p[:valid_len]
            dur_sum[start:start + valid_len] += dur_p[:valid_len]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)
        features_avg = feat_sum / counts[:, None]
        arrow_probs0 = arrow_sum / counts[:, None]
        type_probs = type_sum / counts[:, None]
        duration_log_mean = dur_sum / counts
        duration_pred = StepChartModel.decode_duration(duration_log_mean)
        np.clip(duration_pred, 0.0, None, out=duration_pred)
        return features_avg, arrow_probs0, type_probs, duration_pred

    @torch.no_grad()
    def _arrow_logits_at_frame(
        self, features_t: np.ndarray, prev_arrow_vec: np.ndarray,
    ) -> np.ndarray:
        """Run the arrow head at a single frame with explicit prev_arrow.

        features_t: [H] backbone features at this frame.
        prev_arrow_vec: [4] binary vector of "previous arrow set."
        Returns [4] arrow logits.
        """
        feat_t = torch.from_numpy(features_t).to(self.device).view(1, 1, -1)
        prev = torch.from_numpy(prev_arrow_vec.astype(np.float32)).to(
            self.device
        ).view(1, 1, -1)
        logits = self.model.apply_arrow_head(feat_t, prev_arrow=prev)
        return logits.float().cpu().numpy()[0, 0]

    def _postprocess(
        self,
        features: np.ndarray,        # [T, H] backbone features
        arrow_probs0: np.ndarray,    # [T, 4] per-arrow probs under prev_arrow=0
        type_probs: np.ndarray,      # [T, 3]  (0=tap, 1=jump, 2=hold_start)
        duration_pred: np.ndarray,   # [T] predicted hold duration in seconds
        tempo: float,
        beat_times: np.ndarray,
        duration: float,
        diff_config,
        energy_curve: Optional[np.ndarray] = None,
        brightness_curve: Optional[np.ndarray] = None,
        audio_seed: int = 0,
    ) -> List[Step]:
        """
        Convert (per-arrow onset, type, duration) predictions into Step objects.

        Two-phase pipeline:
            Phase 1: Detect WHEN, WHAT type, HOW LONG (arrow-aware NMS)
                1. NMS peak picking on `arrow_probs0.max(-1)` to find onsets
                   (using the no-context arrow probs — autoregressive
                   re-scoring happens per-onset in phase 2)
                2. Classify each peak via type head: tap / jump / hold_start
                3. For hold_start, read duration from duration head
                4. Density cap, beat-snap, min-gap enforcement

            Phase 2: Determine WHICH arrows (autoregressive arrow head)
                5. Walk onsets in time order, conditioning the arrow head on
                   the previously emitted arrow vector. Per-arrow thresholds.
                6. FootStateArrowAssigner only as a fallback when the arrow
                   head fires on no arrows / wrong cardinality for the type.
        """
        T = arrow_probs0.shape[0]
        any_probs = arrow_probs0.max(axis=-1)

        logger.info(
            f"[postprocess debug] arrow_probs.shape={arrow_probs0.shape} "
            f"any_max={any_probs.max():.4f} any_mean={any_probs.mean():.4f} "
            f"p95={np.percentile(any_probs, 95):.4f} "
            f"p99={np.percentile(any_probs, 99):.4f}"
        )

        min_gap = max(diff_config.min_gap, self.min_note_gap)
        nms_window_frames = max(1, int(round(min_gap * FRAMES_PER_SECOND)))
        jumps_allowed = 'jump' in diff_config.allowed_patterns

        # ================================================================
        # Phase 1: Detect WHEN / WHAT type / HOW LONG
        # ================================================================

        # NMS peak picking on the per-frame "any-arrow" probability.
        # Edge guardrails: drop peaks before `min_first_step_time` and within
        # `min_last_step_buffer` seconds of the end. Backstop alongside the
        # model's learned position conditioning so even a rogue early-frame
        # spike can't produce an arrow at t≈0.
        first_allowed = max(0.0, self.min_first_step_time)
        last_allowed = duration - max(0.0, self.min_last_step_buffer)
        peaks = []
        for frame in range(T):
            p = float(any_probs[frame])
            if p < max(self.confidence_threshold, 1e-4):
                continue
            lo = max(0, frame - nms_window_frames)
            hi = min(T, frame + nms_window_frames + 1)
            if p < any_probs[lo:hi].max():
                continue
            t = frame / FRAMES_PER_SECOND
            if t < first_allowed or t > last_allowed:
                continue
            peaks.append((frame, t, p))

        # Classify each peak and read duration for holds.
        note_events = []
        for frame, t, confidence in peaks:
            note_type = int(np.argmax(type_probs[frame]))
            if note_type == 0:
                note_events.append({
                    'frame': frame, 'time': t, 'type': 'tap',
                    'confidence': confidence, 'num_arrows': 1,
                })
            elif note_type == 1:
                num_arrows = 2 if jumps_allowed else 1
                note_events.append({
                    'frame': frame, 'time': t, 'type': 'tap',
                    'confidence': confidence, 'num_arrows': num_arrows,
                })
            else:
                hold_dur = float(np.clip(duration_pred[frame], 0.2, 5.0))
                note_events.append({
                    'frame': frame, 'time': t, 'type': 'hold',
                    'confidence': confidence, 'hold_duration': hold_dur,
                    'num_arrows': 1,
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

        # Beat-snap (and remember the grid slot for subdivision color).
        if self.snap_to_beats and len(beat_times) > 1:
            beat_interval = 60.0 / tempo
            grid = []
            gt = beat_times[0] if len(beat_times) > 0 else 0.0
            while gt <= duration:
                grid.append(gt)
                gt += beat_interval / 4
            grid = np.array(grid)
            for event in note_events:
                nearest_idx = int(np.argmin(np.abs(grid - event['time'])))
                event['time'] = float(grid[nearest_idx])
                event['grid_idx'] = nearest_idx

        # Sort and enforce min-gap.
        note_events.sort(key=lambda e: e['time'])
        filtered = []
        last_time = -1.0
        for event in note_events:
            if event['time'] - last_time >= min_gap:
                filtered.append(event)
                last_time = event['time']
        note_events = filtered

        # ================================================================
        # Phase 2: Determine WHICH arrows (autoregressive)
        # ================================================================

        assigner = FootStateArrowAssigner(
            seed=audio_seed,
            allowed_patterns=diff_config.allowed_patterns,
        )
        steps = []
        prev_arrow_vec = np.zeros(4, dtype=np.float32)

        for i, event in enumerate(note_events):
            t = round(event['time'], 3)
            subdivision = self._subdivision_from_grid(
                event.get('grid_idx'), t, tempo, beat_times,
            )
            frame_idx = int(round(t * FRAMES_PER_SECOND))
            frame_idx = max(0, min(frame_idx, T - 1))

            # Look up audio features for the fallback assigner.
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

            # Re-score arrows with the autoregressive prev_arrow context.
            arrow_logits_t = self._arrow_logits_at_frame(
                features[frame_idx], prev_arrow_vec,
            )
            arrow_p = 1.0 / (1.0 + np.exp(-arrow_logits_t))   # sigmoid
            target_arity = 2 if event.get('num_arrows', 1) >= 2 else 1
            chosen_arrows = self._choose_arrows(arrow_p, target_arity)

            if event['type'] == 'hold':
                if not chosen_arrows:
                    # Fallback: arrow head fired on nothing → defer to assigner.
                    arrow = assigner.start_hold(
                        t, event['hold_duration'], energy, brightness,
                    )
                    chosen_arrows = [arrow]
                else:
                    # Keep just one arrow for the hold and inform the assigner
                    # so its hold tracking stays consistent if used later.
                    arrow = ARROW_DIRECTIONS[chosen_arrows[0]]
                    assigner.start_hold(
                        t, event['hold_duration'], energy, brightness,
                    )
                    chosen_arrows = [arrow]
                steps.append(Step(
                    time=t,
                    arrows=chosen_arrows,
                    step_type=StepType.HOLD,
                    hold_duration=round(event['hold_duration'], 3),
                    beat_subdivision=subdivision,
                ))
            else:
                if not chosen_arrows:
                    # Fallback for empty arrow set.
                    if target_arity >= 2:
                        chosen_arrows = assigner.assign_jump(t, energy, brightness)
                    else:
                        chosen_arrows = [
                            assigner.assign_single(t, energy, brightness)
                        ]
                else:
                    chosen_arrows = [ARROW_DIRECTIONS[a] for a in chosen_arrows]
                steps.append(Step(
                    time=t,
                    arrows=chosen_arrows,
                    step_type=StepType.TAP,
                    beat_subdivision=subdivision,
                ))

            # Update prev_arrow_vec for the next frame.
            new_prev = np.zeros(4, dtype=np.float32)
            for arr in steps[-1].arrows:
                idx = ARROW_DIRECTIONS.index(arr)
                new_prev[idx] = 1.0
            prev_arrow_vec = new_prev

        return steps

    def _choose_arrows(
        self, arrow_p: np.ndarray, target_arity: int,
    ) -> List[int]:
        """Pick arrow indices from per-arrow sigmoid probs.

        target_arity = 1 (tap/hold) or 2 (jump). When per-arrow thresholds are
        configured (newer checkpoints), arrows must clear their per-channel
        threshold; otherwise we fall back to argmax of the requested arity.
        Returns column indices in [0,3] (L=0, D=1, U=2, R=3).
        """
        thresholds = getattr(self, 'arrow_thresholds', None)
        if thresholds is not None:
            passing = [a for a in range(4) if arrow_p[a] >= thresholds[a]]
        else:
            passing = [a for a in range(4) if arrow_p[a] >= self.confidence_threshold]

        if target_arity == 1:
            if not passing:
                return []
            best = max(passing, key=lambda a: arrow_p[a])
            return [best]
        # target_arity >= 2 (jump)
        if len(passing) >= 2:
            top2 = sorted(passing, key=lambda a: -arrow_p[a])[:2]
            return sorted(top2)
        if len(passing) == 1:
            # Only one arrow crossed threshold — promote runner-up if it has
            # any reasonable signal (>= 0.5 of the top arrow's prob).
            top = passing[0]
            others = [a for a in range(4) if a != top]
            runner = max(others, key=lambda a: arrow_p[a])
            if arrow_p[runner] >= 0.5 * arrow_p[top]:
                return sorted([top, runner])
            return [top]
        return []

    # 16th-grid index → DDR-style subdivision color, mirroring the
    # rational-beat logic in services/sm_parser._beat_subdivision:
    #   slot 0  → on the beat      → quarter (red)
    #   slot 2  → on the half-beat → eighth  (blue)
    #   slot 1, 3 → quarter-beat   → sixteenth (yellow)
    _SUBDIVISION_BY_SLOT = (
        BeatSubdivision.QUARTER,
        BeatSubdivision.SIXTEENTH,
        BeatSubdivision.EIGHTH,
        BeatSubdivision.SIXTEENTH,
    )

    def _subdivision_from_grid(
        self,
        grid_idx: Optional[int],
        time: float,
        tempo: float,
        beat_times: np.ndarray,
    ) -> BeatSubdivision:
        """Subdivision from the snap-grid slot when available, else fall back
        to modular phase. The slot path is exact integer math and matches the
        sm_parser convention; the modular path only runs when snap_to_beats is
        disabled or the audio had < 2 detected beats.
        """
        if grid_idx is not None:
            return self._SUBDIVISION_BY_SLOT[grid_idx % 4]

        if len(beat_times) == 0 or tempo <= 0:
            return BeatSubdivision.QUARTER

        beat_interval = 60.0 / tempo
        anchor = float(beat_times[0])
        beat_position = ((time - anchor) % beat_interval) / beat_interval

        if beat_position < 0.125 or beat_position > 0.875:
            return BeatSubdivision.QUARTER
        if abs(beat_position - 0.5) < 0.125:
            return BeatSubdivision.EIGHTH
        return BeatSubdivision.SIXTEENTH
