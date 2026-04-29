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

# Arrow index to Direction mapping
ARROW_DIRECTIONS = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]

# Difficulty name mapping (app difficulty → model difficulty_id)
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

    Separates arrow selection from onset detection so the model only needs
    to predict WHEN/WHAT (timing + note type) and this class handles WHICH
    arrows deterministically — but with musical variation.

    Determinism guarantee: same seed + same event sequence + same audio
    features = identical arrow assignments.
    """

    # Natural panel assignments per foot
    LEFT_FOOT_PANELS = [Direction.LEFT, Direction.DOWN]
    RIGHT_FOOT_PANELS = [Direction.UP, Direction.RIGHT]

    # All four panels for crossover moves
    ALL_PANELS = [Direction.LEFT, Direction.DOWN, Direction.UP, Direction.RIGHT]

    # --- Stream patterns (sequences of arrow indices 0-3: L D U R) ---
    STREAM_PATTERNS = [
        [0, 2, 1, 3],  # L U D R — standard weave
        [3, 1, 2, 0],  # R D U L — reverse weave
        [0, 1, 2, 3],  # L D U R — staircase up
        [3, 2, 1, 0],  # R U D L — staircase down
        [0, 3, 1, 2],  # L R D U — wide crossover
        [1, 2, 0, 3],  # D U L R — inside-out
    ]

    # --- Jump patterns: (left_foot_arrow, right_foot_arrow) ---
    JUMP_PATTERNS = [
        (Direction.LEFT, Direction.RIGHT),   # wide
        (Direction.DOWN, Direction.UP),       # center
        (Direction.LEFT, Direction.UP),       # left-leaning
        (Direction.DOWN, Direction.RIGHT),    # right-leaning
    ]

    # --- Candle patterns: one foot planted, other foot hits 3 panels ---
    # (planted_arrow, [moving_foot_sequence])
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
        confidence_threshold: float = 0.3,
        min_note_gap: float = 0.05,
        snap_to_beats: bool = True,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.confidence_threshold = confidence_threshold
        self.min_note_gap = min_note_gap
        self.snap_to_beats = snap_to_beats

        self.model = None
        # Per-difficulty inference-time default density (steps/sec). Populated
        # from the checkpoint if it was saved with one, else falls back to the
        # hardcoded preset midpoints.
        self.default_density_by_id = DEFAULT_DENSITY_BY_ID.clone()
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
        onset_probs, type_probs, duration_pred = self._predict_chunked(
            mel_frames, difficulty_id, target_density
        )

        # Post-process into steps using difficulty preset
        diff_config = get_difficulty_config(difficulty)
        logger.info("Post-processing predictions...")
        steps = self._postprocess(
            onset_probs, type_probs, duration_pred,
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
            onset_probs:   [T] float32 sigmoid("note present")
            type_probs:    [T, 3] float32 softmax over {tap, jump, hold_start}
            duration_pred: [T] float32 predicted hold duration in seconds
        """
        T = mel_frames.shape[0]
        onset_sum = np.zeros(T, dtype=np.float32)
        type_sum = np.zeros((T, 3), dtype=np.float32)
        dur_sum = np.zeros(T, dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)

        stride = self.chunk_frames - self.overlap_frames
        density_norm = (target_density - DENSITY_MEAN) / DENSITY_STD

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

            onset_logits, type_logits, dur_pred = self.model(mel_tensor, diff_tensor, density_tensor)
            onset_p = torch.sigmoid(onset_logits.float()).cpu().numpy()[0, :, 0]   # [T_chunk]
            type_p = torch.softmax(type_logits.float(), dim=-1).cpu().numpy()[0]   # [T_chunk, 3]
            dur_p = dur_pred.float().cpu().numpy()[0, :, 0]                         # [T_chunk]

            valid_len = min(end - start, self.chunk_frames)
            onset_sum[start:start + valid_len] += onset_p[:valid_len]
            type_sum[start:start + valid_len] += type_p[:valid_len]
            dur_sum[start:start + valid_len] += dur_p[:valid_len]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)
        onset_probs = onset_sum / counts
        type_probs = type_sum / counts[:, None]
        duration_pred = dur_sum / counts
        return onset_probs, type_probs, duration_pred

    def _postprocess(
        self,
        onset_probs: np.ndarray,    # [T] — single onset probability per frame
        type_probs: np.ndarray,     # [T, 3]  (0=tap, 1=jump, 2=hold_start)
        duration_pred: np.ndarray,  # [T] predicted hold duration in seconds
        tempo: float,
        beat_times: np.ndarray,
        duration: float,
        diff_config,
        energy_curve: Optional[np.ndarray] = None,     # [T_audio] normalized RMS
        brightness_curve: Optional[np.ndarray] = None,  # [T_audio] normalized centroid
        audio_seed: int = 0,
    ) -> List[Step]:
        """
        Convert (onset, type, duration) predictions into Step objects.

        Two-phase pipeline:
            Phase 1 (from model): Detect WHEN notes occur, WHAT type, and HOW LONG
                1. NMS peak picking on onset signal
                2. Classify each peak: tap, jump, hold_start
                3. For hold_start peaks, read predicted duration directly
                4. Density cap, beat-snap, min-gap enforcement

            Phase 2 (audio-driven, deterministic): Determine WHICH arrows
                5. FootStateArrowAssigner uses seeded PRNG + audio features
                   (energy, brightness) to select from expanded pattern vocab
                6. Build Step objects
        """
        T = onset_probs.shape[0]

        logger.info(
            f"[postprocess debug] onset_probs.shape={onset_probs.shape} "
            f"max={onset_probs.max():.4f} mean={onset_probs.mean():.4f} "
            f"p95={np.percentile(onset_probs, 95):.4f} "
            f"p99={np.percentile(onset_probs, 99):.4f}"
        )

        min_gap = max(diff_config.min_gap, self.min_note_gap)
        nms_window_frames = max(1, int(round(min_gap * FRAMES_PER_SECOND)))

        # Per-difficulty jump allowance
        jumps_allowed = 'jump' in diff_config.allowed_patterns

        # ================================================================
        # Phase 1: Detect WHEN, WHAT, and HOW LONG (arrow-agnostic)
        # ================================================================

        # Step 1: NMS peak picking on onset curve
        peaks = []  # list of (frame, time, confidence)
        for frame in range(T):
            p = float(onset_probs[frame])
            if p < max(self.confidence_threshold, 1e-4):
                continue
            lo = max(0, frame - nms_window_frames)
            hi = min(T, frame + nms_window_frames + 1)
            if p < onset_probs[lo:hi].max():
                continue
            t = frame / FRAMES_PER_SECOND
            if t < 0 or t > duration:
                continue
            peaks.append((frame, t, p))

        # Step 2: Classify each peak and read duration for holds
        note_events = []
        for frame, t, confidence in peaks:
            # type_probs[frame]: [3] = {tap, jump, hold_start}
            note_type = int(np.argmax(type_probs[frame]))

            if note_type == 0:  # tap
                note_events.append({
                    'time': t, 'type': 'tap', 'confidence': confidence,
                    'num_arrows': 1,
                })
            elif note_type == 1:  # jump
                num_arrows = 2 if jumps_allowed else 1
                note_events.append({
                    'time': t, 'type': 'tap', 'confidence': confidence,
                    'num_arrows': num_arrows,
                })
            else:  # hold_start — read duration from the duration head
                hold_dur = float(np.clip(duration_pred[frame], 0.2, 5.0))
                note_events.append({
                    'time': t, 'type': 'hold', 'confidence': confidence,
                    'hold_duration': hold_dur, 'num_arrows': 1,
                })

        # Step 3: Density cap — keep top-N events by confidence
        avg_density = (diff_config.min_density + diff_config.max_density) / 2.0
        target_notes = max(1, int(round(avg_density * duration)))
        logger.info(
            f"[postprocess debug] peaks={len(peaks)} note_events={len(note_events)} "
            f"target_notes={target_notes} (density={avg_density:.2f}/s, dur={duration:.1f}s)"
        )
        if len(note_events) > target_notes:
            # Keep holds unconditionally, cap taps
            holds = [e for e in note_events if e['type'] == 'hold']
            taps = [e for e in note_events if e['type'] == 'tap']
            taps.sort(key=lambda e: e['confidence'], reverse=True)
            remaining = max(0, target_notes - len(holds))
            note_events = holds + taps[:remaining]

        # Step 4: Beat-snap
        if self.snap_to_beats and len(beat_times) > 1:
            beat_interval = 60.0 / tempo
            grid = []
            gt = beat_times[0] if len(beat_times) > 0 else 0.0
            while gt <= duration:
                grid.append(gt)
                gt += beat_interval / 4  # 16th note grid
            grid = np.array(grid)

            for event in note_events:
                nearest_idx = int(np.argmin(np.abs(grid - event['time'])))
                event['time'] = float(grid[nearest_idx])

        # Step 5: Sort and enforce minimum gap
        note_events.sort(key=lambda e: e['time'])
        filtered = []
        last_time = -1.0
        for event in note_events:
            if event['time'] - last_time >= min_gap:
                filtered.append(event)
                last_time = event['time']
        note_events = filtered

        # ================================================================
        # Phase 2: Determine WHICH arrows (audio-driven foot-state machine)
        # ================================================================

        assigner = FootStateArrowAssigner(
            seed=audio_seed,
            allowed_patterns=diff_config.allowed_patterns,
        )
        steps = []

        for i, event in enumerate(note_events):
            t = round(event['time'], 3)
            subdivision = self._get_beat_subdivision(t, tempo, beat_times)

            # Look up per-event audio features from the precomputed curves
            frame_idx = int(round(t * FRAMES_PER_SECOND))
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

            # Try to start a stream on high-energy taps
            remaining = len(note_events) - i
            if event['type'] == 'tap' and event.get('num_arrows', 1) == 1:
                assigner.maybe_start_stream(
                    energy, remaining, diff_config.max_stream_length,
                )

            if event['type'] == 'hold':
                arrow = assigner.start_hold(
                    t, event['hold_duration'], energy, brightness,
                )
                steps.append(Step(
                    time=t,
                    arrows=[arrow],
                    step_type=StepType.HOLD,
                    hold_duration=round(event['hold_duration'], 3),
                    beat_subdivision=subdivision,
                ))
            elif event.get('num_arrows', 1) >= 2:
                arrows = assigner.assign_jump(t, energy, brightness)
                steps.append(Step(
                    time=t,
                    arrows=arrows,
                    step_type=StepType.TAP,
                    beat_subdivision=subdivision,
                ))
            else:
                arrow = assigner.assign_single(t, energy, brightness)
                steps.append(Step(
                    time=t,
                    arrows=[arrow],
                    step_type=StepType.TAP,
                    beat_subdivision=subdivision,
                ))

        return steps

    def _get_beat_subdivision(
        self, time: float, tempo: float, beat_times: np.ndarray
    ) -> BeatSubdivision:
        """Determine the beat subdivision of a note time."""
        if len(beat_times) == 0:
            return BeatSubdivision.QUARTER

        beat_interval = 60.0 / tempo

        # Find position within the beat (0.0 to 1.0)
        beat_position = (time % beat_interval) / beat_interval

        # Quarter note: lands on the beat (position ~0.0 or ~1.0)
        if beat_position < 0.125 or beat_position > 0.875:
            return BeatSubdivision.QUARTER
        # Eighth note: lands on the half-beat (position ~0.5)
        if abs(beat_position - 0.5) < 0.125:
            return BeatSubdivision.EIGHTH
        # Sixteenth note: lands at 0.25 or 0.75
        return BeatSubdivision.SIXTEENTH
