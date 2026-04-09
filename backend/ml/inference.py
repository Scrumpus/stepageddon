"""
Inference module: generate step charts from audio using trained model.

Handles chunked processing for arbitrary-length songs, post-processing
(beat-snapping, hold cleanup, minimum gap), and conversion to the
existing Chart/Step schema.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import librosa

from ml.model import StepChartModel
from ml.dataset import DEFAULT_DENSITY_BY_ID, DENSITY_MEAN, DENSITY_STD
from ml.prepare_data import SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, FRAMES_PER_SECOND
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


class MLChartGenerator:
    """
    Generate step charts using the trained neural network.

    Drop-in alternative to ChartGenerationPipeline.
    """

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

        logger.info(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    def generate_from_audio(
        self,
        audio_path: str,
        difficulty: str = 'medium',
        target_density: Optional[float] = None,
    ) -> Chart:
        """
        Generate a step chart from an audio file.

        Compatible with ChartGenerationPipeline.generate_from_audio() interface.

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

        # Extract mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_frames = mel_db.T.astype(np.float32)  # [T, N_MELS]

        # Detect tempo and beats for post-processing
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Run model inference in chunks
        logger.info(f"Running inference (difficulty={difficulty}, id={difficulty_id})...")
        onset_probs, type_probs = self._predict_chunked(
            mel_frames, difficulty_id, target_density
        )

        # Post-process into steps using difficulty preset
        diff_config = get_difficulty_config(difficulty)
        logger.info("Post-processing predictions...")
        steps = self._postprocess(
            onset_probs, type_probs, tempo, beat_times, duration, diff_config
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
            onset_probs: [T, 4] float32 sigmoid("note present")
            type_probs:  [T, 4, 3] float32 softmax over {tap, hold_start, hold_end}
        """
        T = mel_frames.shape[0]
        onset_sum = np.zeros((T, 4), dtype=np.float32)
        type_sum = np.zeros((T, 4, 3), dtype=np.float32)
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

            onset_logits, type_logits = self.model(mel_tensor, diff_tensor, density_tensor)
            onset_p = torch.sigmoid(onset_logits.float()).cpu().numpy()[0]       # [T_chunk, 4]
            type_p = torch.softmax(type_logits.float(), dim=-1).cpu().numpy()[0]  # [T_chunk, 4, 3]

            valid_len = min(end - start, self.chunk_frames)
            onset_sum[start:start + valid_len] += onset_p[:valid_len]
            type_sum[start:start + valid_len] += type_p[:valid_len]
            counts[start:start + valid_len] += 1.0

        counts = np.maximum(counts, 1.0)
        onset_probs = onset_sum / counts[:, None]
        type_probs = type_sum / counts[:, None, None]
        return onset_probs, type_probs

    def _postprocess(
        self,
        onset_probs: np.ndarray,   # [T, 4]
        type_probs: np.ndarray,    # [T, 4, 3]  (0=tap, 1=hold_start, 2=hold_end)
        tempo: float,
        beat_times: np.ndarray,
        duration: float,
        diff_config,
    ) -> List[Step]:
        """
        Convert (onset, type) probabilities into Step objects.

        Pipeline:
            1. Per-arrow onset peak picking (NMS) on sigmoid(onset_logits)
            2. For each picked onset: argmax over type head decides
               tap / hold_start / hold_end
            3. Pair hold_start -> hold_end per arrow to form holds
            4. Difficulty-aware density target: keep top-N tap events globally
            5. Snap to beat grid; enforce min_gap from preset
            6. Cap simultaneous arrows per frame based on difficulty
            7. Build Step objects
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

        # ---- Step 1+2: per-arrow onset NMS + type decoding ----
        tap_candidates = []     # list of {time, arrow, confidence}
        hold_start_events = []  # per-arrow list of (frame, time, confidence)
        hold_end_events = []    # per-arrow list of (frame, time, confidence)
        per_arrow_starts = [[] for _ in range(4)]
        per_arrow_ends = [[] for _ in range(4)]

        for arrow in range(4):
            onset = onset_probs[:, arrow]  # [T]
            for frame in range(T):
                p = float(onset[frame])
                if p < max(self.confidence_threshold, 1e-4):
                    continue
                lo = max(0, frame - nms_window_frames)
                hi = min(T, frame + nms_window_frames + 1)
                if p < onset[lo:hi].max():
                    continue
                t = frame / FRAMES_PER_SECOND
                if t < 0 or t > duration:
                    continue
                ttype = int(np.argmax(type_probs[frame, arrow]))
                if ttype == 0:
                    tap_candidates.append({
                        'time': t, 'arrow': arrow, 'type': 'tap', 'confidence': p,
                    })
                elif ttype == 1:
                    per_arrow_starts[arrow].append((frame, t, p))
                else:  # ttype == 2
                    per_arrow_ends[arrow].append((frame, t, p))

        # ---- Step 3: pair hold_start -> nearest following hold_end ----
        hold_events = []
        for arrow in range(4):
            ends = per_arrow_ends[arrow][:]
            ends_idx = 0
            ends.sort(key=lambda x: x[0])
            for sf, st, sp in per_arrow_starts[arrow]:
                # advance ends_idx to first end with frame > sf
                while ends_idx < len(ends) and ends[ends_idx][0] <= sf:
                    ends_idx += 1
                if ends_idx >= len(ends):
                    # No matching end; treat as tap fallback
                    tap_candidates.append({
                        'time': st, 'arrow': arrow, 'type': 'tap', 'confidence': sp,
                    })
                    continue
                ef, et, ep = ends[ends_idx]
                hold_dur = et - st
                if 0.2 <= hold_dur <= 5.0:
                    hold_events.append({
                        'time': st, 'arrow': arrow, 'type': 'hold',
                        'hold_duration': hold_dur,
                        'confidence': (sp + ep) / 2.0,
                    })
                    ends_idx += 1
                else:
                    tap_candidates.append({
                        'time': st, 'arrow': arrow, 'type': 'tap', 'confidence': sp,
                    })

        # ---- Step 3: difficulty-aware density targeting ----
        # Compute target tap count from preset density (steps/sec)
        avg_density = (diff_config.min_density + diff_config.max_density) / 2.0
        target_taps = max(1, int(round(avg_density * duration)))
        logger.info(
            f"[postprocess debug] tap_candidates pre-cap={len(tap_candidates)} "
            f"hold_events={len(hold_events)} target_taps={target_taps} "
            f"(density={avg_density:.2f}/s, dur={duration:.1f}s)"
        )
        # Keep top-N tap candidates globally by confidence
        tap_candidates.sort(key=lambda e: e['confidence'], reverse=True)
        tap_candidates = tap_candidates[:target_taps]

        note_events = tap_candidates + hold_events

        # Step 3: Snap to beat grid
        if self.snap_to_beats and len(beat_times) > 1:
            beat_interval = 60.0 / tempo
            # Build grid of valid positions (16th note resolution)
            grid = []
            t = beat_times[0] if len(beat_times) > 0 else 0.0
            while t <= duration:
                grid.append(t)
                t += beat_interval / 4  # 16th notes

            grid = np.array(grid)

            for event in note_events:
                nearest_idx = np.argmin(np.abs(grid - event['time']))
                event['time'] = float(grid[nearest_idx])

        # Step 4: Sort by time and enforce minimum gap (from preset) per arrow
        note_events.sort(key=lambda e: (e['time'], e['arrow']))
        filtered = []
        last_time_per_arrow = {}

        for event in note_events:
            arrow = event['arrow']
            last_t = last_time_per_arrow.get(arrow, -1.0)
            if event['time'] - last_t >= min_gap:
                filtered.append(event)
                last_time_per_arrow[arrow] = event['time']

        # Step 5: Group simultaneous notes (same time) into single Steps
        # Apply per-difficulty cap on simultaneous arrows
        diff_name = diff_config.name
        if diff_name in ('beginner', 'easy'):
            max_arrows_per_step = 1
            high_conf_jump_threshold = 1.1  # never allow >cap
        elif diff_name in ('medium', 'hard'):
            max_arrows_per_step = 2
            high_conf_jump_threshold = 0.6
        else:  # challenge
            max_arrows_per_step = 2
            high_conf_jump_threshold = 0.6

        steps = []
        time_groups = {}
        for event in filtered:
            t = round(event['time'], 4)
            if t not in time_groups:
                time_groups[t] = []
            time_groups[t].append(event)

        for t in sorted(time_groups.keys()):
            events = time_groups[t]

            # Separate taps and holds
            taps = [e for e in events if e['type'] == 'tap']
            holds = [e for e in events if e['type'] == 'hold']

            # Cap simultaneous tap arrows by difficulty
            if taps:
                taps.sort(key=lambda e: e['confidence'], reverse=True)
                cap = max_arrows_per_step
                # Allow >cap only when ALL extra arrows are high-confidence
                extras = taps[cap:]
                if extras and all(e['confidence'] >= high_conf_jump_threshold for e in extras):
                    cap = min(4, len(taps))
                taps = taps[:cap]

            # Create tap step if any
            if taps:
                arrows = list(set(ARROW_DIRECTIONS[e['arrow']] for e in taps))
                subdivision = self._get_beat_subdivision(t, tempo, beat_times)
                steps.append(Step(
                    time=round(t, 3),
                    arrows=arrows,
                    step_type=StepType.TAP,
                    beat_subdivision=subdivision,
                ))

            # Create hold steps
            for hold in holds:
                subdivision = self._get_beat_subdivision(t, tempo, beat_times)
                steps.append(Step(
                    time=round(t, 3),
                    arrows=[ARROW_DIRECTIONS[hold['arrow']]],
                    step_type=StepType.HOLD,
                    hold_duration=round(hold['hold_duration'], 3),
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

        # Find nearest beat
        nearest_beat_idx = np.argmin(np.abs(beat_times - time))
        dist_to_beat = abs(time - beat_times[nearest_beat_idx])

        if dist_to_beat < beat_interval * 0.1:
            return BeatSubdivision.QUARTER
        elif dist_to_beat < beat_interval * 0.3:
            return BeatSubdivision.EIGHTH
        else:
            return BeatSubdivision.SIXTEENTH
