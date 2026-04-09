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
from ml.prepare_data import (
    SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, FRAMES_PER_SECOND,
    N_INPUT_CHANNELS, compute_rhythm_channels,
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
            n_mels=N_INPUT_CHANNELS,
            hidden_dim=args.get('hidden_dim', 256),
            n_heads=args.get('n_heads', 8),
            n_transformer_layers=args.get('n_layers', 4),
            n_difficulties=5,
        ).to(self.device)

        # New checkpoints have a density_proj layer in each FiLM block; old ones
        # do not. strict=False so old checkpoints still load (the density layer
        # is zero-init, so the model behaves as before until retrained).
        missing, unexpected = self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
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
        mel_only = mel_db.T.astype(np.float32)  # [T, N_MELS]

        # Rhythm-aware auxiliary channels must match training layout —
        # concatenated after the mel bins along the feature axis.
        rhythm = compute_rhythm_channels(y, sr, mel_only.shape[0]).astype(np.float32)
        mel_frames = np.concatenate([mel_only, rhythm], axis=-1)  # [T, N_INPUT_CHANNELS]

        # Detect tempo and beats for post-processing
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        tempo = float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Run model inference in chunks
        logger.info(f"Running inference (difficulty={difficulty}, id={difficulty_id})...")
        raw_predictions, placement_probs = self._predict_chunked(
            mel_frames, difficulty_id, target_density
        )

        # Post-process into steps using difficulty preset
        diff_config = get_difficulty_config(difficulty)
        logger.info("Post-processing predictions...")
        steps = self._postprocess(
            raw_predictions, placement_probs, tempo, beat_times, duration, diff_config
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
    ) -> np.ndarray:
        """
        Run model on overlapping chunks and merge predictions.

        Args:
            mel_frames: [T, N_MELS] float32
            difficulty_id: int 0-4
            target_density: desired chart density in steps/sec

        Returns:
            tuple of (arrow_probs, placement_probs) where
                arrow_probs: [T, 4, 3] float32 sigmoid probabilities, last
                    axis = (tap, hold_state, hold_end).
                placement_probs: [T] float32 sigmoid probability that any
                    arrow has a step at this frame.
        """
        T = mel_frames.shape[0]
        all_probs = np.zeros((T, 4, 3), dtype=np.float32)
        all_placement = np.zeros(T, dtype=np.float32)
        all_counts = np.zeros(T, dtype=np.float32)

        stride = self.chunk_frames - self.overlap_frames

        # Normalize density to match training-time scale.
        density_norm = (target_density - DENSITY_MEAN) / DENSITY_STD

        for start in range(0, T, stride):
            end = min(start + self.chunk_frames, T)
            chunk = mel_frames[start:end]

            # Pad if chunk is shorter than expected
            if chunk.shape[0] < self.chunk_frames:
                pad = np.zeros(
                    (self.chunk_frames - chunk.shape[0], mel_frames.shape[1]),
                    dtype=np.float32,
                )
                chunk = np.concatenate([chunk, pad], axis=0)

            mel_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            diff_tensor = torch.tensor([difficulty_id], dtype=torch.long, device=self.device)
            density_tensor = torch.tensor(
                [density_norm], dtype=torch.float32, device=self.device
            )

            logits = self.model(mel_tensor, diff_tensor, density_tensor)
            tap_p = torch.sigmoid(logits['tap'])[0]
            hold_state_p = torch.sigmoid(logits['hold_state'])[0]
            hold_end_p = torch.sigmoid(logits['hold_end'])[0]
            probs = torch.stack([tap_p, hold_state_p, hold_end_p], dim=-1).cpu().numpy()
            placement_p = torch.sigmoid(logits['placement'])[0].cpu().numpy()

            valid_len = min(end - start, self.chunk_frames)
            all_probs[start:start + valid_len] += probs[:valid_len]
            all_placement[start:start + valid_len] += placement_p[:valid_len]
            all_counts[start:start + valid_len] += 1.0

        # Average overlapping regions
        all_counts = np.maximum(all_counts, 1.0)
        all_probs /= all_counts[:, None, None]
        all_placement /= all_counts

        return all_probs, all_placement

    def _postprocess(
        self,
        probs: np.ndarray,
        placement_probs: np.ndarray,
        tempo: float,
        beat_times: np.ndarray,
        duration: float,
        diff_config,
    ) -> List[Step]:
        """
        Convert raw model probabilities into Step objects.

        Pipeline:
            1. Per-arrow peak picking (NMS) on tap probability over time
            2. Detect hold_start/hold_end events via independent threshold
            3. Difficulty-aware density target: keep top-N tap events globally
            4. Snap to beat grid; enforce min_gap from preset
            5. Cap simultaneous arrows per frame based on difficulty
            6. Build Step objects
        """
        T = probs.shape[0]

        # probs axis layout from _predict_chunked: [T, 4, 3] = (tap, hold_state, hold_end)
        TAP_IDX, HOLD_STATE_IDX, HOLD_END_IDX = 0, 1, 2

        # Combine the per-arrow tap head with the arrow-agnostic placement
        # head: p(tap_i) ≈ p(placement) * p(arrow=i | placement). The
        # placement head is trained explicitly against "any arrow has a
        # step here", so this product gives an absolute-confidence signal
        # that scales from ~0 in silence to ~1 on real onsets — much
        # cleaner for NMS thresholding than the raw tap head alone.
        tap_gated = probs[:, :, TAP_IDX] * placement_probs[:, None]  # [T, 4]
        logger.info(
            f"[postprocess debug] placement: max={placement_probs.max():.4f} "
            f"mean={placement_probs.mean():.4f} "
            f"p95={np.percentile(placement_probs, 95):.4f}"
        )

        # ---- DEBUG: probability distribution diagnostics ----
        tap_probs_all = probs[:, :, TAP_IDX]
        hold_state_all = probs[:, :, HOLD_STATE_IDX]
        hold_end_all = probs[:, :, HOLD_END_IDX]
        logger.info(f"[postprocess debug] probs.shape={probs.shape}")
        logger.info(
            f"[postprocess debug] tap: max={tap_probs_all.max():.4f} "
            f"mean={tap_probs_all.mean():.4f} "
            f"p95={np.percentile(tap_probs_all, 95):.4f} "
            f"p99={np.percentile(tap_probs_all, 99):.4f}"
        )
        logger.info(
            f"[postprocess debug] hold_state max={hold_state_all.max():.4f}  "
            f"hold_end max={hold_end_all.max():.4f}"
        )
        any_arrow_max = tap_probs_all.max(axis=1)
        for thr in (0.1, 0.2, 0.3, 0.5):
            logger.info(
                f"[postprocess debug] frames with any-arrow tap > {thr}: "
                f"{int((any_arrow_max > thr).sum())}/{T}"
            )

        # min_gap from preset overrides constructor default
        min_gap = max(diff_config.min_gap, self.min_note_gap)
        nms_window_frames = max(1, int(round(min_gap * FRAMES_PER_SECOND)))

        # ---- Step 1: per-arrow tap peak picking via NMS ----
        # Tap head is now a per-arrow sigmoid, so absolute confidence is
        # meaningful (caps near 1.0 on well-trained models). We still keep
        # a tiny floor + NMS so post-processing stays self-calibrating if
        # the trained model is underconfident.
        tap_floor = 1e-3
        tap_candidates = []  # list of {time, arrow, confidence, type='tap'}
        for arrow in range(4):
            tap_probs = tap_gated[:, arrow]  # [T]
            for frame in range(T):
                p = float(tap_probs[frame])
                if p < tap_floor:
                    continue
                lo = max(0, frame - nms_window_frames)
                hi = min(T, frame + nms_window_frames + 1)
                if p < tap_probs[lo:hi].max():
                    continue
                t = frame / FRAMES_PER_SECOND
                if t < 0 or t > duration:
                    continue
                tap_candidates.append({
                    'time': t,
                    'arrow': arrow,
                    'type': 'tap',
                    'confidence': p,
                })

        # ---- Step 2: hold detection via hold_state run-length decoding ----
        # Find contiguous runs where hold_state > threshold. The run start is
        # the note onset, the last frame in the run is the release; duration
        # is the span in seconds. No start/end pairing required — this is
        # the big win from the three-head reformulation.
        hold_events = []
        hold_thr = self.confidence_threshold
        for arrow in range(4):
            state = probs[:, arrow, HOLD_STATE_IDX] > hold_thr  # [T] bool
            if not state.any():
                continue
            # Build (start, end_inclusive) runs.
            edges = np.diff(state.astype(np.int8), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0] - 1  # inclusive end
            for s, e in zip(starts, ends):
                start_time = s / FRAMES_PER_SECOND
                end_time = (e + 1) / FRAMES_PER_SECOND  # release just after last in-hold frame
                hold_dur = end_time - start_time
                if not (0.2 <= hold_dur <= 5.0):
                    continue
                # Confidence = mean hold_state prob over the run, gently
                # boosted by the peak hold_end prob near the release (the
                # end head is the more informative of the two for choosing
                # between competing candidates).
                run_conf = float(probs[s:e + 1, arrow, HOLD_STATE_IDX].mean())
                end_window = probs[max(0, e - 2):e + 2, arrow, HOLD_END_IDX]
                end_conf = float(end_window.max()) if end_window.size else 0.0
                hold_events.append({
                    'time': start_time,
                    'arrow': arrow,
                    'type': 'hold',
                    'hold_duration': hold_dur,
                    'confidence': 0.7 * run_conf + 0.3 * end_conf,
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
