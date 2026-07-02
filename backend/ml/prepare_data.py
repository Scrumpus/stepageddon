"""
Preprocessing script: converts .sm charts + audio into training data.

Run this locally before training on Colab:
    python -m ml.prepare_data --charts-dir charts/ --output-dir ml/training_data/

Output (format_version=8):
    training_data/
    ├── manifest.json    # {format_version, feat_mean[88], feat_std[88], entries: [...]}
    ├── song_0001.npz    # one file per song: feats[T,88] + per-chart label arrays
    │                     # + section_positions[T] + near_boundary[T]
    └── ...

`feats` channels (88 total, all aligned at ~100.23 fps):
    [0..79]  — fixed-dB scaled mel (clipped to [DB_MIN, DB_MAX], mapped to [0,1])
    [80]     — librosa.onset.onset_strength (per-song z-normed)
    [81..87] — librosa.feature.spectral_contrast (7 bands; per-song z-normed)

v8 adds structural features:
    section_positions[T] — position within detected section [0, 1]
    near_boundary[T]     — 1.0 near section boundary, 0.0 elsewhere
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    print("librosa is required: pip install librosa")
    sys.exit(1)

from src.songs.utils import parse_sm_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# v8: adds structural features (section_positions, near_boundary) to npz output.
FORMAT_VERSION = 8

# Audio processing constants
SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 220  # ~100 fps (22050 / 220 = 100.23 fps)
N_FFT = 2048
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH  # ~100.23

# Channel layout for the v8 feats tensor.
N_ONSET_STRENGTH = 1
N_SPECTRAL_CONTRAST = 7
N_FEAT_CHANNELS = N_MELS + N_ONSET_STRENGTH + N_SPECTRAL_CONTRAST  # 88

# Fixed dB scaling for mel storage.
DB_MIN = -80.0
DB_MAX = 20.0
DB_RANGE = DB_MAX - DB_MIN

# Label encoding (arrow-agnostic note types).
LABEL_NONE = 0
LABEL_TAP = 1
LABEL_JUMP = 2
LABEL_HOLD_START = 3


class FeatStatsAccumulator:
    """Online float64 per-channel mean/std accumulator over the 88 feats channels."""

    def __init__(self, n_channels: int = N_FEAT_CHANNELS) -> None:
        self.n_channels = int(n_channels)
        self.sum = np.zeros(self.n_channels, dtype=np.float64)
        self.sum_sq = np.zeros(self.n_channels, dtype=np.float64)
        self.count = 0

    def update(self, feats: np.ndarray) -> None:
        a = feats.astype(np.float64, copy=False)
        assert a.ndim == 2 and a.shape[1] == self.n_channels, (
            f"FeatStatsAccumulator expected [T, {self.n_channels}], got {a.shape}"
        )
        self.sum += a.sum(axis=0)
        self.sum_sq += np.square(a).sum(axis=0)
        self.count += int(a.shape[0])

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            return (
                np.zeros(self.n_channels, dtype=np.float64),
                np.ones(self.n_channels, dtype=np.float64),
            )
        mean = self.sum / self.count
        var = self.sum_sq / self.count - mean * mean
        var = np.maximum(var, 1e-8)
        return mean, np.sqrt(var)


def _per_song_znorm(x: np.ndarray) -> np.ndarray:
    """Per-song zero-mean / unit-std along the time axis."""
    if x.ndim == 1:
        x = x[:, None]
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _detect_sections(y: np.ndarray, sr: int, hop_length: int, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect song structure boundaries and return per-frame position + boundary flag.

    Uses chroma self-similarity + recurrence matrix to find section boundaries,
    then assigns each frame a position within its section (0=start, 1=end) and
    a binary flag for frames near section boundaries.

    Returns:
        section_positions: [n_frames] float32 in [0, 1]
        near_boundary: [n_frames] float32 (1.0 within ~1 beat of boundary)
    """
    try:
        # Chroma features at a coarser hop for efficiency.
        chroma_hop = hop_length * 4
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=chroma_hop)
        if chroma.shape[1] < 2:
            section_positions = np.zeros(n_frames, dtype=np.float32)
            near_boundary = np.zeros(n_frames, dtype=np.float32)
            return section_positions, near_boundary

        # Self-similarity
        S = librosa.segment.recurrence_matrix(chroma, mode='affinity', sym=True)
        # Find boundaries via spectral clustering of the affinity matrix.
        try:
            bounds_frames = librosa.segment.agglomerative(chroma, k=None)
        except Exception:
            # Fallback: use a simpler boundary detection.
            bounds_frames = _simple_boundary_detect(S, min_segment_frames=8)

        # Convert boundary frame indices (in chroma_hop) to sample times.
        bound_times = [0.0]
        for bf in sorted(bounds_frames):
            if bf <= 0 or bf >= chroma.shape[1]:
                continue
            t = (bf * chroma_hop) / sr
            if t > bound_times[-1] + 2.0:  # Minimum 2-second segment.
                bound_times.append(t)

        total_duration = (n_frames * hop_length) / sr
        if not bound_times or bound_times[-1] < total_duration * 0.9:
            bound_times.append(total_duration)

        # Build per-frame section_position.
        section_positions = np.zeros(n_frames, dtype=np.float32)
        near_boundary = np.zeros(n_frames, dtype=np.float32)
        boundary_window = 60.0 / 120.0  # ~1 beat at 120 BPM

        for i in range(len(bound_times) - 1):
            t_start = bound_times[i]
            t_end = bound_times[i + 1]
            if t_end <= t_start:
                continue
            duration = t_end - t_start
            f_start = int(t_start * FRAMES_PER_SECOND)
            f_end = int(t_end * FRAMES_PER_SECOND)
            f_start = max(0, min(f_start, n_frames))
            f_end = max(f_start + 1, min(f_end, n_frames))

            for f in range(f_start, f_end):
                t = f / FRAMES_PER_SECOND
                section_positions[f] = float(np.clip((t - t_start) / duration, 0.0, 1.0))
                # Near boundary if within window of section start/end.
                if (t - t_start) < boundary_window or (t_end - t) < boundary_window:
                    near_boundary[f] = 1.0

        logger.debug(
            f"Detected {len(bound_times)-1} sections "
            f"(boundaries at {[round(t, 1) for t in bound_times[:6]]}...)"
        )
        return section_positions, near_boundary

    except Exception as e:
        logger.warning(f"Section detection failed: {e}")
        return (
            np.zeros(n_frames, dtype=np.float32),
            np.zeros(n_frames, dtype=np.float32),
        )


def _simple_boundary_detect(S: np.ndarray, min_segment_frames: int = 8) -> np.ndarray:
    """Simple boundary detection from a recurrence matrix's novelty curve."""
    if S.shape[0] < 2 * min_segment_frames:
        return np.array([], dtype=int)
    # Novelty: check diagonal of S for drops.
    n = S.shape[0]
    novelty = np.zeros(n)
    for i in range(1, n - 1):
        # Correlation drop in a small window along the diagonal.
        w = min(4, i, n - i - 1)
        if w < 1:
            continue
        before = np.mean(np.diag(S, 0)[max(0, i-w):i])
        after = np.mean(np.diag(S, 0)[i+1:i+1+w])
        novelty[i] = before - after
    novelty = np.maximum(novelty, 0)
    threshold = np.percentile(novelty, 90)
    peaks = np.flatnonzero((novelty > threshold) & (novelty > np.roll(novelty, 1)) & (novelty > np.roll(novelty, -1)))
    # Filter by minimum segment length.
    filtered = []
    last = -min_segment_frames
    for p in peaks:
        if p - last >= min_segment_frames:
            filtered.append(int(p))
            last = p
    return np.array(filtered, dtype=int)


def extract_audio_features(
    audio_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract the v8 feats tensor [T, 88] and structural features from an audio file.

    Layout:
        feats[:, 0:80]   mel (fixed dB → [0,1])
        feats[:, 80:81]  librosa.onset.onset_strength, per-song z-normed
        feats[:, 81:88]  librosa.feature.spectral_contrast (7 bands), per-song z-normed

    Returns:
        (feats, section_positions, near_boundary) or None on failure.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        logger.warning(f"Failed to load audio {audio_path}: {e}")
        return None

    if len(y) < SAMPLE_RATE:  # Less than 1 second
        logger.warning(f"Audio too short: {audio_path}")
        return None

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    np.clip(mel_db, DB_MIN, DB_MAX, out=mel_db)
    mel_scaled = (mel_db - DB_MIN) / DB_RANGE  # -> [0, 1]
    mel_t = mel_scaled.T.astype(np.float32)  # [T, 80]
    n_frames = mel_t.shape[0]

    # Onset strength envelope at the same hop as the mel grid.
    try:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=HOP_LENGTH,
        )
    except Exception as e:
        logger.warning(f"onset_strength failed for {audio_path}: {e}")
        onset_env = np.zeros(n_frames, dtype=np.float32)
    onset_env = np.asarray(onset_env, dtype=np.float32)
    onset_env = _align_length(onset_env, n_frames)

    # Spectral contrast — 7 bands by default.
    try:
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT,
        )
    except Exception as e:
        logger.warning(f"spectral_contrast failed for {audio_path}: {e}")
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
    contrast_t = contrast.T  # [T_c, 7]
    contrast_t = _align_length(contrast_t, n_frames)

    onset_z = _per_song_znorm(onset_env)        # [T, 1]
    contrast_z = _per_song_znorm(contrast_t)    # [T, 7]

    feats = np.concatenate([mel_t, onset_z, contrast_z], axis=1)
    assert feats.shape == (n_frames, N_FEAT_CHANNELS), (
        f"feats shape {feats.shape} != expected ({n_frames}, {N_FEAT_CHANNELS})"
    )

    # v8: structural features.
    section_positions, near_boundary = _detect_sections(y, sr, HOP_LENGTH, n_frames)

    return (
        feats.astype(np.float16),
        section_positions.astype(np.float32),
        near_boundary.astype(np.float32),
    )


def _align_length(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Pad / truncate `arr` along axis 0 to match `n_frames`."""
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


def notes_to_frame_labels(notes, n_frames: int):
    """
    Convert note list to frame-aligned label, duration, and per-arrow arrays.

    Returns three aligned tensors:
      - `labels` (arrow-agnostic note type)
      - `durations` (seconds at hold_start frames)
      - `arrow_labels` (per-arrow binary)
    """
    labels = np.zeros(n_frames, dtype=np.uint8)
    durations = np.zeros(n_frames, dtype=np.float32)
    arrow_labels = np.zeros((n_frames, 4), dtype=np.uint8)

    hold_heads_by_col = {}
    hold_tails_by_col = {}
    for note in notes:
        frame = int(round(note.time * FRAMES_PER_SECOND))
        if frame < 0 or frame >= n_frames:
            continue
        if note.note_type == 'hold_head':
            hold_heads_by_col.setdefault(note.arrow, []).append((note.time, frame))
        elif note.note_type == 'hold_tail':
            hold_tails_by_col.setdefault(note.arrow, []).append((note.time, frame))

    hold_durations_at_frame = {}
    for col, heads in hold_heads_by_col.items():
        tails = hold_tails_by_col.get(col, [])
        heads.sort()
        tails.sort()
        ti = 0
        for h_time, h_frame in heads:
            while ti < len(tails) and tails[ti][0] <= h_time:
                ti += 1
            if ti < len(tails):
                dur = tails[ti][0] - h_time
                if dur > 0:
                    hold_durations_at_frame[h_frame] = max(
                        hold_durations_at_frame.get(h_frame, 0.0), dur
                    )

    for note in notes:
        frame = int(round(note.time * FRAMES_PER_SECOND))
        if frame < 0 or frame >= n_frames:
            continue
        # note.arrow is integer 0-3 (L/D/U/R from n.column), not a string.
        arrow_idx = note.arrow
        if arrow_idx < 0 or arrow_idx > 3:
            continue
        if note.note_type in ('tap', 'hold_head'):
            arrow_labels[frame, arrow_idx] = 1
        # All tap/hold_head frames get classed; jumps = multiple arrows in same frame.
        if note.note_type == 'tap':
            labels[frame] = LABEL_TAP
        elif note.note_type == 'hold_head':
            labels[frame] = LABEL_HOLD_START

    # Post-pass: frames with multiple arrows → jump.
    multi = arrow_labels.sum(axis=1) >= 2
    labels[multi] = LABEL_JUMP

    # Write durations at hold_start frames.
    for hf, dur in hold_durations_at_frame.items():
        durations[hf] = dur

    return labels, durations, arrow_labels


def process_chart_directory(
    chart_dir: Path,
    output_dir: Path,
    song_index: int,
    stats: FeatStatsAccumulator,
) -> list:
    """Process a single chart directory into one .npz file per song.

    All difficulties for the same song share one feats tensor.
    """
    sm_files = sorted(chart_dir.glob("*.sm"))
    audio_files = []
    for ext in ('.mp3', '.ogg', '.wav', '.flac', '.MP3', '.OGG', '.WAV', '.FLAC'):
        audio_files.extend(sorted(chart_dir.glob(f"*{ext}")))
    if not audio_files:
        logger.warning(f"No audio file in {chart_dir}")
        return []
    if not sm_files:
        logger.warning(f"No .sm files in {chart_dir}")
        return []

    audio_path = str(audio_files[0])
    result = extract_audio_features(audio_path)
    if result is None:
        return []
    feats, section_positions, near_boundary = result

    song_filename = f"song_{song_index:04d}.npz"
    arrays = {}
    entries = []
    primary_bpm = 120.0

    for sm_path in sm_files:
        try:
            chart_data = parse_sm_file(str(sm_path))
        except Exception as e:
            logger.warning(f"Failed to parse {sm_path}: {e}")
            continue

        if not chart_data.charts:
            continue
        primary_bpm = chart_data.primary_bpm

        for chart in chart_data.charts:
            diff_name = chart.difficulty or 'medium'
            diff_lower = diff_name.lower()
            difficulty_id = chart.difficulty_id
            notes = chart.notes
            if not notes:
                continue
            n_frames = feats.shape[0]
            labels, durations, arrow_labels_arr = notes_to_frame_labels(notes, n_frames)
            note_frames = int((labels > 0).sum())
            if note_frames == 0:
                continue

            labels_key = f'labels_{diff_lower}'
            durations_key = f'durations_{diff_lower}'
            arrow_labels_key = f'arrow_labels_{diff_lower}'
            arrays[labels_key] = labels
            arrays[durations_key] = durations
            arrays[arrow_labels_key] = arrow_labels_arr

            entries.append({
                'filename': song_filename,
                'song_title': chart_data.title or chart_dir.name,
                'artist': chart_data.artist or '',
                'difficulty': diff_name,
                'difficulty_id': difficulty_id,
                'labels_key': labels_key,
                'durations_key': durations_key,
                'arrow_labels_key': arrow_labels_key,
                'n_frames': n_frames,
                'n_notes': int(note_frames),
                'tempo': round(primary_bpm, 1),
                'duration': round(n_frames / FRAMES_PER_SECOND, 2),
            })

    if not arrays:
        return []

    # v8: include structural features in the npz.
    arrays['section_positions'] = section_positions
    arrays['near_boundary'] = near_boundary

    np.savez_compressed(
        output_dir / song_filename,
        feats=feats,
        **arrays,
    )
    stats.update(feats.astype(np.float32))

    return entries


def main():
    parser = argparse.ArgumentParser(description='Preprocess DDR charts for ML training')
    parser.add_argument('--charts-dir', type=str, default='charts',
                        help='Directory containing chart subdirectories')
    parser.add_argument('--output-dir', type=str, default='ml/training_data',
                        help='Output directory for preprocessed data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N charts (for testing)')
    args = parser.parse_args()

    charts_dir = Path(args.charts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not charts_dir.exists():
        logger.error(f"Charts directory not found: {charts_dir}")
        sys.exit(1)

    chart_dirs = sorted([d for d in charts_dir.iterdir() if d.is_dir()])
    if args.limit:
        chart_dirs = chart_dirs[:args.limit]

    logger.info(f"Processing {len(chart_dirs)} chart directories...")

    entries: list = []
    stats = FeatStatsAccumulator()
    processed = 0
    errors = 0

    for idx, chart_dir in enumerate(chart_dirs):
        try:
            song_entries = process_chart_directory(chart_dir, output_dir, idx, stats)
            entries.extend(song_entries)
            if song_entries:
                processed += 1
        except Exception as e:
            logger.warning(f"Error processing {chart_dir.name}: {e}")
            errors += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"Progress: {idx + 1}/{len(chart_dirs)} "
                          f"({processed} songs, {len(entries)} charts)")

    feat_mean, feat_std = stats.finalize()
    logger.info(
        f"Per-channel feat stats finalized over {stats.count:,} frames "
        f"(mel mean range [{feat_mean[:80].min():.4f}, {feat_mean[:80].max():.4f}])"
    )

    manifest_path = output_dir / 'manifest.json'
    manifest = {
        'format_version': FORMAT_VERSION,
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
        'n_in_channels': N_FEAT_CHANNELS,
        'n_mels': N_MELS,
        'db_min': DB_MIN,
        'db_max': DB_MAX,
        'sample_rate': SAMPLE_RATE,
        'hop_length': HOP_LENGTH,
        'entries': entries,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nDone!")
    logger.info(f"  Songs processed: {processed}")
    logger.info(f"  Charts generated: {len(entries)}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Manifest: {manifest_path}")

    from collections import Counter
    diff_counts = Counter(e['difficulty'] for e in entries)
    logger.info(f"  Difficulty distribution: {dict(diff_counts)}")


if __name__ == '__main__':
    main()