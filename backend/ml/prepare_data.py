"""
Preprocessing script: converts .sm charts + audio into training data.

Run this locally before training on Colab:
    python -m ml.prepare_data --charts-dir charts/ --output-dir ml/training_data/

Output (format_version=7):
    training_data/
    ├── manifest.json    # {format_version, feat_mean[88], feat_std[88], entries: [...]}
    ├── song_0001.npz    # one file per song: feats[T,88] + per-chart label arrays
    ├── song_0002.npz
    └── ...

`feats` channels (88 total, all aligned at ~100.23 fps):
    [0..79]  — fixed-dB scaled mel (clipped to [DB_MIN, DB_MAX], mapped to [0,1])
    [80]     — librosa.onset.onset_strength (per-song z-normed)
    [81..87] — librosa.feature.spectral_contrast (7 bands; per-song z-normed)

Per-chart label arrays still saved under `labels_<diff>`, `durations_<diff>`,
`arrow_labels_<diff>` so style_profiles.py can derive per-chart stats from
the same npz.
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

# v7: hybrid plan. Replaces the per-arrow / per-type targets with a smaller
# 88-channel `feats` tensor (mel + onset_strength + spec_contrast) plus the
# original per-chart label arrays (still needed downstream for style profiles
# and to derive onset/sustain/intensity targets in dataset.py).
FORMAT_VERSION = 7

# Audio processing constants
SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 220  # ~100 fps (22050 / 220 = 100.23 fps)
N_FFT = 2048
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH  # ~100.23

# Channel layout for the v7 feats tensor.
N_ONSET_STRENGTH = 1
N_SPECTRAL_CONTRAST = 7
N_FEAT_CHANNELS = N_MELS + N_ONSET_STRENGTH + N_SPECTRAL_CONTRAST  # 88

# Fixed dB scaling for mel storage. Same recipe as v5 — the only change is
# that mel is now a sub-block inside `feats[:, :80]`.
DB_MIN = -80.0
DB_MAX = 20.0
DB_RANGE = DB_MAX - DB_MIN

# Label encoding (arrow-agnostic note types) — kept so style profile code can
# read them without re-deriving from arrow_labels.
LABEL_NONE = 0
LABEL_TAP = 1
LABEL_JUMP = 2         # 2+ simultaneous arrows
LABEL_HOLD_START = 3


class FeatStatsAccumulator:
    """Online float64 per-channel mean/std accumulator over the 88 feats channels.

    Updated once per saved song so skipped songs (no charts, too short, etc.)
    don't influence the corpus stats.
    """

    def __init__(self, n_channels: int = N_FEAT_CHANNELS) -> None:
        self.n_channels = int(n_channels)
        self.sum = np.zeros(self.n_channels, dtype=np.float64)
        self.sum_sq = np.zeros(self.n_channels, dtype=np.float64)
        self.count = 0  # number of frames seen (per-channel count is identical)

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
    """Per-song zero-mean / unit-std along the time axis. Operates per channel.

    Side channels (onset_strength, spectral_contrast) live on different scales
    than mel and across different songs — z-norming per song before the
    corpus-wide whitening keeps the input distribution stable across the
    library, then `dataset.py` applies the global per-channel stats from the
    manifest to give the model a true zero-mean unit-std input.
    """
    if x.ndim == 1:
        x = x[:, None]
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def extract_audio_features(
    audio_path: str,
) -> Optional[np.ndarray]:
    """
    Extract the v7 feats tensor [T, 88] from an audio file.

    Layout:
        feats[:, 0:80]   mel (fixed dB → [0,1], not z-normed yet — global
                         whitening happens in dataset.py via manifest stats)
        feats[:, 80:81]  librosa.onset.onset_strength, per-song z-normed
        feats[:, 81:88]  librosa.feature.spectral_contrast (7 bands),
                         per-song z-normed

    Returns:
        feats: [T, 88] float16 (mel sub-block in [0,1], side channels z-normed)
        or None on failure.
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

    # Spectral contrast — 7 bands by default (n_bands=6 -> 7 channels).
    try:
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT,
        )
    except Exception as e:
        logger.warning(f"spectral_contrast failed for {audio_path}: {e}")
        contrast = np.zeros((N_SPECTRAL_CONTRAST, n_frames), dtype=np.float32)
    contrast = np.asarray(contrast, dtype=np.float32)
    if contrast.shape[0] != N_SPECTRAL_CONTRAST:
        # Pad/truncate band axis defensively (older librosa versions may
        # return a different default count).
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
    return feats.astype(np.float16)


def _align_length(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Pad / truncate `arr` along axis 0 to match `n_frames`.

    librosa side features can come back ±1 frame off from the mel grid because
    of internal centering / framing differences; we just align to the mel
    length so the channel concat works cleanly.
    """
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

    Hold durations come from greedy hold_head → hold_tail pairing on the same
    column. Stored at the hold_start frame.

    `arrow_labels[t, a] = 1` iff a tap or hold_head landed on arrow `a` at
    frame `t`. Indices are 0=L, 1=D, 2=U, 3=R.

    Args:
        notes: List of Note objects from sm_parser
        n_frames: Total number of feats frames

    Returns:
        labels: [n_frames] uint8, values in {0,1,2,3}
        durations: [n_frames] float32, seconds at hold_start frames, 0 elsewhere
        arrow_labels: [n_frames, 4] uint8, per-arrow onset indicator
    """
    labels = np.zeros(n_frames, dtype=np.uint8)
    durations = np.zeros(n_frames, dtype=np.float32)
    tap_counts = np.zeros(n_frames, dtype=np.uint8)
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
                    ti += 1

    for note in notes:
        frame = int(round(note.time * FRAMES_PER_SECOND))
        if frame < 0 or frame >= n_frames:
            continue

        if note.note_type == 'tap':
            tap_counts[frame] += 1
            if labels[frame] == LABEL_NONE:
                labels[frame] = LABEL_TAP
            if 0 <= note.arrow < 4:
                arrow_labels[frame, note.arrow] = 1
        elif note.note_type == 'hold_head':
            labels[frame] = LABEL_HOLD_START
            if frame in hold_durations_at_frame:
                durations[frame] = hold_durations_at_frame[frame]
            if 0 <= note.arrow < 4:
                arrow_labels[frame, note.arrow] = 1

    jump_mask = tap_counts >= 2
    labels[jump_mask & (labels == LABEL_TAP)] = LABEL_JUMP

    return labels, durations, arrow_labels


def find_audio_file(chart_dir: Path) -> Optional[Path]:
    for ext in ['.ogg', '.mp3', '.wav', '.flac']:
        candidates = list(chart_dir.glob(f'*{ext}'))
        if candidates:
            return candidates[0]
    return None


def process_chart_directory(
    chart_dir: Path,
    output_dir: Path,
    index: int,
    stats: FeatStatsAccumulator,
) -> list:
    """
    Process a single chart directory. Writes one song-level npz containing
    `feats[T,88]` plus per-chart `labels_<diff>`, `durations_<diff>`,
    `arrow_labels_<diff>` arrays (used downstream by dataset.py to derive
    onset/sustain/intensity targets and by style_profiles.py for stats).
    """
    sm_files = list(chart_dir.glob('*.sm'))
    if not sm_files:
        return []

    sm_file = sm_files[0]
    audio_file = find_audio_file(chart_dir)

    if audio_file is None:
        logger.debug(f"No audio found in {chart_dir.name}")
        return []

    sm = parse_sm_file(str(sm_file))
    if sm is None or not sm.charts:
        return []

    feats = extract_audio_features(str(audio_file))
    if feats is None:
        return []

    n_frames = feats.shape[0]
    primary_bpm = sm.primary_bpm

    arrays: dict = {}
    entries: list = []
    song_filename = f"song_{index:04d}.npz"

    for chart in sm.charts:
        if not chart.notes:
            continue

        labels, durations, arrow_labels = notes_to_frame_labels(chart.notes, n_frames)

        note_frames = (labels > 0).sum()
        if note_frames < 10:
            continue

        diff_base = chart.difficulty.lower().replace(' ', '_')
        diff_name = diff_base
        suffix = 2
        while f'labels_{diff_name}' in arrays:
            diff_name = f'{diff_base}_{suffix}'
            suffix += 1
        labels_key = f'labels_{diff_name}'
        durations_key = f'durations_{diff_name}'
        arrows_key = f'arrow_labels_{diff_name}'
        arrays[labels_key] = labels
        arrays[durations_key] = durations
        arrays[arrows_key] = arrow_labels

        entries.append({
            'filename': song_filename,
            'labels_key': labels_key,
            'durations_key': durations_key,
            'arrow_labels_key': arrows_key,
            'song_title': sm.title,
            'artist': sm.artist,
            'difficulty': chart.difficulty,
            'difficulty_id': chart.difficulty_id,
            'meter': chart.meter,
            'n_frames': n_frames,
            'n_notes': int(note_frames),
            'tempo': round(primary_bpm, 1),
            'duration': round(n_frames / FRAMES_PER_SECOND, 2),
        })

    if not arrays:
        return []

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
