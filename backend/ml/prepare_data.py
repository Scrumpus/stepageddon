"""
Preprocessing script: converts .sm charts + audio into training data.

Run this locally before training on Colab:
    python -m ml.prepare_data --charts-dir charts/ --output-dir ml/training_data/

Output (format_version=2):
    training_data/
    ├── manifest.json    # {format_version, mel_mean, mel_std, entries: [...]}
    ├── song_0001.npz    # one file per song: mel + one labels_<diff> per chart
    ├── song_0002.npz
    └── ...

The mel is stored once per song in fixed scaled-dB [0, 1] space (NOT per-song
min-max), so loudness is comparable across songs. Global whitening stats over
the corpus are written into the manifest and applied at load time by the
dataset, which means changing normalization later only requires retraining,
not re-running prepare_data.
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

from services.sm_parser import parse_sm_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# On-disk format version. Bump on any breaking change to npz layout / manifest
# schema. Dataset loaders error loudly on mismatch.
# v4: adds song-level `beats` array (shape [T] uint8) for the auxiliary
# beat-prediction head.
FORMAT_VERSION = 4

# Audio processing constants
SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 220  # ~100 fps (22050 / 220 = 100.23 fps)
N_FFT = 2048
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH  # ~100.23

# Fixed dB scaling for mel storage. We compute log-mel against an absolute
# reference (ref=1.0) and clip to [DB_MIN, DB_MAX] dB, then map to [0, 1] for
# float16 storage. This preserves cross-song loudness, unlike per-file min-max.
# DB_RANGE is exposed so the dataset's gain-jitter augmentation can shift in
# matching units.
DB_MIN = -80.0
DB_MAX = 20.0
DB_RANGE = DB_MAX - DB_MIN  # 100 dB total span

# Label encoding (arrow-agnostic note types)
LABEL_NONE = 0
LABEL_TAP = 1
LABEL_JUMP = 2         # 2+ simultaneous arrows
LABEL_HOLD_START = 3
N_NOTE_TYPES = 4        # total classes for type head (including NONE)


class MelStatsAccumulator:
    """Online float64 accumulator for global mean/std over scaled-dB mel values.

    Used at preprocessing time to compute corpus-wide whitening stats without
    holding the entire dataset in memory. Updated once per saved song so that
    skipped songs (no charts, too short, etc.) don't influence the stats.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0

    def update(self, arr: np.ndarray) -> None:
        a = arr.astype(np.float64, copy=False)
        self.sum += float(a.sum())
        self.sum_sq += float(np.square(a).sum())
        self.count += int(a.size)

    def finalize(self) -> Tuple[float, float]:
        if self.count == 0:
            return 0.0, 1.0
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 1e-8)
        return mean, math.sqrt(var)


def extract_audio_features(
    audio_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract mel spectrogram + beat track from an audio file.

    Mel: fixed scaled-dB [0, 1] space (ref=1.0, clipped to [DB_MIN, DB_MAX]).
    Beats: [T] uint8 binary array (1 at beat frames) at the same hop as mel.
    The auxiliary beat head consumes this as a per-frame BCE target.

    Returns:
        (mel: [T, N_MELS] float16, beats: [T] uint8), or None on failure.
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
    # Absolute dB (no per-file ref=np.max). top_db=None disables librosa's
    # adaptive floor; we apply our own fixed clip range below.
    mel_db = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    np.clip(mel_db, DB_MIN, DB_MAX, out=mel_db)
    mel_scaled = (mel_db - DB_MIN) / DB_RANGE  # -> [0, 1]
    mel_arr = mel_scaled.T.astype(np.float16)  # [T, N_MELS]

    # Beat track at the same hop as the mel grid so beat_frames index directly
    # into mel rows. Failures (very short / silent songs) fall back to an
    # all-zero beat track — the BCE just sees no positives there.
    n_frames = mel_arr.shape[0]
    beats = np.zeros(n_frames, dtype=np.uint8)
    try:
        _tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=HOP_LENGTH,
        )
        beat_frames = np.asarray(beat_frames, dtype=np.int64)
        beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < n_frames)]
        beats[beat_frames] = 1
    except Exception as e:
        logger.warning(f"Beat track failed for {audio_path}: {e}")

    return mel_arr, beats


# Backwards-compat shim: older callers may still import the legacy name.
def extract_mel_spectrogram(audio_path: str) -> Optional[np.ndarray]:
    feats = extract_audio_features(audio_path)
    return feats[0] if feats is not None else None


def notes_to_frame_labels(notes, n_frames: int):
    """
    Convert note list to frame-aligned label and duration arrays (arrow-agnostic).

    Collapses per-arrow information into a single note-type label per frame.
    Counts simultaneous events to distinguish singles from jumps:
        - 1 tap → tap, 2+ taps → jump
        - 1 hold_head → hold_start, 2+ hold_heads → hold_start

    Hold durations are computed by pairing hold_head → hold_tail on the same
    arrow column, then stored at the hold_start frame. The model predicts
    duration directly instead of detecting a separate hold_end event.

    Args:
        notes: List of Note objects from sm_parser
        n_frames: Total number of mel frames

    Returns:
        labels: [n_frames] uint8 array with values 0-3
        durations: [n_frames] float32 array, hold duration in seconds at
            hold_start frames, 0.0 elsewhere
    """
    labels = np.zeros(n_frames, dtype=np.uint8)
    durations = np.zeros(n_frames, dtype=np.float32)
    tap_counts = np.zeros(n_frames, dtype=np.uint8)

    # First pass: pair hold_head → hold_tail per arrow to get durations.
    # Collect hold_heads per column, then match with tails in time order.
    hold_heads_by_col = {}  # arrow -> [(time, frame), ...]
    hold_tails_by_col = {}  # arrow -> [(time, frame), ...]
    for note in notes:
        frame = int(round(note.time * FRAMES_PER_SECOND))
        if frame < 0 or frame >= n_frames:
            continue
        if note.note_type == 'hold_head':
            hold_heads_by_col.setdefault(note.arrow, []).append((note.time, frame))
        elif note.note_type == 'hold_tail':
            hold_tails_by_col.setdefault(note.arrow, []).append((note.time, frame))

    # Greedy pair: each head matches the next tail on the same column
    hold_durations_at_frame = {}  # frame -> max duration (if multiple arrows)
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
                    # If multiple hold_heads land on the same frame, keep longest
                    hold_durations_at_frame[h_frame] = max(
                        hold_durations_at_frame.get(h_frame, 0.0), dur
                    )
                    ti += 1

    # Second pass: assign labels
    for note in notes:
        frame = int(round(note.time * FRAMES_PER_SECOND))
        if frame < 0 or frame >= n_frames:
            continue

        if note.note_type == 'tap':
            tap_counts[frame] += 1
            if labels[frame] == LABEL_NONE:
                labels[frame] = LABEL_TAP
        elif note.note_type == 'hold_head':
            labels[frame] = LABEL_HOLD_START
            if frame in hold_durations_at_frame:
                durations[frame] = hold_durations_at_frame[frame]

    # Upgrade tap → jump where 2+ arrows fired simultaneously
    jump_mask = tap_counts >= 2
    labels[jump_mask & (labels == LABEL_TAP)] = LABEL_JUMP

    return labels, durations


def find_audio_file(chart_dir: Path) -> Optional[Path]:
    """Find the audio file in a chart directory."""
    for ext in ['.ogg', '.mp3', '.wav', '.flac']:
        candidates = list(chart_dir.glob(f'*{ext}'))
        if candidates:
            return candidates[0]
    return None


def process_chart_directory(
    chart_dir: Path,
    output_dir: Path,
    index: int,
    stats: MelStatsAccumulator,
) -> list:
    """
    Process a single chart directory (one song).

    Writes a single song-level npz containing the mel and one labels_<diff>
    array per chart, eliminating the per-difficulty mel duplication of v1.
    Updates `stats` with the song's mel values iff the song actually gets
    saved.

    Returns list of manifest entries (one per chart that survived filtering).
    """
    sm_files = list(chart_dir.glob('*.sm'))
    if not sm_files:
        return []

    sm_file = sm_files[0]
    audio_file = find_audio_file(chart_dir)

    if audio_file is None:
        logger.debug(f"No audio found in {chart_dir.name}")
        return []

    # Parse .sm file
    sm = parse_sm_file(str(sm_file))
    if sm is None or not sm.charts:
        return []

    # Extract mel spectrogram + beat track
    feats = extract_audio_features(str(audio_file))
    if feats is None:
        return []
    mel, beats = feats

    n_frames = mel.shape[0]
    primary_bpm = sm.primary_bpm

    # Build per-chart label arrays first; only write the song file if at
    # least one chart survives. Handle duplicate-difficulty collisions by
    # suffixing _2, _3, ... so charts don't silently overwrite each other.
    arrays: dict = {}
    entries: list = []
    song_filename = f"song_{index:04d}.npz"

    for chart in sm.charts:
        if not chart.notes:
            continue

        labels, durations = notes_to_frame_labels(chart.notes, n_frames)

        # Skip charts with very few notes
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
        arrays[labels_key] = labels
        arrays[durations_key] = durations

        entries.append({
            'filename': song_filename,
            'labels_key': labels_key,
            'durations_key': durations_key,
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
        mel=mel,
        beats=beats,
        **arrays,
    )
    # Stats over the float16 stored values, accumulated in float64.
    stats.update(mel)

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

    # Get all chart directories
    chart_dirs = sorted([d for d in charts_dir.iterdir() if d.is_dir()])
    if args.limit:
        chart_dirs = chart_dirs[:args.limit]

    logger.info(f"Processing {len(chart_dirs)} chart directories...")

    entries: list = []
    stats = MelStatsAccumulator()
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

    mel_mean, mel_std = stats.finalize()
    logger.info(f"Global mel stats: mean={mel_mean:.6f}, std={mel_std:.6f} "
                 f"(over {stats.count:,} values)")

    # Save manifest (format_version=2 wraps entries with corpus-level metadata).
    manifest_path = output_dir / 'manifest.json'
    manifest = {
        'format_version': FORMAT_VERSION,
        'mel_mean': mel_mean,
        'mel_std': mel_std,
        'db_min': DB_MIN,
        'db_max': DB_MAX,
        'sample_rate': SAMPLE_RATE,
        'hop_length': HOP_LENGTH,
        'n_mels': N_MELS,
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

    # Print difficulty distribution
    from collections import Counter
    diff_counts = Counter(e['difficulty'] for e in entries)
    logger.info(f"  Difficulty distribution: {dict(diff_counts)}")


if __name__ == '__main__':
    main()
