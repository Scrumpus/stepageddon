"""
Preprocessing script: converts .sm charts + audio into training data.

Run this locally before training on Colab:
    python -m ml.prepare_data --charts-dir charts/ --output-dir ml/training_data/

Output:
    training_data/
    ├── manifest.json          # Index of all examples
    ├── song_001_beginner.npz  # mel spectrogram + labels per chart
    ├── song_001_easy.npz
    └── ...
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
except ImportError:
    print("librosa is required: pip install librosa")
    sys.exit(1)

from ml.sm_parser import parse_sm_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Audio processing constants
SAMPLE_RATE = 22050
N_MELS = 80
HOP_LENGTH = 220  # ~100 fps (22050 / 220 = 100.23 fps)
N_FFT = 2048
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH  # ~100.23

# Label encoding
LABEL_NONE = 0
LABEL_TAP = 1
LABEL_HOLD_START = 2
LABEL_HOLD_END = 3


def extract_mel_spectrogram(audio_path: str) -> Optional[np.ndarray]:
    """
    Extract mel spectrogram from audio file.

    Returns:
        [T, N_MELS] float16 array, or None on failure.
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
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db.T.astype(np.float16)  # [T, N_MELS]


def notes_to_frame_labels(notes, n_frames: int) -> np.ndarray:
    """
    Convert note list to frame-aligned label array.

    Args:
        notes: List of Note objects from sm_parser
        n_frames: Total number of mel frames

    Returns:
        [n_frames, 4] uint8 array with values 0-3
    """
    labels = np.zeros((n_frames, 4), dtype=np.uint8)

    for note in notes:
        frame = int(round(note.time * FRAMES_PER_SECOND))
        if frame < 0 or frame >= n_frames:
            continue

        arrow = note.arrow
        if note.note_type == 'tap':
            labels[frame, arrow] = LABEL_TAP
        elif note.note_type == 'hold_head':
            labels[frame, arrow] = LABEL_HOLD_START
        elif note.note_type == 'hold_tail':
            labels[frame, arrow] = LABEL_HOLD_END

    return labels


def find_audio_file(chart_dir: Path) -> Optional[Path]:
    """Find the audio file in a chart directory."""
    for ext in ['.ogg', '.mp3', '.wav', '.flac']:
        candidates = list(chart_dir.glob(f'*{ext}'))
        if candidates:
            return candidates[0]
    return None


def process_chart_directory(chart_dir: Path, output_dir: Path, index: int) -> list:
    """
    Process a single chart directory (one song).

    Returns list of manifest entries.
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

    # Extract mel spectrogram
    mel = extract_mel_spectrogram(str(audio_file))
    if mel is None:
        return []

    n_frames = mel.shape[0]
    primary_bpm = sm.primary_bpm
    entries = []

    for chart in sm.charts:
        if not chart.notes:
            continue

        labels = notes_to_frame_labels(chart.notes, n_frames)

        # Skip charts with very few notes
        note_frames = (labels > 0).any(axis=1).sum()
        if note_frames < 10:
            continue

        # Generate filename
        diff_name = chart.difficulty.lower().replace(' ', '_')
        filename = f"song_{index:04d}_{diff_name}.npz"

        # Save
        np.savez_compressed(
            output_dir / filename,
            mel=mel,
            labels=labels,
        )

        entries.append({
            'filename': filename,
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

    manifest = []
    processed = 0
    errors = 0

    for idx, chart_dir in enumerate(chart_dirs):
        try:
            entries = process_chart_directory(chart_dir, output_dir, idx)
            manifest.extend(entries)
            if entries:
                processed += 1
        except Exception as e:
            logger.warning(f"Error processing {chart_dir.name}: {e}")
            errors += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"Progress: {idx + 1}/{len(chart_dirs)} "
                         f"({processed} songs, {len(manifest)} charts)")

    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nDone!")
    logger.info(f"  Songs processed: {processed}")
    logger.info(f"  Charts generated: {len(manifest)}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Manifest: {manifest_path}")

    # Print difficulty distribution
    from collections import Counter
    diff_counts = Counter(e['difficulty'] for e in manifest)
    logger.info(f"  Difficulty distribution: {dict(diff_counts)}")


if __name__ == '__main__':
    main()
