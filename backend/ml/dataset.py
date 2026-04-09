"""
PyTorch Dataset for step chart training data.

Loads preprocessed mel spectrograms and frame-aligned labels,
returns random chunks for training.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Default fallback steps/sec per difficulty_id, used at INFERENCE time only
# when the caller does not provide an explicit target density.
# Order: 0=beginner, 1=easy, 2=medium, 3=hard, 4=challenge.
# These mirror the (min_density + max_density)/2 of the rule-based presets in
# modules/step_generator/difficulty.py and are NOT a training target — at train
# time density is computed per-chunk from the labels (see __getitem__).
DEFAULT_DENSITY_BY_ID = torch.tensor(
    [0.75, 1.25, 1.80, 3.10, 4.75], dtype=torch.float32
)

# Normalization for the density conditioning input fed into the model.
# Chosen so typical training values land roughly in [-2, 2]. The model is
# robust to these being approximate (it's just an input feature scale).
DENSITY_MEAN = 2.0  # steps/sec
DENSITY_STD = 1.5

# Frames per second of the mel grid (must match prepare_data.FRAMES_PER_SECOND).
FRAMES_PER_SECOND = 22050 / 220  # ~100.23

# Difficulty id → human-readable name (for logging).
DIFFICULTY_ID_TO_NAME = {
    0: 'beginner', 1: 'easy', 2: 'medium', 3: 'hard', 4: 'challenge',
}


class StepChartDataset(Dataset):
    """
    Dataset of (mel_spectrogram_chunk, difficulty, label_chunk) triples.

    Each item is a fixed-length chunk from a song/difficulty pair.
    During training, chunks are randomly offset for augmentation.
    During validation, chunks are taken from fixed positions.
    """

    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        chunk_frames: int = 500,
        is_train: bool = True,
        n_mels: int = 80,
    ):
        """
        Args:
            data_dir: Directory containing preprocessed .npz files
            manifest_path: Path to manifest.json listing all examples
            chunk_frames: Number of frames per training chunk (500 = 5s at 100fps)
            is_train: If True, use random chunking; else use sequential chunks
            n_mels: Number of mel bins (for validation)
        """
        self.data_dir = Path(data_dir)
        self.chunk_frames = chunk_frames
        self.is_train = is_train
        self.n_mels = n_mels

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Filter out entries shorter than chunk_frames
        self.entries = [
            e for e in self.manifest
            if e['n_frames'] >= chunk_frames
        ]

        logger.info(f"Loaded {len(self.entries)} examples "
                     f"({len(self.manifest) - len(self.entries)} skipped as too short)")

        # For validation, pre-compute chunk positions
        if not is_train:
            self._val_chunks = []
            for idx, entry in enumerate(self.entries):
                n_frames = entry['n_frames']
                # Take non-overlapping chunks
                for start in range(0, n_frames - chunk_frames + 1, chunk_frames):
                    self._val_chunks.append((idx, start))

    def __len__(self):
        if self.is_train:
            return len(self.entries)
        return len(self._val_chunks)

    def __getitem__(self, idx):
        if self.is_train:
            entry = self.entries[idx]
            n_frames = entry['n_frames']
            # Random start position
            max_start = n_frames - self.chunk_frames
            start = np.random.randint(0, max_start + 1)
        else:
            entry_idx, start = self._val_chunks[idx]
            entry = self.entries[entry_idx]

        # Load data
        data = np.load(self.data_dir / entry['filename'])
        mel = data['mel']          # [T, n_mels] float16
        labels = data['labels']    # [T, 4] uint8

        # Extract chunk
        end = start + self.chunk_frames
        mel_chunk = mel[start:end].astype(np.float32)
        labels_chunk = labels[start:end].astype(np.int64)
        difficulty = entry['difficulty_id']

        # Per-chunk step density (steps/sec): a "step" is any frame where at
        # least one arrow is a tap (class 1) or hold_start (class 2). This is
        # the conditioning signal the model learns to follow — quiet intro
        # chunks legitimately get density=0, dense choruses get high values.
        step_frames = int(((labels_chunk == 1) | (labels_chunk == 2)).any(axis=1).sum())
        chunk_seconds = self.chunk_frames / FRAMES_PER_SECOND
        density = step_frames / chunk_seconds
        density_norm = (density - DENSITY_MEAN) / DENSITY_STD

        return (
            torch.from_numpy(mel_chunk),
            torch.tensor(difficulty, dtype=torch.long),
            torch.tensor(density_norm, dtype=torch.float32),
            torch.from_numpy(labels_chunk),
        )


def compute_class_weights(manifest_path: str, data_dir: str, n_classes: int = 4) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the full dataset.

    Returns a tensor of shape [n_classes] for use with FocalLoss alpha.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    data_dir = Path(data_dir)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Sample a subset for speed if dataset is large
    sample = manifest[:200] if len(manifest) > 200 else manifest

    for entry in sample:
        data = np.load(data_dir / entry['filename'])
        labels = data['labels']  # [T, 4]
        for c in range(n_classes):
            counts[c] += (labels == c).sum()

    # Inverse frequency, normalized
    total = counts.sum()
    weights = total / (n_classes * counts + 1)
    weights = weights / weights.sum() * n_classes

    logger.info(f"Class counts (sampled): {counts}")
    logger.info(f"Class weights: {weights}")

    return torch.from_numpy(weights.astype(np.float32))


def make_balanced_difficulty_sampler(
    train_subset, full_dataset
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that yields each difficulty class with
    equal probability, even when the underlying manifest is imbalanced.

    This is the per-difficulty training balance fix: without it, FiLM gets
    most of its gradient signal from whichever difficulty dominates the
    manifest and never learns to specialize per difficulty.

    Args:
        train_subset: torch.utils.data.Subset wrapping `full_dataset`
            (e.g. produced by `random_split`).
        full_dataset: the underlying StepChartDataset with `.entries`.

    Returns:
        A WeightedRandomSampler sized to len(train_subset).
    """
    if isinstance(train_subset, Subset):
        indices = train_subset.indices
    else:
        indices = list(range(len(train_subset)))

    # Per-class frequency over the train indices
    diff_ids = np.array(
        [full_dataset.entries[i]['difficulty_id'] for i in indices],
        dtype=np.int64,
    )
    counts = np.bincount(diff_ids, minlength=5).astype(np.float64)
    # Avoid div-by-zero for absent classes
    inv = np.where(counts > 0, 1.0 / counts, 0.0)
    sample_weights = inv[diff_ids]

    logger.info(f"Difficulty counts in train split: {counts.astype(int).tolist()}")
    logger.info(f"Per-class sampling weights: {inv.tolist()}")

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(indices),
        replacement=True,
    )


def compute_default_density_per_difficulty(
    manifest_path: str,
    data_dir: str,
    n_difficulties: int = 5,
    frames_per_second: float = FRAMES_PER_SECOND,
) -> torch.Tensor:
    """
    Compute the empirical mean steps/sec per difficulty_id from the manifest.

    Used at inference time as a sensible default when the caller does not
    supply an explicit density. Save the result into the checkpoint so the
    inference path can use data-driven defaults instead of the hardcoded
    DEFAULT_DENSITY_BY_ID fallback.
    """
    sums = np.zeros(n_difficulties, dtype=np.float64)
    secs = np.zeros(n_difficulties, dtype=np.float64)
    data_dir = Path(data_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest:
        d = int(entry['difficulty_id'])
        labels = np.load(data_dir / entry['filename'])['labels']  # [T, 4]
        step_frames = int(((labels == 1) | (labels == 2)).any(axis=1).sum())
        sums[d] += step_frames
        secs[d] += labels.shape[0] / frames_per_second
    means = sums / np.maximum(secs, 1.0)
    logger.info(f"Empirical mean density per difficulty: {means.tolist()}")
    return torch.from_numpy(means.astype(np.float32))
