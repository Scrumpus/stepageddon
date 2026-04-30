"""
PyTorch Dataset for step chart training data.

Loads preprocessed mel spectrograms and frame-aligned labels,
returns random chunks for training.
"""

import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from pathlib import Path
from typing import List, Optional, Tuple
import logging

from ml.prepare_data import DB_RANGE, FORMAT_VERSION

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


def load_manifest(manifest_path: str) -> dict:
    """Load a v2 manifest, validating the format version.

    The v2 manifest is a dict containing top-level corpus stats plus the
    `entries` list (one entry per chart). v1 was a bare list — detected here
    so users get a clear "re-run prepare_data.py" message instead of a
    confusing KeyError downstream.
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    if isinstance(manifest, list):
        raise ValueError(
            f"Manifest at {manifest_path} is in legacy v1 format (a bare list). "
            f"Re-run prepare_data.py to produce format_version={FORMAT_VERSION}."
        )
    version = manifest.get('format_version')
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported manifest format_version={version} at {manifest_path}. "
            f"Expected {FORMAT_VERSION}; re-run prepare_data.py."
        )
    return manifest


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
        onset_smooth_sigma: float = 1.5,
        augment: bool = True,
        spec_time_masks: int = 2,
        spec_time_width: int = 20,
        spec_freq_masks: int = 2,
        spec_freq_width: int = 10,
        gain_jitter_db: float = 3.0,
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
        self.onset_smooth_sigma = float(onset_smooth_sigma)
        self.augment = bool(augment) and is_train
        self.spec_time_masks = spec_time_masks
        self.spec_time_width = spec_time_width
        self.spec_freq_masks = spec_freq_masks
        self.spec_freq_width = spec_freq_width
        self.gain_jitter_db = gain_jitter_db

        # Precomputed Gaussian kernel for onset label smoothing.
        if self.onset_smooth_sigma > 0:
            radius = max(1, int(round(self.onset_smooth_sigma * 3)))
            t = np.arange(-radius, radius + 1, dtype=np.float32)
            kernel = np.exp(-0.5 * (t / self.onset_smooth_sigma) ** 2)
            kernel /= kernel.max()  # keep peak at 1.0 after convolution
            self._onset_kernel = kernel
        else:
            self._onset_kernel = None

        manifest = load_manifest(manifest_path)
        self.manifest = manifest['entries']
        # Global mel whitening stats live in the manifest (v2). Applied to
        # every chunk in __getitem__ after gain jitter, so the model sees
        # zero-mean unit-variance log-mel inputs regardless of which song
        # the chunk came from.
        self.mel_mean = float(manifest['mel_mean'])
        self.mel_std = float(manifest['mel_std'])

        # Filter out entries shorter than chunk_frames
        self.entries = [
            e for e in self.manifest
            if e['n_frames'] >= chunk_frames
        ]

        logger.info(f"Loaded {len(self.entries)} examples "
                     f"({len(self.manifest) - len(self.entries)} skipped as too short)")
        logger.info(f"Mel whitening stats: mean={self.mel_mean:.4f}, std={self.mel_std:.4f}")

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

        # Load data. v3 npz files contain mel, labels_<diff>, and durations_<diff>
        # per chart inside the same song-level archive; only the requested keys
        # are decompressed by NpzFile on access.
        data = np.load(self.data_dir / entry['filename'])
        mel = data['mel']                          # [T, n_mels] float16, [0,1] scaled-dB
        labels = data[entry['labels_key']]         # [T] uint8 (0=none,1=tap,2=jump,3=hold_start)
        durations = data[entry['durations_key']]   # [T] float32, seconds at hold_start frames

        # Extract chunk
        end = start + self.chunk_frames
        mel_chunk = mel[start:end].astype(np.float32)
        labels_chunk = labels[start:end].astype(np.int64)
        durations_chunk = durations[start:end].astype(np.float32)
        difficulty = entry['difficulty_id']

        # Per-chunk step density (steps/sec): a "step" is any frame with a
        # note event (tap, jump, or hold_start).
        step_frames = int((labels_chunk >= 1).sum())
        chunk_seconds = self.chunk_frames / FRAMES_PER_SECOND
        density = step_frames / chunk_seconds
        density_norm = (density - DENSITY_MEAN) / DENSITY_STD

        # Build onset target: [T, 1] (soft during training, hard during val)
        onset_hard = (labels_chunk > 0).astype(np.float32)[:, None]  # [T, 1]
        if self.is_train and self._onset_kernel is not None:
            onset_soft = self._smooth_onset(onset_hard)
        else:
            onset_soft = onset_hard

        # Type target: [T] long with classes {0=tap, 1=jump, 2=hold_start}
        # labels_chunk - 1 maps {1,2,3} -> {0,1,2}; {0} -> -1, then to -100
        type_target = labels_chunk - 1
        type_target[type_target < 0] = -100

        # ---- Mel augmentation + whitening ----
        if self.augment and self.gain_jitter_db > 0:
            db_shift = float(np.random.uniform(-self.gain_jitter_db, self.gain_jitter_db))
            mel_chunk = mel_chunk + (db_shift / DB_RANGE)
            np.clip(mel_chunk, 0.0, 1.0, out=mel_chunk)

        mel_chunk = (mel_chunk - self.mel_mean) / self.mel_std

        if self.augment:
            mel_chunk = self._spec_augment(mel_chunk)

        return (
            torch.from_numpy(mel_chunk),
            torch.tensor(difficulty, dtype=torch.long),
            torch.tensor(density_norm, dtype=torch.float32),
            torch.from_numpy(onset_soft),
            torch.from_numpy(type_target),
            torch.from_numpy(durations_chunk),
        )

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _smooth_onset(self, onset_hard: np.ndarray) -> np.ndarray:
        """Convolve a [T, 1] hard onset map with a 1-D Gaussian along time."""
        k = self._onset_kernel
        out = np.zeros_like(onset_hard)
        out[:, 0] = np.convolve(onset_hard[:, 0], k, mode='same')
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        """SpecAugment time + freq masks. Operates on whitened mel.

        Mask value is 0.0, which equals the dataset mean in whitened space —
        this is the standard SpecAugment recipe and avoids leaking a
        distinct out-of-distribution constant into the model.
        """
        mel = mel.copy()
        T, F = mel.shape

        # Time masks
        for _ in range(self.spec_time_masks):
            w = int(np.random.randint(0, self.spec_time_width + 1))
            if w == 0 or w >= T:
                continue
            t0 = int(np.random.randint(0, T - w + 1))
            mel[t0:t0 + w, :] = 0.0

        # Freq masks
        for _ in range(self.spec_freq_masks):
            w = int(np.random.randint(0, self.spec_freq_width + 1))
            if w == 0 or w >= F:
                continue
            f0 = int(np.random.randint(0, F - w + 1))
            mel[:, f0:f0 + w] = 0.0

        return mel.astype(np.float32)


def compute_class_weights(manifest_path: str, data_dir: str, n_classes: int = 5) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the full dataset.

    Returns a tensor of shape [n_classes] for use with FocalLoss alpha.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    data_dir = Path(data_dir)

    manifest = load_manifest(manifest_path)
    entries = manifest['entries']

    # Sample a subset for speed if dataset is large
    sample = entries[:200] if len(entries) > 200 else entries

    for entry in sample:
        data = np.load(data_dir / entry['filename'])
        labels = data[entry['labels_key']]  # [T]
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


def _entry_song_key(entry: dict) -> str:
    """Best-effort stable song key for by-song splitting.

    Tries common manifest fields in order; falls back to the filename stem
    with the difficulty token stripped.
    """
    for k in ('song_id', 'song_key', 'sm_path', 'song_title', 'title'):
        if k in entry and entry[k]:
            return str(entry[k])
    stem = Path(entry['filename']).stem
    for tok in ('_beginner', '_easy', '_medium', '_hard', '_challenge'):
        if stem.endswith(tok):
            return stem[:-len(tok)]
    return stem


def split_entries_by_song(
    entries: List[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Hash-based by-song train/val split over entry indices.

    All entries belonging to the same song (regardless of difficulty) end up
    in the same partition, so val metrics aren't leaked by in-song chunks.
    """
    rng = np.random.default_rng(seed)
    songs = {}
    for i, e in enumerate(entries):
        songs.setdefault(_entry_song_key(e), []).append(i)
    song_keys = sorted(songs.keys())
    rng.shuffle(song_keys)
    n_val = max(1, int(round(len(song_keys) * val_fraction)))
    val_songs = set(song_keys[:n_val])
    train_idx, val_idx = [], []
    for key, idxs in songs.items():
        (val_idx if key in val_songs else train_idx).extend(idxs)
    logger.info(
        f"By-song split: {len(song_keys)} songs -> "
        f"{len(train_idx)} train entries, {len(val_idx)} val entries"
    )
    return sorted(train_idx), sorted(val_idx)


def compute_note_counts(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Load labels once per entry and return an int64 array of note counts.

    A "note" is a frame with any non-zero label (tap, jump, hold_start, hold_end).
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    counts = np.zeros(len(manifest), dtype=np.int64)
    for i in idxs:
        entry = manifest[i]
        labels = np.load(data_dir / entry['filename'])[entry['labels_key']]
        counts[i] = int((labels > 0).sum())
    return counts


def make_note_weighted_sampler(
    entry_indices: List[int],
    note_counts: np.ndarray,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Weight entries proportionally to max(note_count, 1) for training.

    Passed to a DataLoader wrapping a Subset(full_dataset, entry_indices).
    The sampler indexes *into the subset*, so weights must match that order.
    """
    weights = np.asarray(
        [max(int(note_counts[i]), 1) for i in entry_indices], dtype=np.float64
    )
    weights = weights / weights.sum() * len(weights)
    if num_samples is None:
        num_samples = len(entry_indices)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )


def compute_onset_prior(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 200,
) -> float:
    """Empirical per-frame onset rate (fraction of frames with a note).

    Used to bias-init the onset head so the model starts at the correct base
    rate for a highly imbalanced frame-level prediction problem.
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    if len(idxs) > sample_limit:
        step = max(1, len(idxs) // sample_limit)
        idxs = idxs[::step][:sample_limit]
    pos = 0
    total = 0
    for i in idxs:
        entry = manifest[i]
        labels = np.load(data_dir / entry['filename'])[entry['labels_key']]
        pos += int((labels > 0).sum())
        total += int(labels.shape[0])  # [T] — one value per frame
    p = pos / max(total, 1)
    logger.info(f"Empirical per-frame onset rate: {p:.6f}")
    return float(p)


def compute_type_class_distribution(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 400,
) -> np.ndarray:
    """Empirical P(class | onset) over {tap, jump, hold_start}.

    Reads `labels_<diff>` arrays directly from the v3 npz files.
    Returns a 3-vector that sums to 1 (well-defined as long as any onsets
    were observed; falls back to uniform otherwise).

    Used for two things:
      1. Bias-init the type head's final layer to log(prior).
      2. Derive inverse-frequency class weights for the type-head CE loss.
    Both push the model away from the "always predict tap" trivial solution.
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    if len(idxs) > sample_limit:
        step = max(1, len(idxs) // sample_limit)
        idxs = idxs[::step][:sample_limit]
    counts = np.zeros(3, dtype=np.int64)
    for i in idxs:
        entry = manifest[i]
        labels = np.load(data_dir / entry['filename'])[entry['labels_key']]
        # labels: 0=none, 1=tap, 2=jump, 3=hold_start
        counts[0] += int((labels == 1).sum())
        counts[1] += int((labels == 2).sum())
        counts[2] += int((labels == 3).sum())
    total = counts.sum()
    if total == 0:
        logger.warning("No onsets found while computing type prior; using uniform.")
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    prior = counts.astype(np.float64) / float(total)
    logger.info(
        f"Type class distribution: tap={prior[0]:.4f} "
        f"jump={prior[1]:.4f} hold_start={prior[2]:.4f} "
        f"(counts={counts.tolist()})"
    )
    return prior.astype(np.float32)


def class_weights_from_prior(
    prior: np.ndarray,
    smoothing: float = 0.5,
    max_weight: float = 12.0,
) -> torch.Tensor:
    """Inverse-frequency class weights with sqrt smoothing + a cap.

    Pure inverse frequency on a heavily skewed prior (e.g. tap=0.9,
    hold=0.03) produces weights like [1, 9, 30] which destabilizes
    training — the type loss explodes whenever the rare class shows up.
    Sqrt smoothing (`prior ** smoothing`) flattens the ratio, and the cap
    prevents any single class from dominating the gradient.

    Returns a length-3 tensor in {tap, jump, hold_start} order, normalized
    so the mean weight is 1.0 (preserves overall type-loss magnitude).
    """
    p = np.asarray(prior, dtype=np.float64).clip(min=1e-6)
    raw = 1.0 / (p ** smoothing)
    raw = np.minimum(raw, max_weight)
    raw = raw / raw.mean()  # mean weight = 1
    logger.info(f"Type class weights (smoothing={smoothing}): {raw.tolist()}")
    return torch.from_numpy(raw.astype(np.float32))


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
    manifest = load_manifest(manifest_path)
    for entry in manifest['entries']:
        d = int(entry['difficulty_id'])
        labels = np.load(data_dir / entry['filename'])[entry['labels_key']]  # [T]
        # Count frames with tap (1), jump (2), or hold_start (3) as steps
        step_frames = int(((labels >= 1) & (labels <= 3)).sum())
        sums[d] += step_frames
        secs[d] += labels.shape[0] / frames_per_second
    means = sums / np.maximum(secs, 1.0)
    logger.info(f"Empirical mean density per difficulty: {means.tolist()}")
    return torch.from_numpy(means.astype(np.float32))
