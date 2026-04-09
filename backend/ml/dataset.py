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
        spec_augment: bool = True,
        freq_mask_max: int = 15,
        freq_mask_n: int = 2,
        time_mask_max: int = 40,
        time_mask_n: int = 2,
        mixup_prob: float = 0.25,
        mixup_alpha_range: tuple = (0.3, 0.7),
        samples_per_entry: int = 1,
    ):
        """
        Args:
            data_dir: Directory containing preprocessed .npz files
            manifest_path: Path to manifest.json listing all examples
            chunk_frames: Number of frames per training chunk (500 = 5s at 100fps)
            is_train: If True, use random chunking; else use sequential chunks
            n_mels: Number of mel bins (augmentation only touches this slice
                of the feature axis, leaving rhythm channels intact)
            spec_augment: Apply SpecAugment (freq/time masking) during training
            freq_mask_max / freq_mask_n: width and number of frequency masks
            time_mask_max / time_mask_n: width and number of time masks
            mixup_prob: Probability of mixing two same-difficulty chunks
                during training. Targets are combined with element-wise max
                (since they are binary), mel with a linear mix. Set to 0 to
                disable.
            mixup_alpha_range: (low, high) range for the mixing coefficient
                of the *first* chunk in a mixup pair.
        """
        self.data_dir = Path(data_dir)
        self.chunk_frames = chunk_frames
        self.is_train = is_train
        self.n_mels = n_mels
        self.spec_augment = spec_augment and is_train
        self.freq_mask_max = freq_mask_max
        self.freq_mask_n = freq_mask_n
        self.time_mask_max = time_mask_max
        self.time_mask_n = time_mask_n
        self.mixup_prob = mixup_prob if is_train else 0.0
        self.mixup_alpha_range = mixup_alpha_range
        # samples_per_entry lets each song contribute multiple random
        # chunks per epoch, mitigating the systematic gaps that come from
        # one random chunk per song. Only affects training.
        self.samples_per_entry = samples_per_entry if is_train else 1

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Filter out entries shorter than chunk_frames
        self.entries = [
            e for e in self.manifest
            if e['n_frames'] >= chunk_frames
        ]

        logger.info(f"Loaded {len(self.entries)} examples "
                     f"({len(self.manifest) - len(self.entries)} skipped as too short)")

        # Index entries by difficulty_id so mixup can draw same-difficulty
        # partners without scanning the manifest on every sample.
        self._entries_by_diff: dict = {}
        for i, e in enumerate(self.entries):
            self._entries_by_diff.setdefault(e['difficulty_id'], []).append(i)

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
            return len(self.entries) * self.samples_per_entry
        return len(self._val_chunks)

    def _load_chunk(self, entry_idx: int, start: int = None):
        """Load and slice one (mel_concat, targets, placement, density) tuple.

        Factored out so mixup can reuse it to fetch a second chunk.
        Returns numpy arrays (not tensors).
        """
        entry = self.entries[entry_idx]
        n_frames = entry['n_frames']
        if start is None:
            max_start = n_frames - self.chunk_frames
            start = np.random.randint(0, max_start + 1)

        data = np.load(self.data_dir / entry['filename'])
        mel = data['mel']
        if 'rhythm' in data.files:
            rhythm = data['rhythm']
        else:
            rhythm = np.zeros((mel.shape[0], 2), dtype=np.float16)
        labels_full = data['labels']
        targets_full = labels_to_binary_targets(labels_full)

        end = start + self.chunk_frames
        mel_chunk = mel[start:end].astype(np.float32)
        rhythm_chunk = rhythm[start:end].astype(np.float32)
        features = np.concatenate([mel_chunk, rhythm_chunk], axis=-1)  # [chunk, n_mels + 2]
        targets_chunk = targets_full[start:end].astype(np.float32)     # [chunk, 4, 3]

        labels_chunk = labels_full[start:end]
        placement_mask = ((labels_chunk == 1) | (labels_chunk == 2)).any(axis=1).astype(np.float32)
        step_frames = float(placement_mask.sum())
        chunk_seconds = self.chunk_frames / FRAMES_PER_SECOND
        density = step_frames / chunk_seconds

        return features, targets_chunk, placement_mask, density, entry['difficulty_id']

    def _apply_spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to the mel portion only; time masks also
        zero the rhythm columns so the model can't cheat around them.
        """
        T = features.shape[0]
        mel_end = self.n_mels  # mel bins occupy [:, :n_mels]
        for _ in range(self.freq_mask_n):
            w = int(np.random.randint(0, self.freq_mask_max + 1))
            if w == 0 or w >= mel_end:
                continue
            f0 = int(np.random.randint(0, mel_end - w + 1))
            features[:, f0:f0 + w] = 0.0
        for _ in range(self.time_mask_n):
            w = int(np.random.randint(0, self.time_mask_max + 1))
            if w == 0 or w >= T:
                continue
            t0 = int(np.random.randint(0, T - w + 1))
            features[t0:t0 + w, :] = 0.0
        return features

    def __getitem__(self, idx):
        if self.is_train:
            entry_idx = idx % len(self.entries)
            start = None
        else:
            entry_idx, start = self._val_chunks[idx]

        features, targets_chunk, placement_mask, density, difficulty = \
            self._load_chunk(entry_idx, start)

        # Mixup: blend with another random same-difficulty chunk. Targets
        # are binary, so element-wise max is the natural merge — a mixed
        # chunk should be labeled positive at any frame where *either*
        # source had a step.
        if self.is_train and self.mixup_prob > 0 and np.random.rand() < self.mixup_prob:
            pool = self._entries_by_diff.get(difficulty, [])
            if len(pool) > 1:
                other_idx = pool[np.random.randint(0, len(pool))]
                if other_idx != entry_idx:
                    f2, t2, p2, d2, _ = self._load_chunk(other_idx)
                    lo, hi = self.mixup_alpha_range
                    alpha = float(np.random.uniform(lo, hi))
                    features = alpha * features + (1.0 - alpha) * f2
                    targets_chunk = np.maximum(targets_chunk, t2)
                    placement_mask = np.maximum(placement_mask, p2)
                    # Density reflects the union mask (consistent with
                    # placement_mask after the max merge).
                    chunk_seconds = self.chunk_frames / FRAMES_PER_SECOND
                    density = float(placement_mask.sum()) / chunk_seconds

        if self.spec_augment:
            features = self._apply_spec_augment(features)

        density_norm = (density - DENSITY_MEAN) / DENSITY_STD

        return (
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.tensor(difficulty, dtype=torch.long),
            torch.tensor(density_norm, dtype=torch.float32),
            torch.from_numpy(targets_chunk.astype(np.float32, copy=False)),
            torch.from_numpy(placement_mask.astype(np.float32, copy=False)),
        )


def labels_to_binary_targets(labels: np.ndarray) -> np.ndarray:
    """
    Convert integer class labels to the three-head binary target format.

    Args:
        labels: [T, 4] uint8 with values in {0:none, 1:tap, 2:hold_start, 3:hold_end}

    Returns:
        [T, 4, 3] uint8 where the last axis is (tap, hold_state, hold_end):
            - tap       = 1 on frames where the class is 'tap'
            - hold_state= 1 from a hold_start frame through its hold_end
                          frame inclusive
            - hold_end  = 1 on hold_end frames
    """
    labels = np.asarray(labels)
    tap = (labels == 1).astype(np.uint8)
    starts = (labels == 2).astype(np.int8)
    ends = (labels == 3).astype(np.int8)

    # Build an in-hold indicator via cumulative sum: +1 on start frame,
    # -1 on the frame *after* each end frame, so that the end frame itself
    # remains inside the hold.
    delta = starts.copy()
    if delta.shape[0] > 1:
        delta[1:] -= ends[:-1]
    hold_state = (np.cumsum(delta, axis=0) > 0).astype(np.uint8)
    hold_end = ends.astype(np.uint8)

    return np.stack([tap, hold_state, hold_end], axis=-1)  # [T, 4, 3]


def compute_pos_weights(
    manifest_path: str,
    data_dir: str,
    max_sample: int = 200,
) -> torch.Tensor:
    """
    Compute BCE `pos_weight` (≈ neg/pos ratio) for each of the three heads:
    tap, hold_state, hold_end. Sampled from the first `max_sample` manifest
    entries for speed — the ratios stabilize well before that on any real
    dataset.

    Returns:
        tensor of shape [3], float32, ordered (tap, hold_state, hold_end).
    """
    data_dir = Path(data_dir)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    sample = manifest[:max_sample] if len(manifest) > max_sample else manifest

    pos = np.zeros(3, dtype=np.int64)
    neg = np.zeros(3, dtype=np.int64)
    placement_pos = 0
    placement_neg = 0
    for entry in sample:
        data = np.load(data_dir / entry['filename'])
        labels = data['labels']
        targets = labels_to_binary_targets(labels)  # [T, 4, 3]
        n = targets.shape[0] * targets.shape[1]
        for i in range(3):
            p = int(targets[..., i].sum())
            pos[i] += p
            neg[i] += n - p
        placement_mask = ((labels == 1) | (labels == 2)).any(axis=1)
        placement_pos += int(placement_mask.sum())
        placement_neg += int(labels.shape[0] - placement_mask.sum())

    ratios = neg / np.maximum(pos, 1)
    ratios = np.clip(ratios, 1.0, 200.0)
    placement_ratio = float(np.clip(placement_neg / max(placement_pos, 1), 1.0, 200.0))
    logger.info(f"Head positive fractions (tap, hold_state, hold_end): "
                f"{[int(p) for p in pos]} / {[int(p + n) for p, n in zip(pos, neg)]}")
    logger.info(f"Head pos_weights: {ratios.tolist()}")
    logger.info(f"Placement pos_weight: {placement_ratio:.2f} "
                f"(pos={placement_pos}, neg={placement_neg})")
    head_weights = torch.from_numpy(ratios.astype(np.float32))
    return head_weights, placement_ratio


# Backwards-compat shim: old callers imported `compute_class_weights` and
# passed the result to FocalLoss(alpha=...). The new loss expects
# `pos_weights`, which has the same shape-1 vector semantics, so we route
# old calls through the new function.
def compute_class_weights(manifest_path: str, data_dir: str, n_classes: int = 4) -> torch.Tensor:
    w, _ = compute_pos_weights(manifest_path, data_dir)
    return w


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

    # Per-class frequency over the train indices.
    # When the dataset duplicates entries via samples_per_entry > 1,
    # subset indices run 0..N*K-1 while self.entries has length N; mod
    # back into the entry space before looking up difficulty.
    n_entries = len(full_dataset.entries)
    diff_ids = np.array(
        [full_dataset.entries[i % n_entries]['difficulty_id'] for i in indices],
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
