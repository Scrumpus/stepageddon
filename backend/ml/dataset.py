"""
PyTorch Dataset for step chart training data (v8 — arrow labels + structure).

Loads preprocessed feats[T,88] tensors plus per-chart label arrays, returns
random chunks for training. v8 adds per-arrow label targets (arrow_labels[T,4])
and structural conditioning scalars (section_position, near_boundary).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

from ml.prepare_data import DB_RANGE, FORMAT_VERSION, N_FEAT_CHANNELS, N_MELS

logger = logging.getLogger(__name__)


# Default fallback steps/sec per difficulty_id, used at INFERENCE time only
# when the caller does not provide an explicit target density.
DEFAULT_DENSITY_BY_ID = torch.tensor(
    [0.75, 1.25, 1.80, 3.10, 4.75], dtype=torch.float32
)

DENSITY_MEAN = 2.0
DENSITY_STD = 1.5

# Frames per second of the feats grid (matches prepare_data.FRAMES_PER_SECOND).
FRAMES_PER_SECOND = 22050 / 220  # ~100.23

DIFFICULTY_ID_TO_NAME = {
    0: 'beginner', 1: 'easy', 2: 'medium', 3: 'hard', 4: 'challenge',
}


def _build_sustain_target(
    arrow_labels: np.ndarray,
    labels: np.ndarray,
    durations_seconds: np.ndarray,
    fps: float,
    boundary_ramp: int = 2,
) -> np.ndarray:
    """Per-frame "any arrow currently held" indicator over the full song.

    Hold boundaries are inherently ambiguous to within a frame or two, so the
    target ramps linearly to/from 1.0 just outside each segment instead of
    flipping hard. The hysteresis-based eval decoder expects exactly this
    kind of soft boundary; hard 0/1 labels punished the model for 1-frame
    boundary jitter and depressed val IoU/F1.

    Args:
        arrow_labels: [T, 4] uint8 — per-arrow onset indicator.
        labels: [T] uint8 — frame-level note class. Hold starts have value 3.
        durations_seconds: [T] float32 — total hold duration in seconds at
            hold_start frames, 0 elsewhere.
        boundary_ramp: number of frames on each side of a hold span to soften.

    Returns:
        sustain: [T] float32 in [0, 1]. 1.0 inside any hold span on any arrow,
            ramping down outside the span across `boundary_ramp` frames.
    """
    T = arrow_labels.shape[0]
    out = np.zeros(T, dtype=np.float32)
    if T == 0:
        return out
    hs_indices = np.flatnonzero(labels == 3)
    if hs_indices.size == 0:
        return out
    for hs in hs_indices:
        dur_seconds = float(durations_seconds[hs])
        if dur_seconds <= 0.0:
            continue
        dur_frames = max(1, int(round(dur_seconds * fps)))
        start = int(hs)
        end = min(T, start + dur_frames)
        out[start:end] = 1.0
        for k in range(1, boundary_ramp + 1):
            w = 1.0 - k / (boundary_ramp + 1)
            left = start - k
            if left >= 0:
                out[left] = max(out[left], w)
            right = end + k - 1
            if right < T:
                out[right] = max(out[right], w)
    return out


def _build_intensity_target(
    labels: np.ndarray,
    smear_radius: int = 2,
) -> np.ndarray:
    """Per-frame jump intensity target.

    Jumps (label=2) get a peak of 1.0 smeared over `smear_radius` frames.
    Taps (label=1) contribute 0.0. Frames with no note are 0.0.

    Returns:
        intensity: [T] float32 in [0, 1].
    """
    T = labels.shape[0]
    out = np.zeros(T, dtype=np.float32)
    jump_frames = np.flatnonzero(labels == 2)
    if jump_frames.size == 0:
        return out
    for jf in jump_frames:
        lo = max(0, jf - smear_radius)
        hi = min(T, jf + smear_radius + 1)
        out[lo:hi] = np.maximum(out[lo:hi], 1.0)
    return out


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, 'r') as f:
        return json.load(f)


class StepChartDataset(Dataset):
    """PyTorch Dataset for v8 step chart training.

    Each item is a random (or sequential for val) fixed-length chunk from a
    preprocessed song file. v8 returns arrow labels and structural features
    in addition to the v7 targets.
    """

    _ARROW_KEYS = ['arrow_labels_key', 'arrow_labels_beginner', 'arrow_labels_easy',
                    'arrow_labels_medium', 'arrow_labels_hard', 'arrow_labels_challenge']
    _LABELS_KEYS = ['labels_key', 'labels_beginner', 'labels_easy',
                     'labels_medium', 'labels_hard', 'labels_challenge']
    _DURATIONS_KEYS = ['durations_key', 'durations_beginner', 'durations_easy',
                        'durations_medium', 'durations_hard', 'durations_challenge']

    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        chunk_frames: int = 500,
        is_train: bool = True,
        augment: bool = False,
        density_swap_prob: float = 0.5,
        default_density_by_id: Optional[torch.Tensor] = None,
        intro_outro_oversample_prob: float = 0.15,
        gain_jitter_db: float = 3.0,
        spec_time_masks: int = 2,
        spec_time_width: int = 20,
        spec_freq_masks: int = 2,
        spec_freq_width: int = 8,
        onset_sigma_frames: float = 2.0,
        intensity_smear_radius: int = 2,
    ):
        super().__init__()
        if not (2 <= chunk_frames <= 4000):
            raise ValueError(f"chunk_frames must be in [2, 4000], got {chunk_frames}")
        if spec_freq_width > 24:
            raise ValueError(f"spec_freq_width must be ≤ 24, got {spec_freq_width}")

        self.data_dir = Path(data_dir)
        self.chunk_frames = int(chunk_frames)
        self.is_train = bool(is_train)
        self.augment = bool(augment) and is_train
        self.density_swap_prob = float(density_swap_prob) if is_train else 0.0
        self._default_density_by_id = default_density_by_id
        self.gain_jitter_db = float(gain_jitter_db)
        self.spec_time_masks = int(spec_time_masks)
        self.spec_time_width = int(spec_time_width)
        self.spec_freq_masks = int(spec_freq_masks)
        self.spec_freq_width = int(spec_freq_width)
        self.intro_outro_oversample_prob = float(intro_outro_oversample_prob)
        self.intensity_smear_radius = int(intensity_smear_radius)
        self.n_in_channels = N_FEAT_CHANNELS

        self.manifest = load_manifest(manifest_path)
        manifest_format = self.manifest.get('format_version', 1)
        if isinstance(manifest_format, int) and manifest_format < 7:
            logger.warning(
                f"Manifest format_version={manifest_format} is older than v7. "
                f"Some features may be missing."
            )

        self.n_mels = int(self.manifest.get('n_mels', N_MELS))
        self.db_min = float(self.manifest.get('db_min', -80))
        self.db_max = float(self.manifest.get('db_max', 20))

        self.feat_mean = np.asarray(self.manifest['feat_mean'], dtype=np.float32)
        self.feat_std = np.asarray(self.manifest['feat_std'], dtype=np.float32)
        self.feat_std = np.where(self.feat_std < 1e-6, 1.0, self.feat_std).astype(np.float32)

        self.entries = self.manifest.get('entries', self.manifest)

        # Filter entries whose npz has too few frames.
        min_chunk_frames = int(chunk_frames)
        self.entries = [
            e for e in self.entries
            if int(e.get('n_frames', 0)) >= min_chunk_frames
        ]

        # Build onset Gaussian kernel (σ in frames → 1-D kernel of width ~6σ).
        sigma = max(float(onset_sigma_frames), 0.25)
        half = max(1, int(round(3.0 * sigma)))
        t = np.arange(-half, half + 1, dtype=np.float32)
        self._onset_kernel = np.exp(-0.5 * (t / sigma) ** 2)
        self._onset_kernel /= self._onset_kernel.sum()

        # Pre-build val chunk index so validation is deterministic.
        self._val_chunks: List[Tuple[int, int]] = []

        logger.info(
            f"Loaded {len(self.entries)} entries "
            f"({len(self.manifest.get('entries', self.manifest)) - len(self.entries)} skipped as too short)"
        )
        logger.info(
            f"Per-channel feat stats loaded: "
            f"mel mean range [{self.feat_mean[:80].min():.4f}, {self.feat_mean[:80].max():.4f}] "
            f"side mean range [{self.feat_mean[80:].min():.4f}, {self.feat_mean[80:].max():.4f}]"
        )

        if not is_train:
            self._val_chunks = []
            for idx, entry in enumerate(self.entries):
                n_frames = entry['n_frames']
                for start in range(0, n_frames - chunk_frames + 1, chunk_frames):
                    self._val_chunks.append((idx, start))

    def __len__(self):
        if self.is_train:
            return len(self.entries)
        return len(self._val_chunks)

    def __getitem__(self, idx):
        if self.is_train:
            entry = self.entries[idx]
        else:
            entry_idx, val_start = self._val_chunks[idx]
            entry = self.entries[entry_idx]

        n_frames = entry['n_frames']
        max_start = n_frames - self.chunk_frames

        if self.is_train:
            p_edge = self.intro_outro_oversample_prob
            r = np.random.random() if p_edge > 0.0 else 1.0
            if r < p_edge:
                start = 0
            elif r < 2 * p_edge:
                start = max_start
            else:
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            start = val_start

        data = np.load(self.data_dir / entry['filename'])
        feats_full = data['feats']                        # [T, 88] float16
        labels = data[entry['labels_key']]                # [T] uint8
        durations = data[entry['durations_key']]          # [T] float32 seconds at hold_start
        arrow_labels = data[entry['arrow_labels_key']]    # [T, 4] uint8

        # Build per-song streams before slicing so chunks starting mid-hold
        # still get correct supervision.
        sustain_full = _build_sustain_target(
            arrow_labels, labels, durations, FRAMES_PER_SECOND,
        )
        intensity_full = _build_intensity_target(
            labels, smear_radius=self.intensity_smear_radius,
        )

        # Structural features (v8): read from npz if available, else zeros.
        # section_positions[T] float32: position within detected section [0,1]
        # near_boundary[T] float32: 1.0 near section boundary, 0.0 elsewhere
        if 'section_positions' in data.files:
            section_positions_full = data['section_positions']  # [T] float32
        else:
            section_positions_full = np.zeros(n_frames, dtype=np.float32)
        if 'near_boundary' in data.files:
            near_boundary_full = data['near_boundary']  # [T] float32
        else:
            near_boundary_full = np.zeros(n_frames, dtype=np.float32)

        end = start + self.chunk_frames
        feats_chunk = feats_full[start:end].astype(np.float32)  # [T, 88] in [0,1] for mel sub-block
        labels_chunk = labels[start:end].astype(np.int64)
        sustain_chunk = sustain_full[start:end].astype(np.float32)
        intensity_chunk = intensity_full[start:end].astype(np.float32)
        arrow_chunk = arrow_labels[start:end].astype(np.float32)       # [T, 4]
        section_pos_chunk = section_positions_full[start:end].astype(np.float32)
        near_boundary_chunk = near_boundary_full[start:end].astype(np.float32)
        difficulty = entry['difficulty_id']

        # Per-chunk step density (steps/sec): any frame with a note event.
        step_frames = int((labels_chunk >= 1).sum())
        chunk_seconds = self.chunk_frames / FRAMES_PER_SECOND
        density = step_frames / chunk_seconds

        start_seconds = float(start) / FRAMES_PER_SECOND
        remaining_seconds = max(
            0.0, float(n_frames - end) / FRAMES_PER_SECOND
        )

        if (
            self.is_train
            and self._default_density_by_id is not None
            and self.density_swap_prob > 0.0
            and np.random.random() < self.density_swap_prob
        ):
            density = float(self._default_density_by_id[difficulty].item())
        density_norm = (density - DENSITY_MEAN) / DENSITY_STD

        # Onset target: row-max over arrow_labels at each frame, then smoothed.
        onset_hard = (labels_chunk >= 1).astype(np.float32)
        if self.is_train and self._onset_kernel is not None:
            onset_soft = self._smooth_1d(onset_hard)
        else:
            onset_soft = onset_hard

        # ---- Audio augmentation + whitening ----
        if self.augment and self.gain_jitter_db > 0:
            db_shift = float(np.random.uniform(-self.gain_jitter_db, self.gain_jitter_db))
            mel_block = feats_chunk[:, :self.n_mels]
            mel_block = mel_block + (db_shift / DB_RANGE)
            np.clip(mel_block, 0.0, 1.0, out=mel_block)
            feats_chunk[:, :self.n_mels] = mel_block

        # Per-channel whitening across all 88 channels.
        feats_chunk = (feats_chunk - self.feat_mean) / self.feat_std

        if self.augment:
            feats_chunk = self._spec_augment_mel(feats_chunk)

        # Structural scalars for FiLM: average section_position and near_boundary
        # over the chunk for the batch-level conditioning vector.
        section_position_val = float(section_pos_chunk.mean())
        near_boundary_val = float(near_boundary_chunk.max())  # max so any boundary nearby is detected

        return (
            torch.from_numpy(feats_chunk),
            torch.tensor(difficulty, dtype=torch.long),
            torch.tensor(density_norm, dtype=torch.float32),
            torch.from_numpy(onset_soft).unsqueeze(-1),         # [T, 1]
            torch.from_numpy(sustain_chunk).unsqueeze(-1),      # [T, 1]
            torch.from_numpy(intensity_chunk).unsqueeze(-1),    # [T, 1]
            torch.from_numpy(arrow_chunk),                      # [T, 4]
            torch.tensor(start_seconds, dtype=torch.float32),
            torch.tensor(remaining_seconds, dtype=torch.float32),
            torch.tensor(section_position_val, dtype=torch.float32),
            torch.tensor(near_boundary_val, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _smooth_1d(self, hard: np.ndarray) -> np.ndarray:
        """Convolve a [T] hard label vector with the 1-D Gaussian kernel."""
        k = self._onset_kernel
        out = np.convolve(hard, k, mode='same')
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _spec_augment_mel(self, feats: np.ndarray) -> np.ndarray:
        """SpecAugment-style time + freq masking restricted to the mel sub-block.

        Operates on the whitened feats tensor. We zero (= channel mean in
        whitened space) only the mel channels [0..n_mels-1] — the side
        channels (onset_strength + spectral_contrast) carry per-song-z-normed
        rhythmic cues and don't benefit from masking.
        """
        feats = feats.copy()
        T = feats.shape[0]

        # Time masks (full-frame on the mel block only).
        for _ in range(self.spec_time_masks):
            w = int(np.random.randint(0, self.spec_time_width + 1))
            if w == 0 or w >= T:
                continue
            t0 = int(np.random.randint(0, T - w + 1))
            feats[t0:t0 + w, :self.n_mels] = 0.0

        # Freq masks (mel sub-block only).
        for _ in range(self.spec_freq_masks):
            w = int(np.random.randint(0, self.spec_freq_width + 1))
            if w == 0 or w >= self.n_mels:
                continue
            f0 = int(np.random.randint(0, self.n_mels - w + 1))
            feats[:, f0:f0 + w] = 0.0

        return feats.astype(np.float32)


def _entry_song_key(entry: dict) -> str:
    """Best-effort stable song key for by-song splitting."""
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
    """Hash-based by-song train/val split over entry indices."""
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


def compute_onset_prior(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 200,
) -> float:
    """Empirical per-frame onset rate (fraction of frames with a note)."""
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
        total += int(labels.shape[0])
    p = pos / max(total, 1)
    logger.info(f"Empirical per-frame onset rate: {p:.6f}")
    return float(p)


def compute_sustain_prior(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 200,
) -> float:
    """Empirical per-frame "any arrow held" rate.

    Used to bias-init the sustain head. Returns 0.03 (≈3%) as fallback if
    no holds are found in the sampled archives.
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    if len(idxs) > sample_limit:
        step = max(1, len(idxs) // sample_limit)
        idxs = idxs[::step][:sample_limit]
    pos = 0
    total = 0
    seen_any = False
    for i in idxs:
        entry = manifest[i]
        archive = np.load(data_dir / entry['filename'])
        labels_key = entry.get('labels_key')
        durations_key = entry.get('durations_key')
        arrow_key = entry.get('arrow_labels_key')
        if not (
            labels_key in archive.files
            and durations_key in archive.files
            and arrow_key in archive.files
        ):
            continue
        labels = archive[labels_key]
        durations = archive[durations_key]
        arrow_labels = archive[arrow_key]
        sustain = _build_sustain_target(
            arrow_labels, labels, durations, FRAMES_PER_SECOND,
        )
        seen_any = True
        pos += int((sustain > 0.5).sum())
        total += int(sustain.shape[0])
    if not seen_any or total == 0:
        logger.warning(
            "No sustain data found while computing sustain prior; using 0.03 fallback."
        )
        return 0.03
    p = pos / max(total, 1)
    logger.info(f"Empirical per-frame sustain rate: {p:.6f}")
    return float(p)


def compute_default_density_per_difficulty(
    manifest_path: str,
    data_dir: str,
    n_difficulties: int = 5,
    frames_per_second: float = FRAMES_PER_SECOND,
) -> torch.Tensor:
    """Empirical mean steps/sec per difficulty_id from the manifest.

    Saved into the checkpoint so inference uses data-driven defaults.
    """
    sums = np.zeros(n_difficulties, dtype=np.float64)
    secs = np.zeros(n_difficulties, dtype=np.float64)
    data_dir = Path(data_dir)
    manifest = load_manifest(manifest_path)
    entries = manifest.get('entries', manifest)
    for e in entries:
        did = int(e.get('difficulty_id', -1))
        if did < 0 or did >= n_difficulties:
            continue
        n_frames = int(e.get('n_frames', 0))
        n_notes = int(e.get('n_notes', 0))
        song_sec = n_frames / frames_per_second
        sums[did] += n_notes
        secs[did] += song_sec
    defaults = np.zeros(n_difficulties, dtype=np.float32)
    for d in range(n_difficulties):
        defaults[d] = sums[d] / max(secs[d], 1.0) if secs[d] > 0 else float(DEFAULT_DENSITY_BY_ID[d])
    logger.info(
        f"Empirical per-difficulty densities: {defaults.tolist()}"
    )
    return torch.from_numpy(defaults)
