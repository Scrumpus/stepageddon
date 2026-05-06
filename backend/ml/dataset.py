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


def _build_prev_arrow_stream(arrow_labels: np.ndarray) -> np.ndarray:
    """Per-frame "previous emitted arrow vector" for transition conditioning.

    Args:
        arrow_labels: [T, 4] uint8 — per-arrow onset indicator at each frame.

    Returns:
        prev_arrow: [T, 4] float32. prev_arrow[t] is the binary arrow vector
            at the most recent frame s < t with any onset (any column == 1),
            or zeros if no such s exists. Frames at the very start of a song
            (before any note) get a zero conditioning vector.

    The vector form (vs a single 5-class index) handles jumps naturally —
    both arrows of a jump appear in the conditioning vector. For chunks that
    don't start at frame 0, callers must feed the *song-level* stream so the
    chunk's first frames inherit conditioning from the prior song context.
    """
    T = arrow_labels.shape[0]
    out = np.zeros((T, 4), dtype=np.float32)
    if T == 0:
        return out
    last = np.zeros(4, dtype=np.float32)
    onset_mask = arrow_labels.any(axis=1)
    al = arrow_labels.astype(np.float32)
    for t in range(T):
        out[t] = last
        if onset_mask[t]:
            last = al[t]
    return out


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
        type_target_dilate: int = 1,
        density_swap_prob: float = 0.0,
        default_density_by_id: Optional[torch.Tensor] = None,
        intro_outro_oversample_prob: float = 0.15,
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
        # Frames to dilate the type-target window around each note. The onset
        # head is supervised on a Gaussian-smoothed window (~5 frames) but the
        # type head defaults to a single frame, which means adjacent frames
        # see "onset positive but type ignore". Dilating the type label fills
        # those frames with the same class — only into ignore (-100) slots,
        # never overwriting another note's class.
        self.type_target_dilate = int(type_target_dilate)

        # Density input swap: at training time, with this probability, the
        # density conditioning is replaced by the per-difficulty default
        # (the same constant inference uses) instead of the per-chunk value
        # computed from labels. Closes the train/inference gap on density —
        # without this the model can over-rely on a leaky per-chunk signal.
        self.density_swap_prob = float(density_swap_prob)
        self._default_density_by_id = (
            default_density_by_id.float()
            if default_density_by_id is not None else None
        )

        # Intro/outro oversampling: at training time, force a chunk to start
        # at song frame 0 with this probability (and force a chunk anchored
        # to the song's last frame with the same probability). Without this,
        # song-edge chunks are sampled uniformly — about 1/(n_frames - chunk)
        # — which is far too rare for the model to learn the intro-silence
        # pattern. Only used in the random-chunk training path.
        self.intro_outro_oversample_prob = float(intro_outro_oversample_prob)

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
        # Indices may be either:
        #   - int                     → random-chunk training path or val
        #   - (entry_idx, start) tuple → fixed-chunk training path used by the
        #     rare-class sampler to *guarantee* a chunk contains a jump/hold
        # Resolve `entry` first so `n_frames` is always available regardless
        # of which branch picks `start` below. Keeping the resolution above
        # the branches avoids the UnboundLocalError that bit us when one
        # branch forgot to populate n_frames.
        if self.is_train and isinstance(idx, tuple):
            entry_idx, fixed_start = idx
            entry = self.entries[entry_idx]
        elif self.is_train:
            entry = self.entries[idx]
            fixed_start = None
        else:
            entry_idx, val_start = self._val_chunks[idx]
            entry = self.entries[entry_idx]
        n_frames = entry['n_frames']
        max_start = n_frames - self.chunk_frames

        if self.is_train and isinstance(idx, tuple):
            # Jitter the fixed start so the rare event isn't always
            # centered in the chunk — keeps spatial diversity across
            # epochs even though the chunk is anchored to the rare frame.
            jitter = self.chunk_frames // 8
            low = max(0, fixed_start - jitter)
            high = min(max_start, fixed_start + jitter)
            start = int(np.random.randint(low, high + 1)) if high > low else int(fixed_start)
        elif self.is_train:
            # Intro/outro oversampling: with two independent
            # `intro_outro_oversample_prob` chances, anchor the chunk at
            # song start or song end so the model sees enough edge cases
            # to learn the empirical "silence at start/end" pattern.
            # Otherwise sample uniformly across the song.
            p_edge = self.intro_outro_oversample_prob
            r = np.random.random() if p_edge > 0.0 else 1.0
            if r < p_edge:
                start = 0
            elif r < 2 * p_edge:
                start = max_start
            else:
                start = np.random.randint(0, max_start + 1)
        else:
            start = val_start

        # Load data. v5 npz files contain mel, beats (song-level), and per-chart
        # labels_<diff>, durations_<diff>, arrow_labels_<diff>. Only requested
        # keys are decompressed by NpzFile on access.
        data = np.load(self.data_dir / entry['filename'])
        mel = data['mel']                          # [T, n_mels] float16, [0,1] scaled-dB
        labels = data[entry['labels_key']]         # [T] uint8 (0=none,1=tap,2=jump,3=hold_start)
        durations = data[entry['durations_key']]   # [T] float32, seconds at hold_start frames
        arrow_labels_key = entry.get('arrow_labels_key', f'arrow_{entry["labels_key"]}')
        if arrow_labels_key in data.files:
            arrow_labels_full = data[arrow_labels_key]  # [T, 4] uint8
        else:
            # Pre-v5 archive: synthesize a single-arrow indicator on column 0
            # so legacy data still loads. Per-arrow learning won't work with
            # this — re-run prepare_data.py for real arrow supervision.
            arrow_labels_full = np.zeros((mel.shape[0], 4), dtype=np.uint8)
            arrow_labels_full[:, 0] = (labels > 0).astype(np.uint8)
        if 'beats' in data.files:
            beats = data['beats']                  # [T] uint8 (1 at beat frames)
        else:
            # Pre-v4 archive: no beat track. Fall back to all-zero target so
            # the auxiliary head sees a no-op signal instead of crashing.
            beats = np.zeros(mel.shape[0], dtype=np.uint8)

        # Build the prev-arrow stream over the *full song* before chunking so
        # the chunk's first frames have correct conditioning even for chunks
        # that don't start at the beginning of the song. prev_arrow[t] is the
        # binary arrow vector at the most recent onset frame strictly before t,
        # or zeros if t precedes any onset.
        prev_arrow_full = _build_prev_arrow_stream(arrow_labels_full)

        # Extract chunk
        end = start + self.chunk_frames
        mel_chunk = mel[start:end].astype(np.float32)
        labels_chunk = labels[start:end].astype(np.int64)
        durations_chunk = durations[start:end].astype(np.float32)
        beats_chunk = beats[start:end].astype(np.float32)
        arrow_labels_chunk = arrow_labels_full[start:end].astype(np.float32)  # [T, 4]
        prev_arrow_chunk = prev_arrow_full[start:end].astype(np.float32)      # [T, 4]
        difficulty = entry['difficulty_id']

        # L↔R + D↔U mirror augmentation (180° pad rotation). Arrows are
        # columns [0=L, 1=D, 2=U, 3=R]; the mirrored permutation is [3,2,1,0].
        # The mel input is unchanged because audio carries no chirality — a
        # chart with all arrows rotated is an equally valid mapping for the
        # same audio. Doubles effective per-arrow data and breaks any
        # incidental L-side bias in the corpus. Type targets and durations
        # are arrow-agnostic so they don't need touching.
        if self.is_train and self.augment and np.random.random() < 0.5:
            mirror_idx = [3, 2, 1, 0]
            arrow_labels_chunk = arrow_labels_chunk[:, mirror_idx]
            prev_arrow_chunk = prev_arrow_chunk[:, mirror_idx]

        # Convert hold durations from seconds to *beats* using the song's
        # tempo. Beats factor BPM out of the regression target so the duration
        # head sees a much narrower distribution (DDR holds quantize to 1/4,
        # 1/2, 1, 2 beats regardless of tempo). Inference reverses this with
        # `beats * 60 / tempo` to recover seconds.
        tempo = float(entry.get('tempo', 120.0)) or 120.0
        durations_chunk = (durations_chunk * (tempo / 60.0)).astype(np.float32)

        # Per-chunk step density (steps/sec): a "step" is any frame with a
        # note event (tap, jump, or hold_start).
        step_frames = int((labels_chunk >= 1).sum())
        chunk_seconds = self.chunk_frames / FRAMES_PER_SECOND
        density = step_frames / chunk_seconds

        # Per-chunk song-position scalars used as FiLM conditioning so the
        # model can learn empirical edge-of-song behavior (silence at the
        # start, taper at the end). Both are clipped to >= 0 to guard
        # against the corner case where end > n_frames (padded chunk).
        start_seconds = float(start) / FRAMES_PER_SECOND
        remaining_seconds = max(
            0.0, float(n_frames - end) / FRAMES_PER_SECOND
        )

        # Density swap: at training time, sometimes substitute the
        # per-difficulty default (the inference-time constant) so the model
        # is comfortable with both signals. Eliminates the leaky per-chunk
        # input that's only available with labels.
        if (
            self.is_train
            and self._default_density_by_id is not None
            and self.density_swap_prob > 0.0
            and np.random.random() < self.density_swap_prob
        ):
            density = float(self._default_density_by_id[difficulty].item())
        density_norm = (density - DENSITY_MEAN) / DENSITY_STD

        # Per-arrow onset target: [T, 4] (soft during training, hard during val).
        # Smoothing is applied per channel — the same Gaussian kernel on each
        # arrow column independently. The "any onset" signal is the row-wise
        # max of arrow_labels and is no longer materialized as a separate head.
        if self.is_train and self._onset_kernel is not None:
            arrow_soft = self._smooth_arrows(arrow_labels_chunk)
        else:
            arrow_soft = arrow_labels_chunk

        # Beat target uses the same smoothing kernel as onsets so the auxiliary
        # BCE sees a comparable distribution. Hard beats (val) are kept tight.
        beat_hard = beats_chunk[:, None]
        if self.is_train and self._onset_kernel is not None:
            beat_soft = self._smooth_onset(beat_hard)
        else:
            beat_soft = beat_hard

        # Type target: [T] long with classes {0=tap, 1=jump, 2=hold_start}
        # labels_chunk - 1 maps {1,2,3} -> {0,1,2}; {0} -> -1, then to -100
        type_target = labels_chunk - 1
        type_target[type_target < 0] = -100
        if self.is_train and self.type_target_dilate > 0:
            type_target, durations_chunk = self._dilate_type_and_duration(
                labels_chunk, type_target, durations_chunk,
                self.type_target_dilate,
            )

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
            torch.from_numpy(arrow_soft),
            torch.from_numpy(type_target),
            torch.from_numpy(durations_chunk),
            torch.from_numpy(beat_soft),
            torch.from_numpy(prev_arrow_chunk),
            torch.tensor(start_seconds, dtype=torch.float32),
            torch.tensor(remaining_seconds, dtype=torch.float32),
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

    def _smooth_arrows(self, arrow_hard: np.ndarray) -> np.ndarray:
        """Per-channel Gaussian smoothing of a [T, 4] hard arrow map.

        Each arrow channel is convolved independently along time with the same
        kernel used for onset smoothing. This keeps per-arrow BCE targets in
        the same numeric range the loss expects (peak ~1.0 at the true onset
        frame, decaying to 0 a few frames away).
        """
        k = self._onset_kernel
        out = np.zeros_like(arrow_hard)
        for c in range(arrow_hard.shape[1]):
            out[:, c] = np.convolve(arrow_hard[:, c], k, mode='same')
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _dilate_type_and_duration(
        labels_chunk: np.ndarray,
        type_target: np.ndarray,
        durations_chunk: np.ndarray,
        width: int,
    ):
        """Spread each note's type class (and hold duration) into a ±width
        window of ignore frames.

        Original-frame labels are never overwritten, so when two notes are
        within 2*width of each other, both keep their own class — only the
        ignore frames between/around them get filled in. For hold_start
        notes, the corresponding duration value is also propagated into the
        same dilated frames so the duration loss sees the correct target
        everywhere it now sees `type==hold_start`.
        """
        n = type_target.shape[0]
        note_frames = np.where(labels_chunk > 0)[0]
        if note_frames.size == 0:
            return type_target, durations_chunk
        note_classes = (labels_chunk[note_frames] - 1).astype(np.int64)
        out_type = type_target.copy()
        out_dur = durations_chunk.copy()
        for f, c in zip(note_frames, note_classes):
            lo = max(0, int(f) - width)
            hi = min(n, int(f) + width + 1)
            type_window = out_type[lo:hi]
            ignore_mask = type_window == -100
            type_window[ignore_mask] = int(c)
            out_type[lo:hi] = type_window
            if c == 2:  # hold_start: propagate duration too
                dur_window = out_dur[lo:hi]
                dur_window[ignore_mask] = float(durations_chunk[f])
                out_dur[lo:hi] = dur_window
        return out_type, out_dur

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


def compute_note_type_counts(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Per-entry counts of (tap, jump, hold_start) frames.

    Returns an int64 array of shape [n_entries, 3]. Entries not in `indices`
    are left at zero — callers should index by the same indices used here.

    Reads each entry's labels archive once, so this is O(n_entries) I/O.
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    counts = np.zeros((len(manifest), 3), dtype=np.int64)
    for i in idxs:
        entry = manifest[i]
        labels = np.load(data_dir / entry['filename'])[entry['labels_key']]
        # labels: 0=none, 1=tap, 2=jump, 3=hold_start
        counts[i, 0] = int((labels == 1).sum())
        counts[i, 1] = int((labels == 2).sum())
        counts[i, 2] = int((labels == 3).sum())
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


def make_class_stratified_sampler(
    entry_indices: List[int],
    type_counts: np.ndarray,
    jump_boost: float = 8.0,
    hold_boost: float = 16.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Weight entries by (taps + jump_boost*jumps + hold_boost*hold_starts).

    Boosts entries containing rare class events so each batch is more likely
    to contain at least one chunk with a jump or a hold. The boost values
    were chosen so a song with one hold contributes roughly the same expected
    sampling weight as a song with `hold_boost` taps (compensates for the
    ~22× hold rarity vs taps).

    Indexes into the Subset, so weights are returned in `entry_indices` order.
    """
    boosts = np.array([1.0, float(jump_boost), float(hold_boost)], dtype=np.float64)
    weights = np.array(
        [
            max(float((type_counts[i].astype(np.float64) * boosts).sum()), 1.0)
            for i in entry_indices
        ],
        dtype=np.float64,
    )
    weights = weights / weights.sum() * len(weights)
    if num_samples is None:
        num_samples = len(entry_indices)
    logger.info(
        f"Class-stratified sampler: jump_boost={jump_boost} hold_boost={hold_boost} "
        f"weight_min={weights.min():.3f} weight_max={weights.max():.3f} "
        f"weight_mean={weights.mean():.3f}"
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=num_samples,
        replacement=True,
    )


def compute_rare_chunk_index(
    manifest: List[dict],
    data_dir: str,
    indices: List[int],
    chunk_frames: int,
) -> List[Tuple[int, int]]:
    """Per-rare-frame (entry_idx, chunk_start) anchors covering jump/hold events.

    For each frame labeled jump (2) or hold_start (3) in the train split,
    emit a chunk start that is guaranteed to contain that frame — the
    sampler can then draw uniformly over rare events instead of over songs,
    which is what fixes the per-batch rare-class coverage problem.

    Cost: O(n_train_entries) full label reads, ~the same as compute_note_counts.
    """
    data_dir = Path(data_dir)
    out: List[Tuple[int, int]] = []
    for i in indices:
        entry = manifest[i]
        n_frames = int(entry['n_frames'])
        if n_frames < chunk_frames:
            continue
        labels = np.load(data_dir / entry['filename'])[entry['labels_key']]
        rare = np.where((labels == 2) | (labels == 3))[0]
        if rare.size == 0:
            continue
        max_start = n_frames - chunk_frames
        half = chunk_frames // 2
        for f in rare:
            start = int(f) - half
            if start < 0:
                start = 0
            elif start > max_start:
                start = max_start
            out.append((i, start))
    logger.info(
        f"Rare-chunk index: {len(out)} (entry, start) pairs over "
        f"{len(indices)} train entries (chunk_frames={chunk_frames})"
    )
    return out


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


def compute_arrow_priors(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 200,
) -> np.ndarray:
    """Empirical per-frame fire rate for each of the 4 arrows.

    Returns a length-4 float64 array P(arrow_a fires at any frame). Used to
    bias-init the arrow head with per-arrow logit(p) and to derive per-arrow
    BCE pos_weight (otherwise the model collapses to "predict no arrow" or
    leans on the most common arrow). Falls back to a uniform 0.01 if no
    arrow_labels are found (legacy data).
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    if len(idxs) > sample_limit:
        step = max(1, len(idxs) // sample_limit)
        idxs = idxs[::step][:sample_limit]
    pos = np.zeros(4, dtype=np.int64)
    total = 0
    seen_any = False
    for i in idxs:
        entry = manifest[i]
        archive = np.load(data_dir / entry['filename'])
        key = entry.get('arrow_labels_key')
        if not key or key not in archive.files:
            continue
        seen_any = True
        arr = archive[key]                     # [T, 4] uint8
        pos += arr.sum(axis=0).astype(np.int64)
        total += int(arr.shape[0])
    if not seen_any or total == 0:
        logger.warning(
            "No arrow_labels found while computing arrow priors; using 0.01 "
            "uniform fallback. Re-run prepare_data.py."
        )
        return np.full(4, 0.01, dtype=np.float64)
    priors = pos.astype(np.float64) / float(total)
    logger.info(
        f"Per-arrow positive rate: L={priors[0]:.5f} D={priors[1]:.5f} "
        f"U={priors[2]:.5f} R={priors[3]:.5f}"
    )
    return priors


def compute_beat_prior(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 200,
) -> float:
    """Empirical per-frame beat rate (fraction of frames marked as a beat).

    Used to bias-init the beat head. Returns the corpus fallback (0.02) when
    no `beats` arrays are present (legacy npz files), so the head still
    starts at a reasonable rate even on stale data.
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    if len(idxs) > sample_limit:
        step = max(1, len(idxs) // sample_limit)
        idxs = idxs[::step][:sample_limit]
    pos = 0
    total = 0
    seen_any_beats = False
    seen_song_files = set()
    for i in idxs:
        entry = manifest[i]
        # Beats live at song level; only count each file once even if multiple
        # difficulties of the same song appear in `idxs`.
        if entry['filename'] in seen_song_files:
            continue
        seen_song_files.add(entry['filename'])
        archive = np.load(data_dir / entry['filename'])
        if 'beats' not in archive.files:
            continue
        seen_any_beats = True
        beats = archive['beats']
        pos += int((beats > 0).sum())
        total += int(beats.shape[0])
    if not seen_any_beats:
        logger.warning(
            "No `beats` arrays found in sampled archives; using 0.02 fallback. "
            "Re-run prepare_data.py to enable the auxiliary beat head."
        )
        return 0.02
    p = pos / max(total, 1)
    logger.info(f"Empirical per-frame beat rate: {p:.6f}")
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


def compute_hold_duration_median(
    manifest: List[dict],
    data_dir: str,
    indices: Optional[List[int]] = None,
    sample_limit: int = 400,
) -> float:
    """Empirical median hold duration in *beats* (across all hold_start frames).

    Used to bias-init the duration head's final layer to log(median+offset).
    Reported in beats because that's the unit the model is trained on
    (see `StepChartDataset.__getitem__` — the per-chunk duration target is
    converted with `seconds * tempo / 60`). A median of ~1.0 beat for DDR
    holds is the expected ballpark.
    """
    data_dir = Path(data_dir)
    idxs = indices if indices is not None else list(range(len(manifest)))
    if len(idxs) > sample_limit:
        step = max(1, len(idxs) // sample_limit)
        idxs = idxs[::step][:sample_limit]
    durations_beats: List[float] = []
    for i in idxs:
        entry = manifest[i]
        tempo = float(entry.get('tempo', 120.0)) or 120.0
        archive = np.load(data_dir / entry['filename'])
        labels = archive[entry['labels_key']]
        dur = archive[entry['durations_key']]
        # hold_start label = 3 in the v3 raw label encoding
        mask = labels == 3
        if mask.any():
            scale = tempo / 60.0
            durations_beats.extend(float(x) * scale for x in dur[mask] if x > 0)
    if not durations_beats:
        logger.warning("No holds found while computing duration median; using 2.0 beats.")
        return 2.0
    median = float(np.median(durations_beats))
    logger.info(
        f"Hold duration median: {median:.3f} beats "
        f"(min={min(durations_beats):.3f}, max={max(durations_beats):.3f}, n={len(durations_beats)})"
    )
    return median


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
