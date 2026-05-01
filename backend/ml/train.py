"""
Training script for step chart model (two-head onset-style).

Designed to run on Google Colab with a T4 GPU or similar. Uses:
    - mixed precision
    - by-song train/val split
    - note-weighted WeightedRandomSampler (more gradient on dense chunks)
    - BCE(onset) + CE(type) loss with empirical onset-prior bias init
    - per-step cosine scheduler with step-based warmup
    - tolerance-window F1 as the primary metric (± a few frames)
    - EMA weights for validation and best-model selection
    - best + last checkpoint only

All the heavy lifting lives in importable helpers so `train_colab.ipynb`
can reuse this exact code path (no duplicated training loop).

Usage (from Colab):
    !python -m ml.train \
        --data-dir /content/drive/MyDrive/stepageddon/training_data \
        --checkpoint-dir /content/drive/MyDrive/stepageddon/checkpoints \
        --epochs 80 --batch-size 32
"""

import argparse
import json
import logging
import math
import os
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Sampler, Subset

from ml.model import StepChartLoss, StepChartModel
from ml.dataset import (
    DENSITY_MEAN,
    DENSITY_STD,
    FRAMES_PER_SECOND,
    StepChartDataset,
    class_weights_from_prior,
    compute_beat_prior,
    compute_default_density_per_difficulty,
    compute_hold_duration_median,
    compute_note_counts,
    compute_note_type_counts,
    compute_onset_prior,
    compute_rare_chunk_index,
    compute_type_class_distribution,
    make_class_stratified_sampler,
    make_note_weighted_sampler,
    split_entries_by_song,
)


def make_mixup_collate(p_mixup: float, alpha: float):
    """Per-batch MixUp on mel + onset target, with type/duration masked out.

    The dataset yields (mel, diff, density, onset_soft, type_target, durations).
    With probability `p_mixup`, draw lambda ~ Beta(alpha, alpha), permute the
    batch, and replace mel/onset with the convex combination. Type and
    duration supervision is dropped for the mixed batch (set to -100 / 0):
    teaching the type head on mixed targets is ill-defined, but the onset
    BCE handles soft targets natively, so MixUp regularizes onset learning
    without poisoning the rare-class supervision the type head needs.

    p_mixup=0.0 returns a default collate (no-op).
    """
    from torch.utils.data._utils.collate import default_collate

    p_mixup = float(p_mixup)
    alpha = float(alpha)

    def collate(batch):
        out = default_collate(batch)
        if p_mixup <= 0.0 or torch.rand(()).item() >= p_mixup:
            return out
        mel, diff, dens, onset_soft, type_t, dur_t, beat_soft = out
        B = mel.size(0)
        if B < 2:
            return out
        lam = float(np.random.beta(alpha, alpha))
        lam = max(min(lam, 0.95), 0.05)  # keep both samples meaningfully present
        perm = torch.randperm(B)
        mel_mixed = lam * mel + (1.0 - lam) * mel[perm]
        onset_mixed = lam * onset_soft + (1.0 - lam) * onset_soft[perm]
        # Beats also mix proportionally — both songs' beat tracks remain
        # valid frame-level supervision under mixup.
        beat_mixed = lam * beat_soft + (1.0 - lam) * beat_soft[perm]
        type_masked = torch.full_like(type_t, -100)
        dur_zeroed = torch.zeros_like(dur_t)
        return mel_mixed, diff, dens, onset_mixed, type_masked, dur_zeroed, beat_mixed

    return collate


class MixedRareSampler(Sampler):
    """Yields full-dataset training indices, mixing two streams:

      - With probability p_rare, a (entry_idx, chunk_start) tuple drawn
        uniformly from the precomputed rare-frame index. The dataset's
        __getitem__ recognizes the tuple and returns a chunk *guaranteed*
        to contain a jump or hold event.
      - Otherwise, an int entry_idx drawn from the song-level stratified
        sampler (translated from subset-space to full-space).

    The song sampler stays as a backstop so tap-only sections still
    contribute gradient — we just stop letting them dominate the batch.
    """

    def __init__(
        self,
        song_sampler: Sampler,
        train_indices: List[int],
        rare_chunks: List[Tuple[int, int]],
        p_rare: float,
        num_samples: int,
        seed: int = 0,
    ):
        self._song_sampler = song_sampler
        self._train_indices = list(train_indices)
        self._rare_chunks = list(rare_chunks)
        self._p_rare = float(p_rare) if rare_chunks else 0.0
        self._num_samples = int(num_samples)
        self._epoch = 0
        self._seed = int(seed)

    def __iter__(self):
        # Re-derive a generator each epoch so multi-worker DataLoaders see a
        # fresh, deterministic stream that still varies between epochs.
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch * 0x9E3779B97F4A7C15 & 0xFFFFFFFF)
        self._epoch += 1
        coin = torch.rand(self._num_samples, generator=gen).tolist()
        rare_idx = (
            torch.randint(0, len(self._rare_chunks), (self._num_samples,), generator=gen).tolist()
            if self._rare_chunks else None
        )
        song_iter = iter(self._song_sampler)
        for k in range(self._num_samples):
            if coin[k] < self._p_rare:
                yield self._rare_chunks[rare_idx[k]]
            else:
                try:
                    sub = next(song_iter)
                except StopIteration:
                    song_iter = iter(self._song_sampler)
                    sub = next(song_iter)
                yield self._train_indices[int(sub)]

    def __len__(self):
        return self._num_samples

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'

@dataclass
class BuildOutputs:
    full_dataset: StepChartDataset
    train_loader: DataLoader
    val_loader: DataLoader
    train_idx: List[int]
    val_idx: List[int]
    note_counts: np.ndarray
    onset_prior: float
    default_density_by_id: torch.Tensor
    type_prior: np.ndarray  # [3] P(class | onset) over {tap, jump, hold_start}
    hold_duration_median: float  # beats; bias-init for the duration head
    beat_prior: float  # per-frame beat rate; bias-init for the beat head


def build_dataloaders(args) -> BuildOutputs:
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / 'manifest.json'

    # Compute the per-difficulty default density first so the dataset can
    # use it for the density-swap regularization (and we still save it on
    # the checkpoint for inference defaults).
    default_density_by_id = compute_default_density_per_difficulty(
        str(manifest_path), str(data_dir)
    )

    full_dataset = StepChartDataset(
        data_dir=str(data_dir),
        manifest_path=str(manifest_path),
        chunk_frames=args.chunk_frames,
        is_train=True,
        augment=True,
        density_swap_prob=float(getattr(args, 'p_density_swap', 0.5)),
        default_density_by_id=default_density_by_id,
    )

    # By-song split (non-leaky)
    train_idx, val_idx = split_entries_by_song(
        full_dataset.entries, val_fraction=args.val_split, seed=args.seed,
    )

    # Per-entry note counts (for weighted sampler + onset prior)
    note_counts = compute_note_counts(full_dataset.entries, str(data_dir))
    note_type_counts = compute_note_type_counts(
        full_dataset.entries, str(data_dir), train_idx,
    )
    onset_prior = compute_onset_prior(full_dataset.entries, str(data_dir), train_idx)
    type_prior = compute_type_class_distribution(
        full_dataset.entries, str(data_dir), train_idx,
    )
    hold_duration_median = compute_hold_duration_median(
        full_dataset.entries, str(data_dir), train_idx,
    )
    beat_prior = compute_beat_prior(
        full_dataset.entries, str(data_dir), train_idx,
    )

    # Training sampler: chunk-level mixed rare sampler.
    # The song-level stratified sampler still runs as the backstop stream
    # (so songs without holds aren't starved), but with probability
    # `p_rare` we draw from a precomputed list of (entry, chunk_start)
    # pairs that are guaranteed to contain a jump or hold — the only
    # reliable way to keep per-batch rare-class frame coverage above zero.
    song_sampler = make_class_stratified_sampler(
        train_idx, note_type_counts,
        jump_boost=getattr(args, 'jump_sample_boost', 8.0),
        hold_boost=getattr(args, 'hold_sample_boost', 16.0),
    )
    rare_chunks = compute_rare_chunk_index(
        full_dataset.entries, str(data_dir), train_idx, args.chunk_frames,
    )
    p_rare = float(getattr(args, 'p_rare', 0.5))
    if not rare_chunks:
        logger.warning("No rare-class chunks found in train split; falling back to song sampler only.")
        p_rare = 0.0
    train_sampler = MixedRareSampler(
        song_sampler=song_sampler,
        train_indices=train_idx,
        rare_chunks=rare_chunks,
        p_rare=p_rare,
        num_samples=len(train_idx),
        seed=args.seed,
    )

    # Validation: non-overlapping chunks, no augmentation
    val_base = StepChartDataset(
        data_dir=str(data_dir),
        manifest_path=str(manifest_path),
        chunk_frames=args.chunk_frames,
        is_train=False,
        augment=False,
    )
    val_entry_set = set(val_idx)
    val_base._val_chunks = [
        (i, s) for (i, s) in val_base._val_chunks if i in val_entry_set
    ]

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    # Use the full dataset (not a Subset) so MixedRareSampler can yield
    # tuple keys for the fixed-chunk rare path. The sampler restricts
    # output to train indices.
    mixup_p = float(getattr(args, 'mixup_p', 0.0))
    mixup_alpha = float(getattr(args, 'mixup_alpha', 0.4))
    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=make_mixup_collate(mixup_p, mixup_alpha),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_base,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    logger.info(
        f"Train entries: {len(train_idx)}, val entries: {len(val_idx)}, "
        f"val chunks: {len(val_base._val_chunks)}"
    )
    return BuildOutputs(
        full_dataset=full_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        train_idx=train_idx,
        val_idx=val_idx,
        note_counts=note_counts,
        onset_prior=onset_prior,
        default_density_by_id=default_density_by_id,
        type_prior=type_prior,
        hold_duration_median=hold_duration_median,
        beat_prior=beat_prior,
    )


def build_model(
    args,
    onset_prior: float,
    type_prior: np.ndarray,
    device: torch.device,
    hold_duration_median: float = 1.0,
    beat_prior: float = 0.02,
) -> StepChartModel:
    tcn_dilations = tuple(int(x) for x in args.tcn_dilations.split(','))
    model = StepChartModel(
        n_mels=80,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        n_difficulties=5,
        dropout=args.dropout,
        tcn_dilations=tcn_dilations,
        onset_head_layers=args.onset_head_layers,
    ).to(device)
    model.set_onset_prior(onset_prior)
    model.set_type_prior(type_prior)
    model.set_duration_prior(hold_duration_median)
    model.set_beat_prior(beat_prior)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    return model


def build_optimizer_and_scheduler(
    model: nn.Module, args, steps_per_epoch: int
):
    # Per-head LR multipliers. The type and duration heads receive far less
    # gradient signal per batch than the onset head (frame-masked, rare
    # classes), so a higher LR helps them catch up without destabilizing
    # onset training. Backbone + onset stay at base LR.
    head_mult = float(getattr(args, 'head_lr_mult', 1.0))
    base_lr = float(args.lr)
    backbone_params, onset_params, type_params, dur_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('type_head'):
            type_params.append(p)
        elif name.startswith('duration_head'):
            dur_params.append(p)
        elif name.startswith('onset_head'):
            onset_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = [
        {'params': backbone_params, 'lr': base_lr, 'name': 'backbone'},
        {'params': onset_params, 'lr': base_lr, 'name': 'onset_head'},
        {'params': type_params, 'lr': base_lr * head_mult, 'name': 'type_head'},
        {'params': dur_params, 'lr': base_lr * head_mult, 'name': 'duration_head'},
    ]
    logger.info(
        f"Optimizer param groups: backbone/onset lr={base_lr:.2e}, "
        f"type/duration lr={base_lr * head_mult:.2e} (head_lr_mult={head_mult:.2f})"
    )
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay,
    )
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def build_loss(
    onset_prior: float, type_prior: np.ndarray, args,
) -> StepChartLoss:
    p = max(onset_prior, 1e-5)
    if args.pos_weight is not None:
        pos_weight = float(args.pos_weight)
        logger.info(f"BCE pos_weight = {pos_weight:.2f} (override)")
    else:
        pos_weight = min(max(((1.0 - p) / p) ** 0.5, 1.0), 50.0)
        logger.info(
            f"BCE pos_weight = {pos_weight:.2f} (sqrt((1-p)/p) "
            f"from onset_prior={p:.4f})"
        )
    type_class_weights = class_weights_from_prior(
        type_prior,
        smoothing=args.type_weight_smoothing,
        max_weight=args.type_weight_cap,
    )
    return StepChartLoss(
        pos_weight=pos_weight,
        type_weight=args.type_weight,
        duration_weight=args.duration_weight,
        type_class_weights=type_class_weights,
        focal_gamma=args.focal_gamma,
        type_focal_gamma=getattr(args, 'type_focal_gamma', 0.0),
        beat_weight=getattr(args, 'beat_weight', 0.0),
    )

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device,
    ema_model: Optional[AveragedModel] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_onset = 0.0
    total_type = 0.0
    total_dur = 0.0
    total_beat = 0.0
    total_samples = 0
    n_batches = len(loader)

    # Running averages for intra-epoch logging
    interval_loss = 0.0
    interval_onset = 0.0
    interval_type = 0.0
    interval_dur = 0.0
    interval_beat = 0.0
    interval_samples = 0
    t_start = time.time()

    for step, batch in enumerate(loader):
        mel, difficulty, density, onset_soft, type_target, duration_target, beat_soft = batch
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        duration_target = duration_target.to(device, non_blocking=True)
        beat_soft = beat_soft.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            onset_logits, type_logits, duration_pred, beat_logits = model(
                mel, difficulty, density,
            )
            loss, onset_l, type_l, dur_l, beat_l = criterion(
                onset_logits, type_logits, duration_pred,
                onset_soft, type_target, duration_target,
                beat_logits=beat_logits, beat_soft=beat_soft,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        if ema_model is not None:
            ema_model.update_parameters(model)

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_onset += onset_l.item() * bs
        total_type += type_l.item() * bs
        total_dur += dur_l.item() * bs
        total_beat += beat_l.item() * bs
        total_samples += bs

        interval_loss += loss.item() * bs
        interval_onset += onset_l.item() * bs
        interval_type += type_l.item() * bs
        interval_dur += dur_l.item() * bs
        interval_beat += beat_l.item() * bs
        interval_samples += bs

        if (step + 1) % log_interval == 0 or (step + 1) == n_batches:
            avg_l = interval_loss / max(interval_samples, 1)
            avg_o = interval_onset / max(interval_samples, 1)
            avg_t = interval_type / max(interval_samples, 1)
            avg_d = interval_dur / max(interval_samples, 1)
            avg_b = interval_beat / max(interval_samples, 1)
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            logger.info(
                f"  step {step+1:4d}/{n_batches} "
                f"| loss={avg_l:.4f} (onset={avg_o:.4f} type={avg_t:.4f} "
                f"dur={avg_d:.4f} beat={avg_b:.4f}) "
                f"| lr={lr:.2e} | {elapsed:.1f}s"
            )
            interval_loss = 0.0
            interval_onset = 0.0
            interval_type = 0.0
            interval_dur = 0.0
            interval_beat = 0.0
            interval_samples = 0
            t_start = time.time()

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'duration_loss': total_dur / max(total_samples, 1),
        'beat_loss': total_beat / max(total_samples, 1),
    }


def _local_max_peaks(
    probs: np.ndarray, threshold: float, window: int
) -> np.ndarray:
    """Per-arrow 1-D NMS on a [T] probability vector. Returns peak indices."""
    T = probs.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int64)
    peaks = []
    last = -window - 1
    order = np.argsort(-probs)  # descending
    taken = np.zeros(T, dtype=bool)
    for idx in order:
        p = probs[idx]
        if p < threshold:
            break
        lo = max(0, idx - window)
        hi = min(T, idx + window + 1)
        if taken[lo:hi].any():
            continue
        taken[idx] = True
        peaks.append(int(idx))
    peaks.sort()
    return np.asarray(peaks, dtype=np.int64)


def tolerance_f1(
    pred_probs: np.ndarray,     # [T, 1] or [T]
    true_hard: np.ndarray,      # [T, 1] or [T] in {0,1}
    tol_frames: int = 3,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Greedy ±tol_frames matching on onset predictions. Returns (precision, recall, f1)."""
    # Flatten to 1-D
    pred_1d = pred_probs.reshape(-1)
    true_1d = true_hard.reshape(-1)

    pred_peaks = _local_max_peaks(pred_1d, threshold, tol_frames)
    true_peaks = np.where(true_1d > 0.5)[0]
    matched = np.zeros(len(true_peaks), dtype=bool)
    tp = 0
    for pf in pred_peaks:
        if len(true_peaks) == 0:
            break
        dists = np.abs(true_peaks - pf)
        valid = (dists <= tol_frames) & (~matched)
        if not valid.any():
            continue
        j = int(np.argmin(np.where(valid, dists, 10**9)))
        matched[j] = True
        tp += 1
    fp = len(pred_peaks) - tp
    fn = len(true_peaks) - tp
    if tp + fp + fn == 0:
        return 0.0, 0.0, 0.0
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


@torch.no_grad()
def validate(
    model, loader, criterion, device, tol_frames: int = 3,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_onset = 0.0
    total_type = 0.0
    total_dur = 0.0
    total_beat = 0.0
    total_samples = 0

    all_pred = []
    all_true = []

    type_correct = 0
    type_total = 0

    # Per-class confusion for {tap=0, jump=1, hold_start=2}.
    type_tp = np.zeros(3, dtype=np.int64)
    type_fp = np.zeros(3, dtype=np.int64)
    type_fn = np.zeros(3, dtype=np.int64)

    # Collected raw type logits + targets at supervised frames so we can
    # post-hoc sweep (jump_bias, hold_bias) for calibration. Cheap: only
    # ~2% of frames are supervised so this is a few MB total.
    all_type_logits: List[np.ndarray] = []
    all_type_targets: List[np.ndarray] = []

    dur_abs_errors = []

    for batch in loader:
        mel, difficulty, density, onset_soft, type_target, duration_target, beat_soft = batch
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        duration_target = duration_target.to(device, non_blocking=True)
        beat_soft = beat_soft.to(device, non_blocking=True)

        with autocast('cuda'):
            onset_logits, type_logits, duration_pred, beat_logits = model(
                mel, difficulty, density,
            )
            loss, onset_l, type_l, dur_l, beat_l = criterion(
                onset_logits, type_logits, duration_pred,
                onset_soft, type_target, duration_target,
                beat_logits=beat_logits, beat_soft=beat_soft,
            )

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_onset += onset_l.item() * bs
        total_type += type_l.item() * bs
        total_dur += dur_l.item() * bs
        total_beat += beat_l.item() * bs
        total_samples += bs

        probs = torch.sigmoid(onset_logits.float()).cpu().numpy()  # [B, T, 1]
        true_hard = (onset_soft.cpu().numpy() > 0.5).astype(np.float32)  # [B, T, 1]
        all_pred.append(probs)
        all_true.append(true_hard)

        # Type accuracy on frames with an onset
        type_pred = type_logits.argmax(dim=-1)  # [B, T]
        mask = type_target >= 0  # [B, T]
        if mask.any():
            tp_flat = type_pred[mask].cpu().numpy()
            tt_flat = type_target[mask].cpu().numpy()
            type_correct += int((tp_flat == tt_flat).sum())
            # Store the raw 3-way logits at supervised frames for the
            # post-hoc calibration sweep below.
            all_type_logits.append(
                type_logits[mask].float().cpu().numpy()
            )
            all_type_targets.append(tt_flat)
            type_total += int(tt_flat.size)
            for c in range(3):
                pred_c = (tp_flat == c)
                true_c = (tt_flat == c)
                type_tp[c] += int((pred_c & true_c).sum())
                type_fp[c] += int((pred_c & ~true_c).sum())
                type_fn[c] += int((~pred_c & true_c).sum())

        # Duration MAE on hold_start frames. Model output is log(beats+offset);
        # decode and report in *beats* — the unit the head was trained on after
        # the BPM-factoring change. 1 beat ≈ 0.5s at 120 BPM, so a beats MAE
        # of 0.25 corresponds to a 16th-note error at common tempos.
        hold_mask = (type_target == 2)  # hold_start class
        if hold_mask.any():
            pred_log = duration_pred[:, :, 0][hold_mask].float().cpu().numpy()
            pred_d = np.clip(StepChartModel.decode_duration(pred_log), 0.0, None)
            true_d = duration_target[hold_mask].float().cpu().numpy()
            dur_abs_errors.append(np.abs(pred_d - true_d))

    pred_concat = np.concatenate(all_pred, axis=0).reshape(-1, 1)
    true_concat = np.concatenate(all_true, axis=0).reshape(-1, 1)
    best_f1, best_thr, best_p, best_r = -1.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.05, 0.86, 0.05):
        p_, r_, f_ = tolerance_f1(
            pred_concat, true_concat,
            tol_frames=tol_frames, threshold=float(thr),
        )
        if f_ > best_f1:
            best_f1, best_thr, best_p, best_r = f_, float(thr), p_, r_
    prec, rec, f1 = best_p, best_r, best_f1

    # Multi-tolerance F1 at the primary best threshold. 50 ms / 100 ms windows
    # tell us how much of the recall gap is jitter (recoverable with a softer
    # match) vs genuine misses. 3/5/10 frames ≈ 30/50/100 ms at 100 fps.
    extra_tols: Dict[int, Tuple[float, float, float]] = {}
    for extra_tol in (5, 10):
        if extra_tol == tol_frames:
            extra_tols[extra_tol] = (best_p, best_r, best_f1)
            continue
        p_e, r_e, f_e = tolerance_f1(
            pred_concat, true_concat,
            tol_frames=extra_tol, threshold=float(best_thr),
        )
        extra_tols[extra_tol] = (p_e, r_e, f_e)

    pred_flat = pred_concat.reshape(-1)
    true_flat = true_concat.reshape(-1)
    pos_mask = true_flat > 0.5
    pred_max = float(pred_flat.max()) if pred_flat.size else 0.0
    pred_p99 = float(np.quantile(pred_flat, 0.99)) if pred_flat.size else 0.0
    pred_p999 = float(np.quantile(pred_flat, 0.999)) if pred_flat.size else 0.0
    if pos_mask.any():
        pos_probs = pred_flat[pos_mask]
        pos_max = float(pos_probs.max())
        pos_mean = float(pos_probs.mean())
        pos_p50 = float(np.quantile(pos_probs, 0.5))
    else:
        pos_max = pos_mean = pos_p50 = 0.0
    if (~pos_mask).any():
        neg_mean = float(pred_flat[~pos_mask].mean())
    else:
        neg_mean = 0.0

    dur_mae_beats = float(np.concatenate(dur_abs_errors).mean()) if dur_abs_errors else 0.0

    def _f1(tp, fp, fn):
        prec_c = tp / max(tp + fp, 1)
        rec_c = tp / max(tp + fn, 1)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
        return float(prec_c), float(rec_c), float(f1_c)

    tap_p, tap_r, tap_f1 = _f1(type_tp[0], type_fp[0], type_fn[0])
    jump_p, jump_r, jump_f1 = _f1(type_tp[1], type_fp[1], type_fn[1])
    hold_p, hold_r, hold_f1 = _f1(type_tp[2], type_fp[2], type_fn[2])
    type_macro_f1 = (tap_f1 + jump_f1 + hold_f1) / 3.0

    # Post-hoc per-class calibration: sweep additive logit biases for jump
    # and hold to maximize macro-F1 over the val set. The same biases are
    # what inference.py applies as `jump_bias` / `hold_bias`, so saving the
    # best values per epoch lets the inference pipeline ship a calibrated
    # default without manual tuning.
    cal_jump_bias = 0.0
    cal_hold_bias = 0.0
    cal_macro_f1 = type_macro_f1
    cal_tap_f1 = tap_f1
    cal_jump_f1 = jump_f1
    cal_hold_f1 = hold_f1
    if all_type_logits:
        logits_cat = np.concatenate(all_type_logits, axis=0)   # [N, 3]
        targets_cat = np.concatenate(all_type_targets, axis=0)  # [N]
        bias_grid = np.arange(-2.0, 2.001, 0.25)
        best = (cal_macro_f1, 0.0, 0.0)
        for jb in bias_grid:
            for hb in bias_grid:
                adj = logits_cat.copy()
                adj[:, 1] += jb
                adj[:, 2] += hb
                pred = adj.argmax(axis=-1)
                f1s = []
                per_class = []
                for c in range(3):
                    tp_c = int(((pred == c) & (targets_cat == c)).sum())
                    fp_c = int(((pred == c) & (targets_cat != c)).sum())
                    fn_c = int(((pred != c) & (targets_cat == c)).sum())
                    prec_c = tp_c / max(tp_c + fp_c, 1)
                    rec_c = tp_c / max(tp_c + fn_c, 1)
                    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
                    f1s.append(f1_c)
                    per_class.append(f1_c)
                macro = sum(f1s) / 3
                if macro > best[0]:
                    best = (macro, float(jb), float(hb))
                    cal_tap_f1, cal_jump_f1, cal_hold_f1 = per_class
        cal_macro_f1, cal_jump_bias, cal_hold_bias = best

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'duration_loss': total_dur / max(total_samples, 1),
        'beat_loss': total_beat / max(total_samples, 1),
        'tol_precision': prec,
        'tol_recall': rec,
        'tol_f1': f1,
        # Multi-tolerance variants at the primary threshold (visibility only;
        # they don't drive checkpoint selection).
        'tol_f1_50ms': float(extra_tols[5][2]),
        'tol_f1_100ms': float(extra_tols[10][2]),
        'best_thr': float(best_thr),
        'type_acc': type_correct / max(type_total, 1),
        'tap_f1': tap_f1, 'tap_p': tap_p, 'tap_r': tap_r,
        'jump_f1': jump_f1, 'jump_p': jump_p, 'jump_r': jump_r,
        'hold_f1': hold_f1, 'hold_p': hold_p, 'hold_r': hold_r,
        'type_macro_f1': type_macro_f1,
        # Calibrated equivalents — what inference would achieve with the
        # best (jump_bias, hold_bias) found via post-hoc bias sweep.
        'cal_jump_bias': cal_jump_bias,
        'cal_hold_bias': cal_hold_bias,
        'cal_macro_f1': cal_macro_f1,
        'cal_tap_f1': cal_tap_f1,
        'cal_jump_f1': cal_jump_f1,
        'cal_hold_f1': cal_hold_f1,
        'dur_mae_beats': dur_mae_beats,
        'prob_max': pred_max,
        'prob_p99': pred_p99,
        'prob_p999': pred_p999,
        'prob_pos_max': pos_max,
        'prob_pos_mean': pos_mean,
        'prob_pos_p50': pos_p50,
        'prob_neg_mean': neg_mean,
    }

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    ema_model: Optional[AveragedModel],
    optimizer,
    scheduler,
    scaler,
    metrics: Dict[str, float],
    default_density_by_id: torch.Tensor,
    mel_mean: float,
    mel_std: float,
    type_prior: np.ndarray,
    args,
) -> None:
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': (
            ema_model.module.state_dict() if ema_model is not None else None
        ),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        # Marker for inference: this checkpoint's duration head was trained
        # on beats targets (seconds * tempo / 60). Inference must convert
        # back to seconds via `beats * 60 / tempo`. Legacy checkpoints
        # without this flag are interpreted as 'seconds' for backward compat.
        'duration_unit': 'beats',
        'best_threshold': float(metrics.get('best_thr', 0.5)),
        # Per-class calibration biases — applied additively to the type
        # head's logits at inference. Saving them with the checkpoint means
        # MLChartGenerator gets calibrated defaults out of the box.
        'best_jump_bias': float(metrics.get('cal_jump_bias', 0.0)),
        'best_hold_bias': float(metrics.get('cal_hold_bias', 0.0)),
        'default_density_by_id': default_density_by_id.tolist(),
        'mel_mean': float(mel_mean),
        'mel_std': float(mel_std),
        'type_prior': [float(x) for x in type_prior],
        'args': vars(args) if not isinstance(args, dict) else args,
        'git_sha': _git_sha(),
    }
    torch.save(ckpt, path)

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train step chart model')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='ml/checkpoints')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--chunk-frames', type=int, default=500)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=float, default=2.0)
    parser.add_argument('--type-weight', type=float, default=4.0,
                        help='Multiplier on type-head CE loss; raised because '
                             'the type head is gradient-starved relative to onset '
                             'BCE — empirically the head collapses to all-tap '
                             'unless this is >=4 with aggressive class balancing')
    parser.add_argument('--duration-weight', type=float, default=1.0)
    parser.add_argument('--type-weight-smoothing', type=float, default=0.5,
                        help='Exponent for inverse-frequency type-class weights '
                             '(0=uniform, 1=raw inverse-frequency). 0.5 keeps '
                             'an order of magnitude of imbalance — required to '
                             'lift jump/hold above the dominant tap class.')
    parser.add_argument('--type-weight-cap', type=float, default=30.0,
                        help='Max per-class CE weight after smoothing')
    parser.add_argument('--type-focal-gamma', type=float, default=1.0,
                        help='Focal-loss gamma on the type CE. 0 disables; '
                             '1.0 down-weights confidently-correct frames so '
                             'rare classes contribute relatively more gradient.')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='Override BCE pos_weight on the onset head. '
                             'Default = sqrt((1-p)/p) capped at 50.')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal-loss gamma on the onset head BCE. '
                             '0 disables focal weighting; 2.0 is standard.')
    parser.add_argument('--jump-sample-boost', type=float, default=8.0,
                        help='Multiplier for jump-frame counts in the '
                             'class-stratified train sampler.')
    parser.add_argument('--hold-sample-boost', type=float, default=16.0,
                        help='Multiplier for hold_start-frame counts in the '
                             'class-stratified train sampler.')
    parser.add_argument('--p-rare', type=float, default=0.5,
                        help='Probability that a training chunk is drawn from '
                             'the rare-class anchor index (jump/hold guaranteed). '
                             '0.0 disables — falls back to the song-level sampler.')
    parser.add_argument('--p-density-swap', type=float, default=0.5,
                        help='Probability that the density input is swapped at '
                             'training time for the per-difficulty default '
                             '(the constant inference uses). 0.0 disables.')
    parser.add_argument('--mixup-p', type=float, default=0.3,
                        help='Per-batch probability of applying MixUp to mel + '
                             'onset target. Type/duration supervision is dropped '
                             'on mixed batches. 0.0 disables.')
    parser.add_argument('--mixup-alpha', type=float, default=0.4,
                        help='Beta(alpha, alpha) parameter for MixUp lambda.')
    parser.add_argument('--head-lr-mult', type=float, default=2.0,
                        help='LR multiplier applied to the type and duration '
                             'heads. Backbone + onset head stay at base LR. '
                             '1.0 disables the differential.')
    parser.add_argument('--beat-weight', type=float, default=0.3,
                        help='Loss weight on the auxiliary beat-prediction '
                             'head. 0.0 disables — head still emits logits but '
                             'gets no gradient.')
    parser.add_argument('--tcn-dilations', type=str, default='1,2,4,8,16,32',
                        help='Comma-separated dilations for the TCN stack')
    parser.add_argument('--onset-head-layers', type=int, default=3,
                        help='Number of hidden Linear→GELU blocks in the onset head')
    parser.add_argument('--plateau-patience', type=int, default=4,
                        help='Patience (epochs) for ReduceLROnPlateau on tol_f1')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='LR multiplier when ReduceLROnPlateau triggers')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tol-frames', type=int, default=3,
                        help='Tolerance window (in frames) for F1; ~30 ms at 100 fps')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Log training metrics every N batches within an epoch')
    return parser


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} | git_sha={_git_sha()} | seed={args.seed}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    built = build_dataloaders(args)
    model = build_model(
        args, built.onset_prior, built.type_prior, device,
        hold_duration_median=built.hold_duration_median,
        beat_prior=built.beat_prior,
    )
    criterion = build_loss(built.onset_prior, built.type_prior, args).to(device)
    steps_per_epoch = max(1, len(built.train_loader))
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=1e-6,
    )
    scaler = GradScaler('cuda')

    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)
    )

    start_epoch = 0
    best_composite = -1.0
    best_macro_f1 = -1.0
    best_dur_mae = float('inf')

    def _composite(metrics: Dict[str, float]) -> float:
        """Headline checkpoint score: balances onset timing (tol_f1) and
        the calibrated type macro-F1, so a tol_f1 spike that collapses jump
        and hold doesn't get crowned the best model."""
        tol = float(metrics.get('tol_f1', 0.0))
        cal = float(
            metrics.get('cal_macro_f1', metrics.get('type_macro_f1', 0.0))
        )
        return 0.6 * tol + 0.4 * cal

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('ema_state_dict') is not None:
            ema_model.module.load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_composite = _composite(ckpt.get('metrics', {}))

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, built.train_loader, criterion, optimizer, scheduler,
            scaler, device, ema_model=ema_model,
            log_interval=args.log_interval,
        )
        val_metrics = validate(
            ema_model.module, built.val_loader, criterion, device,
            tol_frames=args.tol_frames,
        )
        plateau_scheduler.step(val_metrics['tol_f1'])

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"(onset={train_metrics['onset_loss']:.4f} "
            f"type={train_metrics['type_loss']:.4f} "
            f"dur={train_metrics['duration_loss']:.4f}) "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"tol_f1={val_metrics['tol_f1']:.3f} "
            f"(P={val_metrics['tol_precision']:.3f} "
            f"R={val_metrics['tol_recall']:.3f} "
            f"thr={val_metrics['best_thr']:.2f}) "
            f"f1@50/100ms="
            f"{val_metrics['tol_f1_50ms']:.3f}/"
            f"{val_metrics['tol_f1_100ms']:.3f} "
            f"type_acc={val_metrics['type_acc']:.3f} "
            f"type_F1[tap/jump/hold]="
            f"{val_metrics['tap_f1']:.3f}/"
            f"{val_metrics['jump_f1']:.3f}/"
            f"{val_metrics['hold_f1']:.3f} "
            f"(macro={val_metrics['type_macro_f1']:.3f}) "
            f"dur_mae={val_metrics['dur_mae_beats']:.3f}b "
            f"| probs[max={val_metrics['prob_max']:.3f} "
            f"p99={val_metrics['prob_p99']:.3f} "
            f"p999={val_metrics['prob_p999']:.3f}] "
            f"pos[max={val_metrics['prob_pos_max']:.3f} "
            f"mean={val_metrics['prob_pos_mean']:.3f} "
            f"p50={val_metrics['prob_pos_p50']:.3f}] "
            f"neg_mean={val_metrics['prob_neg_mean']:.3f} "
            f"| lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Always overwrite 'last'
        save_checkpoint(
            checkpoint_dir / 'last_model.pt', epoch, model, ema_model,
            optimizer, scheduler, scaler, val_metrics,
            built.default_density_by_id,
            built.full_dataset.mel_mean, built.full_dataset.mel_std,
            built.type_prior,
            args,
        )

        composite = _composite(val_metrics)
        if composite > best_composite:
            best_composite = composite
            save_checkpoint(
                checkpoint_dir / 'best_model.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id,
                built.full_dataset.mel_mean, built.full_dataset.mel_std,
                built.type_prior,
                args,
            )
            logger.info(
                f"  -> New best composite={best_composite:.3f} "
                f"(tol_f1={val_metrics['tol_f1']:.3f} "
                f"cal_macro_f1={val_metrics.get('cal_macro_f1', val_metrics['type_macro_f1']):.3f} "
                f"thr={val_metrics['best_thr']:.2f})"
            )

        # Track separate "best by macro F1" and "best by duration MAE"
        # checkpoints. tol_f1 favors the onset head and would happily
        # discard a model that learned holds well at the cost of a tiny
        # onset regression — these split checkpoints prevent that.
        cal_macro = val_metrics.get('cal_macro_f1', val_metrics['type_macro_f1'])
        if cal_macro > best_macro_f1:
            best_macro_f1 = cal_macro
            save_checkpoint(
                checkpoint_dir / 'best_macro_f1.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id,
                built.full_dataset.mel_mean, built.full_dataset.mel_std,
                built.type_prior,
                args,
            )
            logger.info(f"  -> New best macro F1 = {best_macro_f1:.3f}")
        if val_metrics['dur_mae_beats'] > 0 and val_metrics['dur_mae_beats'] < best_dur_mae:
            best_dur_mae = val_metrics['dur_mae_beats']
            save_checkpoint(
                checkpoint_dir / 'best_dur_mae.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id,
                built.full_dataset.mel_mean, built.full_dataset.mel_std,
                built.type_prior,
                args,
            )
            logger.info(f"  -> New best dur MAE = {best_dur_mae:.3f}b")

    logger.info(
        f"Training complete. Best composite: {best_composite:.3f} "
        f"| macro_f1: {best_macro_f1:.3f} | dur_mae: {best_dur_mae:.3f}b"
    )


if __name__ == '__main__':
    main()
