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
    compute_arrow_priors,
    compute_beat_prior,
    compute_default_density_per_difficulty,
    compute_hold_duration_median,
    compute_hold_priors,
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
    """Per-batch MixUp on mel + per-arrow onset / hold targets, with type
    and dense duration masked out.

    The dataset yields (mel, diff, density, arrow_soft, type_target,
    remaining_beats, beat_soft, prev_arrow, start_seconds, remaining_seconds,
    in_hold_target). With probability `p_mixup`, draw lambda ~
    Beta(alpha, alpha), permute the batch, and replace mel / arrow_soft /
    beat_soft / in_hold_target / remaining_beats with the convex combination.
    Type supervision is dropped on mixed batches (set to -100): teaching the
    type head on mixed targets is ill-defined. The remaining_beats target is
    blended linearly because the dense duration loss masks by in_hold > 0.5;
    a blended in_hold may still trigger the mask, so we want the target to
    be a sensible interpolation rather than zeroed garbage. prev_arrow and
    the two song-position scalars are linearly mixed — same conditioning
    logic as the existing per-arrow blend.

    p_mixup=0.0 returns a default collate (no-op).
    """
    from torch.utils.data._utils.collate import default_collate

    p_mixup = float(p_mixup)
    alpha = float(alpha)

    def collate(batch):
        out = default_collate(batch)
        if p_mixup <= 0.0 or torch.rand(()).item() >= p_mixup:
            return out
        (
            mel, diff, dens, arrow_soft, type_t, dur_t,
            beat_soft, prev_arrow, start_s, remain_s, in_hold,
        ) = out
        B = mel.size(0)
        if B < 2:
            return out
        lam = float(np.random.beta(alpha, alpha))
        lam = max(min(lam, 0.95), 0.05)  # keep both samples meaningfully present
        perm = torch.randperm(B)
        mel_mixed = lam * mel + (1.0 - lam) * mel[perm]
        arrow_mixed = lam * arrow_soft + (1.0 - lam) * arrow_soft[perm]
        beat_mixed = lam * beat_soft + (1.0 - lam) * beat_soft[perm]
        prev_mixed = lam * prev_arrow + (1.0 - lam) * prev_arrow[perm]
        start_mixed = lam * start_s + (1.0 - lam) * start_s[perm]
        remain_mixed = lam * remain_s + (1.0 - lam) * remain_s[perm]
        in_hold_mixed = lam * in_hold + (1.0 - lam) * in_hold[perm]
        dur_blended = lam * dur_t + (1.0 - lam) * dur_t[perm]
        type_masked = torch.full_like(type_t, -100)
        return (
            mel_mixed, diff, dens, arrow_mixed, type_masked, dur_blended,
            beat_mixed, prev_mixed, start_mixed, remain_mixed, in_hold_mixed,
        )

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
    arrow_priors: np.ndarray  # [4] per-arrow positive rate; arrow head bias init
    default_density_by_id: torch.Tensor
    type_prior: np.ndarray  # [3] P(class | onset) over {tap, jump, hold_start}
    hold_duration_median: float  # beats; bias-init for the duration head
    beat_prior: float  # per-frame beat rate; bias-init for the beat head
    hold_priors: np.ndarray  # [4] per-arrow in-hold rate; hold head bias init


def build_dataloaders(args) -> BuildOutputs:
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / 'manifest.json'
    print(f"[build_dataloaders] data_dir={data_dir} manifest={manifest_path}", flush=True)

    print("[build_dataloaders] compute_default_density_per_difficulty...", flush=True)
    default_density_by_id = compute_default_density_per_difficulty(
        str(manifest_path), str(data_dir)
    )
    print("[build_dataloaders] default_density done", flush=True)

    print("[build_dataloaders] constructing StepChartDataset (train)...", flush=True)
    full_dataset = StepChartDataset(
        data_dir=str(data_dir),
        manifest_path=str(manifest_path),
        chunk_frames=args.chunk_frames,
        is_train=True,
        augment=True,
        density_swap_prob=float(getattr(args, 'p_density_swap', 0.5)),
        default_density_by_id=default_density_by_id,
        intro_outro_oversample_prob=float(
            getattr(args, 'intro_outro_oversample_prob', 0.15)
        ),
    )
    print(f"[build_dataloaders] dataset built, n_entries={len(full_dataset.entries)}", flush=True)

    print("[build_dataloaders] split_entries_by_song...", flush=True)
    train_idx, val_idx = split_entries_by_song(
        full_dataset.entries, val_fraction=args.val_split, seed=args.seed,
    )
    print(f"[build_dataloaders] split: train={len(train_idx)} val={len(val_idx)}", flush=True)

    # Per-difficulty filter: restrict both train and val indices to entries
    # whose difficulty_id matches `args.difficulty`. All downstream priors
    # (note counts, arrow priors, type priors, beat priors, rare-chunk index)
    # are computed from these filtered indices, so the loss / sampler
    # naturally specialize to this difficulty without further plumbing.
    if getattr(args, 'difficulty', None) is not None:
        target = int(args.difficulty)
        train_idx = [
            i for i in train_idx
            if int(full_dataset.entries[i]['difficulty_id']) == target
        ]
        val_idx = [
            i for i in val_idx
            if int(full_dataset.entries[i]['difficulty_id']) == target
        ]
        if not train_idx:
            raise SystemExit(
                f"--difficulty={target} filter produced 0 train entries; "
                f"check the manifest's difficulty_id distribution."
            )
        logger.info(
            f"--difficulty={target} filter: train={len(train_idx)} "
            f"val={len(val_idx)} (after restricting to one class)"
        )

    print("[build_dataloaders] compute_note_counts...", flush=True)
    note_counts = compute_note_counts(full_dataset.entries, str(data_dir))
    print("[build_dataloaders] compute_note_type_counts...", flush=True)
    note_type_counts = compute_note_type_counts(
        full_dataset.entries, str(data_dir), train_idx,
    )
    print("[build_dataloaders] compute_onset_prior...", flush=True)
    onset_prior = compute_onset_prior(full_dataset.entries, str(data_dir), train_idx)
    print("[build_dataloaders] compute_arrow_priors...", flush=True)
    arrow_priors = compute_arrow_priors(
        full_dataset.entries, str(data_dir), train_idx,
    )
    print("[build_dataloaders] compute_type_class_distribution...", flush=True)
    type_prior = compute_type_class_distribution(
        full_dataset.entries, str(data_dir), train_idx,
    )
    print("[build_dataloaders] compute_hold_duration_median...", flush=True)
    hold_duration_median = compute_hold_duration_median(
        full_dataset.entries, str(data_dir), train_idx,
    )
    print("[build_dataloaders] compute_beat_prior...", flush=True)
    beat_prior = compute_beat_prior(
        full_dataset.entries, str(data_dir), train_idx,
    )
    print("[build_dataloaders] compute_hold_priors...", flush=True)
    hold_priors = compute_hold_priors(
        full_dataset.entries, str(data_dir), train_idx,
    )
    print("[build_dataloaders] all priors done", flush=True)

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
    print("[build_dataloaders] compute_rare_chunk_index...", flush=True)
    rare_chunks = compute_rare_chunk_index(
        full_dataset.entries, str(data_dir), train_idx, args.chunk_frames,
    )
    print(f"[build_dataloaders] rare_chunks={len(rare_chunks)}", flush=True)
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

    print("[build_dataloaders] constructing StepChartDataset (val)...", flush=True)
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
    print(f"[build_dataloaders] val dataset built, val_chunks={len(val_base._val_chunks)}", flush=True)

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
    print("[build_dataloaders] constructing train DataLoader...", flush=True)
    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=make_mixup_collate(mixup_p, mixup_alpha),
        **loader_kwargs,
    )
    print("[build_dataloaders] constructing val DataLoader...", flush=True)
    val_loader = DataLoader(
        val_base,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    print("[build_dataloaders] done", flush=True)

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
        arrow_priors=arrow_priors,
        default_density_by_id=default_density_by_id,
        type_prior=type_prior,
        hold_duration_median=hold_duration_median,
        beat_prior=beat_prior,
        hold_priors=hold_priors,
    )


def build_model(
    args,
    arrow_priors: np.ndarray,
    type_prior: np.ndarray,
    device: torch.device,
    hold_duration_median: float = 1.0,
    beat_prior: float = 0.02,
    hold_priors: Optional[np.ndarray] = None,
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
    model.set_arrow_priors(arrow_priors)
    model.set_type_prior(type_prior)
    model.set_duration_prior(hold_duration_median)
    model.set_beat_prior(beat_prior)
    if hold_priors is not None:
        model.set_hold_priors(hold_priors)
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
    backbone_params, arrow_params, type_params, dur_params, hold_params = (
        [], [], [], [], [],
    )
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('type_head'):
            type_params.append(p)
        elif name.startswith('duration_head'):
            dur_params.append(p)
        elif name.startswith('arrow_head'):
            arrow_params.append(p)
        elif name.startswith('hold_head'):
            # Hold supervision is dense (~3% per arrow per frame) — much
            # denser than the type/duration heads' frame-masked CE/L1 — so
            # head_lr_mult would overshoot. Keep hold at base_lr alongside
            # arrow (the other dense-supervised head) and out of the
            # backbone bucket so its membership is explicit.
            hold_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = [
        {'params': backbone_params, 'lr': base_lr, 'name': 'backbone'},
        {'params': arrow_params, 'lr': base_lr, 'name': 'arrow_head'},
        {'params': hold_params, 'lr': base_lr, 'name': 'hold_head'},
        {'params': type_params, 'lr': base_lr * head_mult, 'name': 'type_head'},
        {'params': dur_params, 'lr': base_lr * head_mult, 'name': 'duration_head'},
    ]
    logger.info(
        f"Optimizer param groups: backbone/arrow/hold lr={base_lr:.2e}, "
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
    arrow_priors: np.ndarray,
    type_prior: np.ndarray,
    args,
    hold_priors: Optional[np.ndarray] = None,
) -> StepChartLoss:
    """Build the combined loss with per-arrow pos_weight derived from priors.

    Per-arrow pos_weight = sqrt((1 - p_a) / p_a) per arrow, capped at 50.
    Computing it per arrow (instead of a single onset rate) prevents the
    model from leaning on whichever arrow happens to be most frequent —
    pos_weight is highest for the rarest arrow, so its positives matter most.
    `--pos-weight` (scalar) still overrides if set, broadcast across arrows.
    """
    if args.pos_weight is not None:
        pw = torch.full((4,), float(args.pos_weight), dtype=torch.float32)
        logger.info(f"BCE pos_weight = {float(args.pos_weight):.2f} (override, all arrows)")
    else:
        clipped = np.clip(arrow_priors, 1e-5, 0.5)
        per_arrow = np.sqrt((1.0 - clipped) / clipped)
        per_arrow = np.minimum(np.maximum(per_arrow, 1.0), 50.0)
        pw = torch.from_numpy(per_arrow.astype(np.float32))
        logger.info(
            f"BCE pos_weight per arrow [L,D,U,R] = {per_arrow.tolist()} "
            f"(sqrt((1-p)/p) from arrow_priors)"
        )
    type_class_weights = class_weights_from_prior(
        type_prior,
        smoothing=args.type_weight_smoothing,
        max_weight=args.type_weight_cap,
    )
    # Hold pos_weight: explicit CLI override first, else derive per-arrow
    # from the in-hold prior with the same sqrt((1-p)/p) recipe used for the
    # arrow head (capped lower because in-hold rates are ~3% — much higher
    # than the per-arrow onset rate — so the raw weight is already smaller).
    hpw_arg = float(getattr(args, 'hold_pos_weight', 0.0))
    if hpw_arg > 0.0:
        hpw = torch.full((4,), hpw_arg, dtype=torch.float32)
        logger.info(
            f"Hold-head pos_weight = {hpw_arg:.2f} (override, all arrows)"
        )
    elif hold_priors is not None:
        hp_clipped = np.clip(np.asarray(hold_priors), 1e-4, 0.5)
        per_arrow_hpw = np.sqrt((1.0 - hp_clipped) / hp_clipped)
        per_arrow_hpw = np.minimum(np.maximum(per_arrow_hpw, 1.0), 20.0)
        hpw = torch.from_numpy(per_arrow_hpw.astype(np.float32))
        logger.info(
            f"Hold-head pos_weight per arrow [L,D,U,R] = "
            f"{per_arrow_hpw.tolist()} (sqrt((1-p)/p) from hold_priors)"
        )
    else:
        hpw = torch.full((4,), 5.0, dtype=torch.float32)
        logger.info("Hold-head pos_weight = 5.0 (uniform fallback)")
    jack_strides_arg = getattr(args, 'jack_strides', '3,6,9')
    if isinstance(jack_strides_arg, str):
        jack_strides = tuple(
            int(s.strip()) for s in jack_strides_arg.split(',') if s.strip()
        )
    else:
        jack_strides = tuple(int(x) for x in jack_strides_arg)
    return StepChartLoss(
        pos_weight=pw,
        type_weight=args.type_weight,
        duration_weight=args.duration_weight,
        type_class_weights=type_class_weights,
        focal_gamma=args.focal_gamma,
        type_focal_gamma=getattr(args, 'type_focal_gamma', 0.0),
        beat_weight=getattr(args, 'beat_weight', 0.0),
        diversity_weight=getattr(args, 'diversity_weight', 0.2),
        commit_weight=getattr(args, 'arrow_commit_weight', 0.5),
        jack_weight=getattr(args, 'jack_weight', 0.3),
        jack_strides=jack_strides,
        jack_stride_decay=float(getattr(args, 'jack_stride_decay', 0.5)),
        jump_cofire_weight=float(getattr(args, 'jump_cofire_weight', 0.5)),
        hold_weight=float(getattr(args, 'hold_weight', 0.3)),
        hold_pos_weight=hpw,
    )

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device,
    ema_model: Optional[AveragedModel] = None,
    log_interval: int = 50,
    prev_arrow_dropout: float = 0.1,
) -> Dict[str, float]:
    """One epoch of training.

    `prev_arrow_dropout` zeros the entire prev_arrow vector for a fraction of
    frames per chunk, so the arrow head learns to make sensible predictions
    even with no transition context (matches inference at the start of a
    song or after long silences). Independent Bernoulli mask per (B, T).
    """
    print("  [train_one_epoch] model.train() done; about to iterate loader", flush=True)
    model.train()
    total_loss = 0.0
    total_arrow = 0.0
    total_type = 0.0
    total_dur = 0.0
    total_beat = 0.0
    total_div = 0.0
    total_hold = 0.0
    total_samples = 0
    n_batches = len(loader)
    print(f"  [train_one_epoch] n_batches={n_batches}", flush=True)

    interval_loss = 0.0
    interval_arrow = 0.0
    interval_type = 0.0
    interval_dur = 0.0
    interval_beat = 0.0
    interval_div = 0.0
    interval_hold = 0.0
    interval_samples = 0
    t_start = time.time()

    _t_fetch = time.time()
    for step, batch in enumerate(loader):
        if step < 3:
            print(
                f"  [train_one_epoch] got batch {step} "
                f"(fetch={time.time()-_t_fetch:.2f}s)",
                flush=True,
            )
        (
            mel, difficulty, density, arrow_soft, type_target, duration_target,
            beat_soft, prev_arrow, start_seconds, remaining_seconds, in_hold_target,
        ) = batch
        if step < 3:
            print(f"  [train_one_epoch] batch {step} unpacked, mel={tuple(mel.shape)}", flush=True)
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        arrow_soft = arrow_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        duration_target = duration_target.to(device, non_blocking=True)
        beat_soft = beat_soft.to(device, non_blocking=True)
        prev_arrow = prev_arrow.to(device, non_blocking=True)
        start_seconds = start_seconds.to(device, non_blocking=True)
        remaining_seconds = remaining_seconds.to(device, non_blocking=True)
        in_hold_target = in_hold_target.to(device, non_blocking=True)

        # Per-frame prev-arrow dropout: zero the conditioning vector with
        # probability `prev_arrow_dropout`. Forces the head to remain useful
        # without conditioning context, matching inference at song start or
        # after long gaps where there's no recent prior arrow.
        if prev_arrow_dropout > 0.0:
            keep = (torch.rand(prev_arrow.shape[:2], device=device)
                    >= prev_arrow_dropout).float().unsqueeze(-1)
            prev_arrow = prev_arrow * keep

        optimizer.zero_grad(set_to_none=True)
        if step < 3:
            print(f"  [train_one_epoch] batch {step} forward...", flush=True)
        with autocast('cuda'):
            arrow_logits, type_logits, duration_pred, beat_logits, hold_logits = model(
                mel, difficulty, density,
                start_seconds, remaining_seconds, prev_arrow,
            )
            loss, arrow_l, type_l, dur_l, beat_l, div_l, hold_l = criterion(
                arrow_logits, type_logits, duration_pred,
                arrow_soft, type_target, duration_target,
                beat_logits=beat_logits, beat_soft=beat_soft,
                hold_logits=hold_logits, in_hold_target=in_hold_target,
            )
        if step < 3:
            print(f"  [train_one_epoch] batch {step} loss={loss.item():.4f} backward...", flush=True)

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
        if step < 3:
            print(f"  [train_one_epoch] batch {step} step done", flush=True)
        _t_fetch = time.time()

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_arrow += arrow_l.item() * bs
        total_type += type_l.item() * bs
        total_dur += dur_l.item() * bs
        total_beat += beat_l.item() * bs
        total_div += div_l.item() * bs
        total_hold += hold_l.item() * bs
        total_samples += bs

        interval_loss += loss.item() * bs
        interval_arrow += arrow_l.item() * bs
        interval_type += type_l.item() * bs
        interval_dur += dur_l.item() * bs
        interval_beat += beat_l.item() * bs
        interval_div += div_l.item() * bs
        interval_hold += hold_l.item() * bs
        interval_samples += bs

        if (step + 1) % log_interval == 0 or (step + 1) == n_batches:
            avg_l = interval_loss / max(interval_samples, 1)
            avg_a = interval_arrow / max(interval_samples, 1)
            avg_t = interval_type / max(interval_samples, 1)
            avg_d = interval_dur / max(interval_samples, 1)
            avg_b = interval_beat / max(interval_samples, 1)
            avg_v = interval_div / max(interval_samples, 1)
            avg_h = interval_hold / max(interval_samples, 1)
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            print(
                f"  step {step+1:4d}/{n_batches} "
                f"| loss={avg_l:.4f} (arrow={avg_a:.4f} type={avg_t:.4f} "
                f"dur={avg_d:.4f} beat={avg_b:.4f} div={avg_v:.4f} "
                f"hold={avg_h:.4f}) "
                f"| lr={lr:.2e} | {elapsed:.1f}s",
                flush=True,
            )
            interval_loss = 0.0
            interval_arrow = 0.0
            interval_type = 0.0
            interval_dur = 0.0
            interval_beat = 0.0
            interval_div = 0.0
            interval_hold = 0.0
            interval_samples = 0
            t_start = time.time()

    return {
        'loss': total_loss / max(total_samples, 1),
        'arrow_loss': total_arrow / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'duration_loss': total_dur / max(total_samples, 1),
        'beat_loss': total_beat / max(total_samples, 1),
        'diversity_loss': total_div / max(total_samples, 1),
        'hold_loss': total_hold / max(total_samples, 1),
    }


def _all_local_max_peaks(
    probs: np.ndarray, window: int, min_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized 1-D NMS. Returns (peak_indices, peak_probs) sorted by index.

    A frame is a peak iff its prob equals the max over a ±window neighborhood
    AND exceeds `min_threshold`. Plateau ties are broken by keeping the earliest
    occurrence (then re-suppressing within `window`). Uses scipy's C-implemented
    1-D max filter — orders of magnitude faster than the previous Python loop
    over all frames, which dominated validation time on large val sets.
    """
    T = probs.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=probs.dtype)
    from scipy.ndimage import maximum_filter1d
    pooled = maximum_filter1d(probs, size=2 * window + 1, mode='constant', cval=-np.inf)
    is_peak = (probs == pooled) & (probs >= min_threshold)
    idx = np.flatnonzero(is_peak)
    if idx.size <= 1:
        return idx.astype(np.int64), probs[idx]
    # Plateau handling: drop subsequent peaks whose distance to the prior kept
    # peak is <= window. Mirrors the original greedy-by-prob behavior closely
    # enough for F1 — exact equivalence isn't worth the cost.
    keep = np.ones(idx.size, dtype=bool)
    last_kept = idx[0]
    for k in range(1, idx.size):
        if idx[k] - last_kept <= window:
            if probs[idx[k]] > probs[last_kept]:
                keep[k - 1] = False
                last_kept = idx[k]
            else:
                keep[k] = False
        else:
            last_kept = idx[k]
    idx = idx[keep]
    return idx.astype(np.int64), probs[idx]


def _f1_at_threshold(
    peak_idx: np.ndarray,
    peak_probs: np.ndarray,
    true_peaks: np.ndarray,
    tol_frames: int,
    threshold: float,
) -> Tuple[float, float, float]:
    """Greedy ±tol_frames matching given precomputed peaks. O(P + T) via two
    pointers on sorted index arrays."""
    sel = peak_probs >= threshold
    pred_peaks = peak_idx[sel]
    if pred_peaks.size == 0 and true_peaks.size == 0:
        return 0.0, 0.0, 0.0
    matched_true = np.zeros(true_peaks.size, dtype=bool)
    tp = 0
    j_start = 0
    for pf in pred_peaks:
        # Advance j_start past true peaks that are too far below pf.
        while j_start < true_peaks.size and true_peaks[j_start] < pf - tol_frames:
            j_start += 1
        # Search forward within the window for an unmatched true peak.
        j = j_start
        best_j = -1
        best_d = tol_frames + 1
        while j < true_peaks.size and true_peaks[j] <= pf + tol_frames:
            if not matched_true[j]:
                d = abs(int(true_peaks[j]) - int(pf))
                if d < best_d:
                    best_d = d
                    best_j = j
            j += 1
        if best_j >= 0:
            matched_true[best_j] = True
            tp += 1
    fp = pred_peaks.size - tp
    fn = true_peaks.size - tp
    if tp + fp + fn == 0:
        return 0.0, 0.0, 0.0
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def tolerance_f1(
    pred_probs: np.ndarray,     # [T, 1] or [T]
    true_hard: np.ndarray,      # [T, 1] or [T] in {0,1}
    tol_frames: int = 3,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Greedy ±tol_frames matching on onset predictions. Returns (P, R, F1)."""
    pred_1d = pred_probs.reshape(-1)
    true_1d = true_hard.reshape(-1)
    peak_idx, peak_probs = _all_local_max_peaks(pred_1d, tol_frames, min_threshold=threshold)
    true_peaks = np.flatnonzero(true_1d > 0.5)
    return _f1_at_threshold(peak_idx, peak_probs, true_peaks, tol_frames, threshold)


@torch.no_grad()
def validate(
    model, loader, criterion, device, tol_frames: int = 3,
) -> Dict[str, float]:
    print("  [validate] model.eval()...", flush=True)
    model.eval()
    n_val_batches = len(loader)
    print(f"  [validate] n_val_batches={n_val_batches}", flush=True)
    total_loss = 0.0
    total_arrow = 0.0
    total_type = 0.0
    total_dur = 0.0
    total_beat = 0.0
    total_div = 0.0
    total_hold = 0.0
    total_samples = 0
    # Per-arrow hold-head probabilities and targets, accumulated across the
    # val set so we can sweep a per-arrow threshold for raw hold F1 — the
    # diagnostic that tells us the new dense supervision is actually being
    # absorbed by the head (vs. just collapsing to the prior bias).
    all_hold_probs: List[np.ndarray] = []
    all_hold_targets: List[np.ndarray] = []

    # Aggregate "any-onset" prob per frame via row-max on per-arrow probs.
    all_pred = []
    all_true = []

    # Per-arrow predictions/targets for per-arrow F1 calibration.
    # `all_arrow_probs` is zero prev_arrow context — matches inference Phase 1
    # (NMS peak picking). `all_arrow_probs_tf` is teacher-forced — matches
    # inference Phase 2 (per-onset arrow re-scoring with the previous emitted
    # arrow as context). The per-arrow threshold sweep below uses TF, the
    # any-onset threshold sweep uses zero-context.
    all_arrow_probs = []      # list of [B, T, 4] (prev_arrow=0)
    all_arrow_probs_tf = []   # list of [B, T, 4] (teacher-forced prev_arrow)
    all_arrow_true = []       # list of [B, T, 4]

    type_correct = 0
    type_total = 0

    type_tp = np.zeros(3, dtype=np.int64)
    type_fp = np.zeros(3, dtype=np.int64)
    type_fn = np.zeros(3, dtype=np.int64)

    all_type_logits: List[np.ndarray] = []
    all_type_targets: List[np.ndarray] = []

    dur_abs_errors = []

    _t_val = time.time()
    for v_step, batch in enumerate(loader):
        if v_step < 3 or (v_step + 1) % 25 == 0 or (v_step + 1) == n_val_batches:
            print(
                f"  [validate] batch {v_step+1}/{n_val_batches} "
                f"({time.time()-_t_val:.1f}s elapsed)",
                flush=True,
            )
        (
            mel, difficulty, density, arrow_soft, type_target, duration_target,
            beat_soft, prev_arrow, start_seconds, remaining_seconds, in_hold_target,
        ) = batch
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        arrow_soft = arrow_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        duration_target = duration_target.to(device, non_blocking=True)
        beat_soft = beat_soft.to(device, non_blocking=True)
        prev_arrow = prev_arrow.to(device, non_blocking=True)
        start_seconds = start_seconds.to(device, non_blocking=True)
        remaining_seconds = remaining_seconds.to(device, non_blocking=True)
        in_hold_target = in_hold_target.to(device, non_blocking=True)

        with autocast('cuda'):
            # Share the backbone encode across the two arrow-head calls below.
            features = model.encode(
                mel, difficulty, density, start_seconds, remaining_seconds,
            )
            # Teacher-forced arrow logits: used for the loss (matches training)
            # and per-frame loss reporting.
            arrow_logits = model.apply_arrow_head(features, prev_arrow)
            beat_logits = model.apply_beat_head(features)
            # Zero-context arrow logits: matches the inference peak-picking
            # distribution (`_predict_chunked` calls apply_arrow_head with
            # prev_arrow=None, i.e. zeros). The threshold sweep below uses
            # these so saved best_threshold / best_arrow_thresholds are
            # calibrated against the distribution inference actually sees.
            # The hold head, type head, and duration head are all conditioned
            # on these zero-context arrow logits, matching inference.
            zero_prev = torch.zeros_like(prev_arrow)
            arrow_logits_zero = model.apply_arrow_head(features, zero_prev)
            hold_logits = model.apply_hold_head(features, arrow_logits_zero)
            type_logits = model.apply_type_head(
                features, arrow_logits_zero, beat_logits, hold_logits,
            )
            duration_pred = model.apply_duration_head(
                features, arrow_logits_zero, beat_logits, hold_logits,
            )
            loss, arrow_l, type_l, dur_l, beat_l, div_l, hold_l = criterion(
                arrow_logits, type_logits, duration_pred,
                arrow_soft, type_target, duration_target,
                beat_logits=beat_logits, beat_soft=beat_soft,
                hold_logits=hold_logits, in_hold_target=in_hold_target,
            )

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_arrow += arrow_l.item() * bs
        total_type += type_l.item() * bs
        total_dur += dur_l.item() * bs
        total_beat += beat_l.item() * bs
        total_div += div_l.item() * bs
        total_hold += hold_l.item() * bs
        total_samples += bs

        # Stash hold-head probs and targets for the per-arrow F1 sweep
        # below. Hold supervision is dense (in-hold spans), so we don't need
        # NMS — a plain per-frame threshold sweep is the right diagnostic.
        all_hold_probs.append(
            torch.sigmoid(hold_logits.float()).cpu().numpy()
        )
        all_hold_targets.append(in_hold_target.cpu().numpy())

        # Any-onset threshold sweep / KL / stream_coh use the zero-context
        # distribution (matches inference Phase 1 peak picking).
        arrow_probs = torch.sigmoid(arrow_logits_zero.float()).cpu().numpy()  # [B, T, 4]
        # Per-arrow threshold sweep uses the teacher-forced distribution
        # (matches inference Phase 2, where prev_arrow tracks the previously
        # emitted arrow per-onset).
        arrow_probs_tf = torch.sigmoid(arrow_logits.float()).cpu().numpy()    # [B, T, 4]
        arrow_true = (arrow_soft.cpu().numpy() > 0.5).astype(np.float32)   # [B, T, 4]
        all_arrow_probs.append(arrow_probs)
        all_arrow_probs_tf.append(arrow_probs_tf)
        all_arrow_true.append(arrow_true)
        # Aggregate "any-onset" probability is the max over the 4 arrows.
        any_probs = arrow_probs.max(axis=-1, keepdims=True)                # [B, T, 1]
        any_true = arrow_true.max(axis=-1, keepdims=True)                  # [B, T, 1]
        all_pred.append(any_probs)
        all_true.append(any_true)

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

    print("  [validate] val loop done; concatenating preds...", flush=True)
    pred_concat = np.concatenate(all_pred, axis=0).reshape(-1, 1)
    true_concat = np.concatenate(all_true, axis=0).reshape(-1, 1)
    arrow_probs_concat = np.concatenate(all_arrow_probs, axis=0).reshape(-1, 4)  # [N, 4] zero-context
    arrow_probs_tf_concat = np.concatenate(all_arrow_probs_tf, axis=0).reshape(-1, 4)  # [N, 4] teacher-forced
    arrow_true_concat = np.concatenate(all_arrow_true, axis=0).reshape(-1, 4)    # [N, 4]
    pred_1d = pred_concat.reshape(-1)
    true_1d_hard = true_concat.reshape(-1)
    true_peaks = np.flatnonzero(true_1d_hard > 0.5)
    thresholds = np.arange(0.05, 0.86, 0.05)
    min_thr = float(thresholds.min())
    print(
        f"  [validate] computing peaks once at min_thr={min_thr:.2f} over "
        f"{pred_1d.shape[0]} frames...", flush=True,
    )
    peak_idx, peak_probs = _all_local_max_peaks(
        pred_1d, tol_frames, min_threshold=min_thr,
    )
    print(f"  [validate] {peak_idx.size} candidate peaks; sweeping {thresholds.size} thresholds...", flush=True)
    best_f1, best_thr, best_p, best_r = -1.0, 0.5, 0.0, 0.0
    for thr in thresholds:
        p_, r_, f_ = _f1_at_threshold(
            peak_idx, peak_probs, true_peaks, tol_frames, float(thr),
        )
        if f_ > best_f1:
            best_f1, best_thr, best_p, best_r = f_, float(thr), p_, r_
    prec, rec, f1 = best_p, best_r, best_f1

    # Per-arrow threshold sweep + F1. Each arrow is treated as an independent
    # frame-level binary problem so the model can be calibrated separately
    # per channel (outer arrows fire ~2× as often as D/U; one shared
    # threshold under-fits the rarer ones). Uses the teacher-forced probs
    # because inference Phase 2 (per-onset arrow re-scoring in
    # `_choose_arrows`) sees the real previously-emitted arrow as context;
    # calibrating against the zero-context distribution would make these
    # thresholds far too conservative for that path.
    per_arrow_best_thr = np.zeros(4, dtype=np.float32)
    per_arrow_f1 = np.zeros(4, dtype=np.float32)
    for a in range(4):
        ap = arrow_probs_tf_concat[:, a]
        at = arrow_true_concat[:, a]
        true_a_peaks = np.flatnonzero(at > 0.5)
        peak_a_idx, peak_a_probs = _all_local_max_peaks(
            ap, tol_frames, min_threshold=min_thr,
        )
        best_af1, best_athr = -1.0, 0.5
        for thr in thresholds:
            _, _, fa = _f1_at_threshold(
                peak_a_idx, peak_a_probs, true_a_peaks, tol_frames, float(thr),
            )
            if fa > best_af1:
                best_af1, best_athr = fa, float(thr)
        per_arrow_best_thr[a] = best_athr
        per_arrow_f1[a] = max(best_af1, 0.0)

    # Stream coherence on the predicted arrow argmax in dense regions.
    # In any 1-second window with >= 4 onsets (predicted by max-arrow > thr),
    # the fraction of consecutive non-equal arrows. ~0.7-0.85 on real charts.
    stream_coh = _stream_coherence(
        arrow_probs_concat, threshold=float(best_thr),
        min_dense_count=4, frames_per_second=int(round(FRAMES_PER_SECOND)),
    )
    # Per-arrow KL divergence between predicted and true marginals over the
    # *positive* frames. Punishes mode collapse: KL is 0 only when the
    # predicted arrow distribution matches the ground-truth distribution
    # over note-bearing frames.
    arrow_kl = _arrow_distribution_kl(arrow_probs_concat, arrow_true_concat)

    # Multi-tolerance F1 at the primary best threshold. 50 ms / 100 ms windows
    # tell us how much of the recall gap is jitter (recoverable with a softer
    # match) vs genuine misses. 3/5/10 frames ≈ 30/50/100 ms at 100 fps.
    extra_tols: Dict[int, Tuple[float, float, float]] = {}
    for extra_tol in (5, 10):
        if extra_tol == tol_frames:
            extra_tols[extra_tol] = (best_p, best_r, best_f1)
            continue
        # Wider tolerance => recompute peaks at that window (NMS uses tol).
        peak_idx_e, peak_probs_e = _all_local_max_peaks(
            pred_1d, extra_tol, min_threshold=float(best_thr),
        )
        p_e, r_e, f_e = _f1_at_threshold(
            peak_idx_e, peak_probs_e, true_peaks, extra_tol, float(best_thr),
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
    print("  [validate] threshold sweep done; running calibration sweep...", flush=True)
    if all_type_logits:
        logits_cat = np.concatenate(all_type_logits, axis=0)   # [N, 3]
        targets_cat = np.concatenate(all_type_targets, axis=0)  # [N]
        print(f"  [validate] calibration: {logits_cat.shape[0]} supervised frames, 17x17 grid", flush=True)
        # Widened from ±2.0 → ±3.0: at the previous range, the optimal
        # (jump_bias, hold_bias) saturated at the lower edge, meaning the
        # sweep wanted to push further. ±3.0 gives more headroom while
        # still being a reasonable shift in logit space.
        bias_grid = np.arange(-3.0, 3.001, 0.25)
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

    # Per-arrow hold-head raw F1: sanity-check that the new dense
    # supervision is being absorbed. Sweep a single threshold per arrow on
    # plain per-frame BCE probs (no NMS — in-hold is a span, not a peak).
    hold_arrow_f1 = np.zeros(4, dtype=np.float32)
    hold_arrow_thr = np.full(4, 0.5, dtype=np.float32)
    if all_hold_probs:
        hp = np.concatenate(all_hold_probs, axis=0).reshape(-1, 4)
        ht = (np.concatenate(all_hold_targets, axis=0).reshape(-1, 4) > 0.5)
        for a in range(4):
            best_af1, best_athr = 0.0, 0.5
            for thr in np.arange(0.1, 0.91, 0.05):
                pred = hp[:, a] >= thr
                tp_h = int(np.sum(pred & ht[:, a]))
                fp_h = int(np.sum(pred & ~ht[:, a]))
                fn_h = int(np.sum(~pred & ht[:, a]))
                if tp_h + fp_h == 0 or tp_h + fn_h == 0:
                    continue
                prec_h = tp_h / max(tp_h + fp_h, 1)
                rec_h = tp_h / max(tp_h + fn_h, 1)
                f1_h = 2 * prec_h * rec_h / max(prec_h + rec_h, 1e-8)
                if f1_h > best_af1:
                    best_af1, best_athr = float(f1_h), float(thr)
            hold_arrow_f1[a] = best_af1
            hold_arrow_thr[a] = best_athr
    print("  [validate] done", flush=True)

    return {
        'loss': total_loss / max(total_samples, 1),
        'arrow_loss': total_arrow / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'duration_loss': total_dur / max(total_samples, 1),
        'beat_loss': total_beat / max(total_samples, 1),
        'diversity_loss': total_div / max(total_samples, 1),
        'hold_loss': total_hold / max(total_samples, 1),
        # Per-arrow raw hold-head F1 (diagnostic on the new dense head).
        'hold_head_f1_L': float(hold_arrow_f1[0]),
        'hold_head_f1_D': float(hold_arrow_f1[1]),
        'hold_head_f1_U': float(hold_arrow_f1[2]),
        'hold_head_f1_R': float(hold_arrow_f1[3]),
        'hold_head_f1_macro': float(hold_arrow_f1.mean()),
        'hold_head_thr_L': float(hold_arrow_thr[0]),
        'hold_head_thr_D': float(hold_arrow_thr[1]),
        'hold_head_thr_U': float(hold_arrow_thr[2]),
        'hold_head_thr_R': float(hold_arrow_thr[3]),
        # "Any-onset" F1 (row-max over per-arrow probs); kept for continuity
        # with the legacy onset-head metric and for checkpoint selection.
        'tol_precision': prec,
        'tol_recall': rec,
        'tol_f1': f1,
        'tol_f1_50ms': float(extra_tols[5][2]),
        'tol_f1_100ms': float(extra_tols[10][2]),
        'best_thr': float(best_thr),
        # Per-arrow F1 + per-arrow calibrated thresholds.
        'arrow_f1_L': float(per_arrow_f1[0]),
        'arrow_f1_D': float(per_arrow_f1[1]),
        'arrow_f1_U': float(per_arrow_f1[2]),
        'arrow_f1_R': float(per_arrow_f1[3]),
        'arrow_f1_macro': float(per_arrow_f1.mean()),
        'arrow_thr_L': float(per_arrow_best_thr[0]),
        'arrow_thr_D': float(per_arrow_best_thr[1]),
        'arrow_thr_U': float(per_arrow_best_thr[2]),
        'arrow_thr_R': float(per_arrow_best_thr[3]),
        # Diagnostic: KL(true||pred) on per-arrow marginal in note frames.
        # Lower is better; > ~0.1 usually means the model is collapsing to
        # one or two arrows.
        'arrow_marginal_kl': float(arrow_kl),
        # Stream coherence in dense sections; ground truth sits ~0.7-0.85.
        'stream_coherence': float(stream_coh),
        'type_acc': type_correct / max(type_total, 1),
        'tap_f1': tap_f1, 'tap_p': tap_p, 'tap_r': tap_r,
        'jump_f1': jump_f1, 'jump_p': jump_p, 'jump_r': jump_r,
        'hold_f1': hold_f1, 'hold_p': hold_p, 'hold_r': hold_r,
        'type_macro_f1': type_macro_f1,
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


def _stream_coherence(
    arrow_probs: np.ndarray,
    threshold: float,
    min_dense_count: int = 4,
    frames_per_second: int = 100,
) -> float:
    """Fraction of consecutive non-equal predicted arrows in dense windows.

    A "dense" window is a 1-second slice containing ≥ min_dense_count frames
    where any arrow exceeds `threshold`. Inside each dense window, take the
    sequence of argmax-arrows at those onset frames and compute the fraction
    of adjacent pairs that differ. Real DDR charts in stream sections sit
    around 0.7-0.85 — pure L-R-L-R is 1.0 (suspiciously uniform), pure jacks
    is 0.0. Returns 0.0 if no dense windows are found.
    """
    N = arrow_probs.shape[0]
    if N == 0:
        return 0.0
    onset_mask = arrow_probs.max(axis=-1) >= threshold      # [N]
    if onset_mask.sum() < min_dense_count:
        return 0.0
    onset_idx = np.flatnonzero(onset_mask)
    onset_arrows = arrow_probs[onset_idx].argmax(axis=-1)   # [n_onsets]
    win = frames_per_second
    diffs, totals = 0, 0
    i = 0
    while i < onset_idx.size:
        j = i
        end_frame = onset_idx[i] + win
        while j < onset_idx.size and onset_idx[j] < end_frame:
            j += 1
        seg = onset_arrows[i:j]
        if seg.size >= min_dense_count:
            diffs += int((seg[1:] != seg[:-1]).sum())
            totals += int(seg.size - 1)
        i = j
    if totals == 0:
        return 0.0
    return float(diffs) / float(totals)


def _arrow_distribution_kl(
    arrow_probs: np.ndarray, arrow_true: np.ndarray, eps: float = 1e-6,
) -> float:
    """KL(true_marginal || pred_marginal) over note-bearing frames.

    Marginal is per-arrow fraction of fires across positive frames. A pure
    L-R model on a balanced ground truth gets KL ≈ ln(2) ≈ 0.69. Returns 0
    when no frames have any onset.
    """
    onset_frames = arrow_true.max(axis=-1) > 0.5
    if not onset_frames.any():
        return 0.0
    true_dist = arrow_true[onset_frames].sum(axis=0)
    true_dist = true_dist / max(float(true_dist.sum()), eps)
    pred_dist = arrow_probs[onset_frames].sum(axis=0)
    pred_dist = pred_dist / max(float(pred_dist.sum()), eps)
    pred_dist = np.clip(pred_dist, eps, 1.0)
    true_dist = np.clip(true_dist, eps, 1.0)
    return float(np.sum(true_dist * (np.log(true_dist) - np.log(pred_dist))))

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
    arrow_priors: np.ndarray,
    args,
) -> None:
    ckpt = {
        'epoch': epoch,
        # arch_version history:
        #   1 = legacy single-onset head (missing key)
        #   2 = per-arrow head with prev_arrow conditioning
        #   3 = +(start_seconds, remaining_seconds) FiLM conditioning
        #   4 = type/duration heads conditioned on zero-context arrow + beat logits
        #   5 = +hold_head (per-arrow in-hold), dense duration loss,
        #       jump co-fire, multi-stride jack penalty
        #   6 = hold_head conditioned on zero-context arrow logits (input
        #       dim hidden+4 instead of hidden); separate optimizer bucket
        'arch_version': 6,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': (
            ema_model.module.state_dict() if ema_model is not None else None
        ),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'duration_unit': 'beats',
        'best_threshold': float(metrics.get('best_thr', 0.5)),
        # Per-arrow calibrated thresholds saved alongside the legacy
        # `best_threshold` so inference can use them per channel.
        'best_arrow_thresholds': [
            float(metrics.get('arrow_thr_L', 0.5)),
            float(metrics.get('arrow_thr_D', 0.5)),
            float(metrics.get('arrow_thr_U', 0.5)),
            float(metrics.get('arrow_thr_R', 0.5)),
        ],
        'best_jump_bias': float(metrics.get('cal_jump_bias', 0.0)),
        'best_hold_bias': float(metrics.get('cal_hold_bias', 0.0)),
        'default_density_by_id': default_density_by_id.tolist(),
        'mel_mean': float(mel_mean),
        'mel_std': float(mel_std),
        'type_prior': [float(x) for x in type_prior],
        'arrow_priors': [float(x) for x in arrow_priors],
        'args': vars(args) if not isinstance(args, dict) else args,
        'git_sha': _git_sha(),
    }
    torch.save(ckpt, path)

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train step chart model')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='ml/checkpoints')
    parser.add_argument('--difficulty', type=int, default=None, choices=[0, 1, 2, 3, 4],
                        help='Train on a single difficulty only (0=beginner, '
                             '1=easy, 2=medium, 3=hard, 4=challenge). When set, '
                             'train+val indices are filtered to that difficulty '
                             'and checkpoints land in <checkpoint-dir>/diff_<N>/ '
                             'so multiple per-difficulty runs do not collide. '
                             'Trade-off: each model fully specializes but sees '
                             'only ~1/5 the data, so the audio backbone learns '
                             'less generalizable rhythmic features. Run 5x to '
                             'get a per-difficulty model bank.')
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
    parser.add_argument('--type-weight-smoothing', type=float, default=0.65,
                        help='Exponent for inverse-frequency type-class weights '
                             '(0=uniform, 1=raw inverse-frequency). 0.65 gives '
                             'jumps/holds noticeably more weight than 0.5 — the '
                             'last run finished with cal_F1[j/h]=0.24/0.28, '
                             'and the calibration sweep landed on hb=-0.5 '
                             '(suppress holds further), indicating the head '
                             'needs more class-imbalance correction.')
    parser.add_argument('--type-weight-cap', type=float, default=30.0,
                        help='Max per-class CE weight after smoothing')
    parser.add_argument('--type-focal-gamma', type=float, default=1.0,
                        help='Focal-loss gamma on the type CE. 0 disables; '
                             '1.0 lightly down-weights confidently-correct '
                             '(easy tap) frames so rare jump/hold frames '
                             'contribute more gradient. Lowered from 2.0 in '
                             'v5: with the new dedicated hold_head carrying '
                             'the bulk of hold supervision, gamma=2.0 was '
                             'over-aggressive on top of type_weight=4.0 + '
                             'class weights + 8×/16× sampler boosts and '
                             'pinned the calibration sweep at hb=-0.25 for '
                             '27 consecutive epochs.')
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
    parser.add_argument('--intro-outro-oversample-prob', type=float, default=0.15,
                        help='Per-sample probability of forcing a training '
                             'chunk to start at song frame 0 (intro), with the '
                             'same probability of anchoring to song end. '
                             'Without this, edge chunks are far too rare for '
                             'the model to learn intro-silence patterns.')
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
    parser.add_argument('--diversity-weight', type=float, default=0.10,
                        help='Loss weight on the per-chunk arrow-distribution '
                             'KL regularizer. Rewards predicted marginals '
                             'matching the target marginals — useful against '
                             'mode collapse, but in tension with the per-frame '
                             'commit loss since both want to constrain the '
                             'arrow head. Modest bump from 0.05 paired with '
                             'the new jack penalty to push stream_coh up '
                             'from the 0.31 floor seen last run. 0.0 disables.')
    parser.add_argument('--arrow-commit-weight', type=float, default=0.5,
                        help='Loss weight on the per-frame softmax-CE applied '
                             'to single-arrow frames. Forces the arrow head '
                             'to commit to a specific arrow per onset rather '
                             'than predicting the per-arrow marginal — fixes '
                             'low stream coherence. 0.0 disables.')
    parser.add_argument('--jack-weight', type=float, default=0.3,
                        help='Loss weight on the anti-jack penalty: '
                             'discourages same-arrow argmax on co-active '
                             'onset pairs separated by any of --jack-strides '
                             'frames. Targets stream_coherence directly '
                             '(real charts sit ~0.7-0.85). 0.0 disables.')
    parser.add_argument('--jack-strides', type=str, default='3,6,9',
                        help='Comma-separated frame offsets for the anti-jack '
                             'penalty pair. Stride 3 sits just outside the '
                             'onset-smoothing kernel (sigma=1.5 → ~2-frame '
                             'radius); strides 6/9 catch 8th/16th-note jacks '
                             'at 120-180 BPM that the kernel masks at stride '
                             '3 alone. Per-stride contributions weighted by '
                             '--jack-stride-decay^k and averaged.')
    parser.add_argument('--jack-stride-decay', type=float, default=0.5,
                        help='Geometric decay applied to per-stride jack '
                             'penalty contributions (later strides get '
                             'decay^k weight). 1.0 = uniform, 0.0 = stride-0 '
                             'only.')
    parser.add_argument('--jump-cofire-weight', type=float, default=0.5,
                        help='Loss weight on the jump co-fire term: extra '
                             'BCE on positive arrows of jump frames, '
                             'rewarding the joint event the per-arrow BCE '
                             'marginal cannot see (P(L)·P(R) caps at the '
                             'product of marginals, so independent BCE '
                             'never directly pulls the model toward firing '
                             'both arrows together). 0.0 disables.')
    parser.add_argument('--hold-weight', type=float, default=0.3,
                        help='Loss weight on the per-arrow hold-head BCE. '
                             'The hold head is supervised on dense in-hold '
                             'spans synthesized by the dataloader, giving '
                             '~10× more gradient than the single-frame '
                             'hold_start signal the type head sees. 0.0 '
                             'disables — head still emits logits but gets '
                             'no gradient.')
    parser.add_argument('--hold-pos-weight', type=float, default=0.0,
                        help='Override per-arrow BCE pos_weight on the '
                             'hold head (broadcast across arrows). 0 = '
                             'derive from the in-hold prior with the same '
                             'sqrt((1-p)/p) recipe as the arrow head.')
    parser.add_argument('--prev-arrow-dropout', type=float, default=0.1,
                        help='Per-frame probability of zeroing the prev_arrow '
                             'conditioning vector during training. Forces the '
                             'arrow head to remain useful without context, '
                             'matching inference at song start / after gaps.')
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
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging. Requires WANDB_API_KEY '
                             'in the environment (or `wandb login` already run).')
    parser.add_argument('--wandb-project', type=str, default='stepageddon',
                        help='W&B project name (used only if --wandb)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Optional W&B run name (defaults to W&B auto-name)')
    return parser


def main():
    print("[main] parsing args...", flush=True)
    args = build_argparser().parse_args()
    print("[main] args parsed; seeding...", flush=True)
    seed_everything(args.seed)

    print("[main] selecting device...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} | git_sha={_git_sha()} | seed={args.seed}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if getattr(args, 'difficulty', None) is not None:
        # Sub-folder per single-difficulty run so 5 parallel runs don't
        # overwrite each other's `best_model.pt`.
        checkpoint_dir = checkpoint_dir / f'diff_{int(args.difficulty)}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            wandb_run.config.update({'git_sha': _git_sha()}, allow_val_change=True)
            logger.info(f"W&B logging enabled: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed ({e}); continuing without it")
            wandb_run = None

    print("Building dataloaders...", flush=True)
    built = build_dataloaders(args)
    print("Dataloaders built. Building model...", flush=True)
    model = build_model(
        args, built.arrow_priors, built.type_prior, device,
        hold_duration_median=built.hold_duration_median,
        beat_prior=built.beat_prior,
        hold_priors=built.hold_priors,
    )
    print("[main] model built; building loss...", flush=True)
    criterion = build_loss(
        built.arrow_priors, built.type_prior, args,
        hold_priors=built.hold_priors,
    ).to(device)
    print("[main] loss built; computing steps_per_epoch (calls len(train_loader))...", flush=True)
    steps_per_epoch = max(1, len(built.train_loader))
    print(f"[main] steps_per_epoch={steps_per_epoch}; building optimizer...", flush=True)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
    print("[main] optimizer built", flush=True)
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

    print("Starting training...", flush=True)
    print(
        f"  steps_per_epoch={steps_per_epoch} "
        f"batch_size={args.batch_size} "
        f"num_workers={args.num_workers} "
        f"chunk_frames={args.chunk_frames}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        print(f"[epoch {epoch+1}] entering train_one_epoch", flush=True)
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, built.train_loader, criterion, optimizer, scheduler,
            scaler, device, ema_model=ema_model,
            log_interval=args.log_interval,
            prev_arrow_dropout=args.prev_arrow_dropout,
        )
        val_metrics = validate(
            ema_model.module, built.val_loader, criterion, device,
            tol_frames=args.tol_frames,
        )
        plateau_scheduler.step(val_metrics['tol_f1'])

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        if wandb_run is not None:
            wandb_run.log(
                {
                    'epoch': epoch + 1,
                    'lr': lr,
                    'epoch_time_s': elapsed,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                },
                step=epoch + 1,
            )
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"(arrow={train_metrics['arrow_loss']:.4f} "
            f"type={train_metrics['type_loss']:.4f} "
            f"dur={train_metrics['duration_loss']:.4f} "
            f"div={train_metrics['diversity_loss']:.4f} "
            f"hold={train_metrics['hold_loss']:.4f}) "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"tol_f1={val_metrics['tol_f1']:.3f} "
            f"(P={val_metrics['tol_precision']:.3f} "
            f"R={val_metrics['tol_recall']:.3f} "
            f"thr={val_metrics['best_thr']:.2f}) "
            f"arrow_F1[L/D/U/R]="
            f"{val_metrics['arrow_f1_L']:.2f}/"
            f"{val_metrics['arrow_f1_D']:.2f}/"
            f"{val_metrics['arrow_f1_U']:.2f}/"
            f"{val_metrics['arrow_f1_R']:.2f} "
            f"(macro={val_metrics['arrow_f1_macro']:.3f}) "
            f"arrow_kl={val_metrics['arrow_marginal_kl']:.4f} "
            f"stream_coh={val_metrics['stream_coherence']:.3f} "
            f"| type_F1[tap/jump/hold]="
            f"{val_metrics['tap_f1']:.3f}/"
            f"{val_metrics['jump_f1']:.3f}/"
            f"{val_metrics['hold_f1']:.3f} "
            f"(macro={val_metrics['type_macro_f1']:.3f}) "
            f"hold_head_F1={val_metrics['hold_head_f1_macro']:.3f} "
            f"dur_mae={val_metrics['dur_mae_beats']:.3f}b "
            f"| lr={lr:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        # Always overwrite 'last'
        save_checkpoint(
            checkpoint_dir / 'last_model.pt', epoch, model, ema_model,
            optimizer, scheduler, scaler, val_metrics,
            built.default_density_by_id,
            built.full_dataset.mel_mean, built.full_dataset.mel_std,
            built.type_prior, built.arrow_priors,
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
                built.type_prior, built.arrow_priors,
                args,
            )
            print(
                f"  -> New best composite={best_composite:.3f} "
                f"(tol_f1={val_metrics['tol_f1']:.3f} "
                f"cal_macro_f1={val_metrics.get('cal_macro_f1', val_metrics['type_macro_f1']):.3f} "
                f"thr={val_metrics['best_thr']:.2f})",
                flush=True,
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
                built.type_prior, built.arrow_priors,
                args,
            )
            print(f"  -> New best macro F1 = {best_macro_f1:.3f}", flush=True)
        if val_metrics['dur_mae_beats'] > 0 and val_metrics['dur_mae_beats'] < best_dur_mae:
            best_dur_mae = val_metrics['dur_mae_beats']
            save_checkpoint(
                checkpoint_dir / 'best_dur_mae.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id,
                built.full_dataset.mel_mean, built.full_dataset.mel_std,
                built.type_prior, built.arrow_priors,
                args,
            )
            print(f"  -> New best dur MAE = {best_dur_mae:.3f}b", flush=True)

    print(
        f"Training complete. Best composite: {best_composite:.3f} "
        f"| macro_f1: {best_macro_f1:.3f} | dur_mae: {best_dur_mae:.3f}b",
        flush=True,
    )

    if wandb_run is not None:
        wandb_run.summary['best_composite'] = best_composite
        wandb_run.summary['best_macro_f1'] = best_macro_f1
        wandb_run.summary['best_dur_mae_beats'] = best_dur_mae
        best_path = checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            try:
                artifact = wandb.Artifact('best_model', type='model')
                artifact.add_file(str(best_path))
                wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"W&B artifact upload failed: {e}")
        wandb_run.finish()


if __name__ == '__main__':
    main()
