"""
Training script for the v9 hybrid step chart model.

Three dense heads (onset / intensity / direction), no per-arrow / type /
duration / beat heads. The sustain head was dropped in v9 — holds are derived
algorithmically at inference (see `inference._derive_holds`). Composite val
metric:
    0.75 * onset_tol_f1 + 0.10 * intensity_jump_auc + 0.15 * arrow_col_f1

Usage:
    python -m ml.train \
        --data-dir <path>/training_data \
        --checkpoint-dir <path>/checkpoints \
        --epochs 50
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
from torch.utils.data import DataLoader, RandomSampler

from ml.model import StepChartLoss, StepChartModel, ARCH_VERSION
from ml.dataset import (
    DENSITY_MEAN,
    DENSITY_STD,
    FRAMES_PER_SECOND,
    StepChartDataset,
    compute_default_density_per_difficulty,
    compute_onset_prior,
    split_entries_by_song,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _amp_ctx(device: torch.device):
    """Mixed precision only on CUDA; a no-op on CPU/MPS so smoke runs work anywhere."""
    from contextlib import nullcontext
    return autocast('cuda') if device.type == 'cuda' else nullcontext()


def _select_device(name: str = 'auto') -> torch.device:
    if name and name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


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
    onset_prior: float
    default_density_by_id: torch.Tensor


def build_dataloaders(args) -> BuildOutputs:
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / 'manifest.json'
    print(f"[build_dataloaders] data_dir={data_dir} manifest={manifest_path}", flush=True)

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
        intro_outro_oversample_prob=float(
            getattr(args, 'intro_outro_oversample_prob', 0.15)
        ),
    )

    train_idx, val_idx = split_entries_by_song(
        full_dataset.entries, val_fraction=args.val_split, seed=args.seed,
    )

    max_songs = getattr(args, 'max_songs', None)
    if max_songs is not None:
        # Smoke-test cap: keep only the first N entries per split (deterministic).
        train_idx = train_idx[:max_songs]
        val_idx = val_idx[:max(1, max_songs // 5)]
        logger.info(f"--max-songs={max_songs}: train={len(train_idx)} val={len(val_idx)}")

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
                f"--difficulty={target} filter produced 0 train entries"
            )
        logger.info(
            f"--difficulty={target} filter: train={len(train_idx)} val={len(val_idx)}"
        )

    onset_prior = compute_onset_prior(full_dataset.entries, str(data_dir), train_idx)

    # Plain RandomSampler over train indices (no rare-class boosts).
    # Train DataLoader iterates over the *subset* via Subset for cleanliness.
    from torch.utils.data import Subset
    train_subset = Subset(full_dataset, train_idx)
    train_sampler = RandomSampler(train_subset)

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

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
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
        onset_prior=onset_prior,
        default_density_by_id=default_density_by_id,
    )


def build_model(
    args,
    onset_prior: float,
    n_in_channels: int,
    device: torch.device,
) -> StepChartModel:
    tcn_dilations = tuple(int(x) for x in args.tcn_dilations.split(','))
    model = StepChartModel(
        n_in_channels=n_in_channels,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        n_difficulties=5,
        dropout=args.dropout,
        tcn_dilations=tcn_dilations,
    ).to(device)
    model.set_onset_prior(onset_prior)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    return model


def build_optimizer_and_scheduler(model: nn.Module, args, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=args.weight_decay,
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
    onset_prior: float,
    args,
) -> StepChartLoss:
    """Onset pos_weight derived from the empirical positive rate:
    sqrt((1-p)/p) capped at 50.
    """
    if args.onset_pos_weight is not None:
        opw = float(args.onset_pos_weight)
    else:
        p = max(min(onset_prior, 0.5), 1e-5)
        opw = float(np.clip(np.sqrt((1.0 - p) / p), 1.0, 50.0))
    logger.info(f"Onset pos_weight={opw:.2f}")
    return StepChartLoss(
        onset_pos_weight=opw,
        focal_gamma=float(args.focal_gamma),
        intensity_weight=float(args.intensity_weight),
        arrow_weight=float(getattr(args, 'arrow_weight', 1.0)),
    )


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device,
    ema_model: Optional[AveragedModel] = None,
    log_interval: int = 50,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_onset = 0.0
    total_intensity = 0.0
    total_arrow = 0.0
    total_samples = 0
    n_batches = len(loader)

    interval = {'loss': 0.0, 'onset': 0.0, 'intensity': 0.0, 'arrow': 0.0, 'n': 0}
    t_start = time.time()

    for step, batch in enumerate(loader):
        (
            feats, difficulty, density,
            onset_soft, intensity_target,
            start_seconds, remaining_seconds, arrow_target,
        ) = batch
        feats = feats.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        intensity_target = intensity_target.to(device, non_blocking=True)
        start_seconds = start_seconds.to(device, non_blocking=True)
        remaining_seconds = remaining_seconds.to(device, non_blocking=True)
        arrow_target = arrow_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(device):
            onset_logits, intensity_pred, arrow_logits = model(
                feats, difficulty, density, start_seconds, remaining_seconds,
            )
            loss, onset_l, intensity_l, arrow_l = criterion(
                onset_logits, intensity_pred,
                onset_soft, intensity_target,
                arrow_logits, arrow_target,
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

        bs = feats.size(0)
        total_loss += loss.item() * bs
        total_onset += onset_l.item() * bs
        total_intensity += intensity_l.item() * bs
        total_arrow += arrow_l.item() * bs
        total_samples += bs

        interval['loss'] += loss.item() * bs
        interval['onset'] += onset_l.item() * bs
        interval['intensity'] += intensity_l.item() * bs
        interval['arrow'] += arrow_l.item() * bs
        interval['n'] += bs

        if (step + 1) % log_interval == 0 or (step + 1) == n_batches:
            n = max(interval['n'], 1)
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            print(
                f"  step {step+1:4d}/{n_batches} | "
                f"loss={interval['loss']/n:.4f} "
                f"(onset={interval['onset']/n:.4f} "
                f"intensity={interval['intensity']/n:.4f} "
                f"arrow={interval['arrow']/n:.4f}) "
                f"| lr={lr:.2e} | {elapsed:.1f}s",
                flush=True,
            )
            interval = {'loss': 0.0, 'onset': 0.0, 'intensity': 0.0, 'arrow': 0.0, 'n': 0}
            t_start = time.time()

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'intensity_loss': total_intensity / max(total_samples, 1),
        'arrow_loss': total_arrow / max(total_samples, 1),
    }


def _all_local_max_peaks(
    probs: np.ndarray, window: int, min_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """1-D NMS via scipy max-filter."""
    T = probs.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=probs.dtype)
    from scipy.ndimage import maximum_filter1d
    pooled = maximum_filter1d(probs, size=2 * window + 1, mode='constant', cval=-np.inf)
    is_peak = (probs == pooled) & (probs >= min_threshold)
    idx = np.flatnonzero(is_peak)
    if idx.size <= 1:
        return idx.astype(np.int64), probs[idx]
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
    """Greedy ±tol_frames matching."""
    sel = peak_probs >= threshold
    pred_peaks = peak_idx[sel]
    if pred_peaks.size == 0 and true_peaks.size == 0:
        return 0.0, 0.0, 0.0
    matched_true = np.zeros(true_peaks.size, dtype=bool)
    tp = 0
    j_start = 0
    for pf in pred_peaks:
        while j_start < true_peaks.size and true_peaks[j_start] < pf - tol_frames:
            j_start += 1
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


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank-sum identity. Returns 0.5 if class is degenerate."""
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    rank_sum = ranks[pos].sum()
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def validate(
    model, loader, criterion, device, tol_frames: int = 3,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_onset = 0.0
    total_intensity = 0.0
    total_samples = 0

    all_onset_pred: List[np.ndarray] = []
    all_onset_true: List[np.ndarray] = []
    all_intensity_pred: List[np.ndarray] = []
    all_intensity_true: List[np.ndarray] = []
    total_arrow = 0.0
    arrow_tp = arrow_fp = arrow_fn = 0  # per-column onset agreement @0.5, onset frames only

    for batch in loader:
        (
            feats, difficulty, density,
            onset_soft, intensity_target,
            start_seconds, remaining_seconds, arrow_target,
        ) = batch
        feats = feats.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        intensity_target = intensity_target.to(device, non_blocking=True)
        start_seconds = start_seconds.to(device, non_blocking=True)
        remaining_seconds = remaining_seconds.to(device, non_blocking=True)
        arrow_target = arrow_target.to(device, non_blocking=True)

        with _amp_ctx(device):
            onset_logits, intensity_pred, arrow_logits = model(
                feats, difficulty, density, start_seconds, remaining_seconds,
            )
            loss, onset_l, intensity_l, arrow_l = criterion(
                onset_logits, intensity_pred,
                onset_soft, intensity_target,
                arrow_logits, arrow_target,
            )

        bs = feats.size(0)
        total_loss += loss.item() * bs
        total_onset += onset_l.item() * bs
        total_intensity += intensity_l.item() * bs
        total_arrow += arrow_l.item() * bs
        total_samples += bs

        if arrow_logits is not None:
            ap = (torch.sigmoid(arrow_logits.float()) >= 0.5)
            at = (arrow_target >= 0.5)
            onset_frames = at.any(dim=-1, keepdim=True)          # [B,T,1]
            ap = ap & onset_frames                                # only judge onset frames
            arrow_tp += int((ap & at).sum().item())
            arrow_fp += int((ap & ~at).sum().item())
            arrow_fn += int((~ap & at).sum().item())

        all_onset_pred.append(
            torch.sigmoid(onset_logits.float()).cpu().numpy().reshape(-1)
        )
        all_onset_true.append(onset_soft.cpu().numpy().reshape(-1))
        all_intensity_pred.append(intensity_pred.float().cpu().numpy().reshape(-1))
        all_intensity_true.append(intensity_target.cpu().numpy().reshape(-1))

    onset_pred = np.concatenate(all_onset_pred)
    onset_true_soft = np.concatenate(all_onset_true)
    onset_true_hard = (onset_true_soft >= 0.5).astype(np.float32)
    intensity_pred_all = np.concatenate(all_intensity_pred)
    intensity_true_all = np.concatenate(all_intensity_true)

    # Onset tolerance F1 sweep at tol_frames (~30 ms) and 5 frames (~50 ms).
    true_peaks = np.flatnonzero(onset_true_hard > 0.5)
    thresholds = np.arange(0.05, 0.91, 0.05)

    def _sweep(tol):
        peak_idx, peak_probs = _all_local_max_peaks(
            onset_pred, tol, min_threshold=float(thresholds.min()),
        )
        best = (0.0, 0.5, 0.0, 0.0)  # f1, thr, p, r
        for thr in thresholds:
            p_, r_, f_ = _f1_at_threshold(
                peak_idx, peak_probs, true_peaks, tol, float(thr),
            )
            if f_ > best[0]:
                best = (f_, float(thr), p_, r_)
        return best

    onset_f1, onset_thr, onset_p, onset_r = _sweep(tol_frames)
    onset_f1_50ms, _, _, _ = _sweep(5)

    # Intensity AUC: discriminate "jump-ish frame" (target > 0.5) vs.
    # everything else, using the regression head's raw output as score.
    jump_label = (intensity_true_all > 0.5).astype(np.float32)
    intensity_auc = _binary_auc(intensity_pred_all, jump_label)
    intensity_mse = float(np.mean((intensity_pred_all - intensity_true_all) ** 2))

    # Per-column direction F1 at 0.5, over onset frames only.
    a_prec = arrow_tp / max(arrow_tp + arrow_fp, 1)
    a_rec = arrow_tp / max(arrow_tp + arrow_fn, 1)
    arrow_col_f1 = 2 * a_prec * a_rec / max(a_prec + a_rec, 1e-8)

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'intensity_loss': total_intensity / max(total_samples, 1),
        'arrow_loss': total_arrow / max(total_samples, 1),
        'arrow_col_f1': float(arrow_col_f1),
        'onset_tol_f1': float(onset_f1),
        'onset_tol_p': float(onset_p),
        'onset_tol_r': float(onset_r),
        'onset_tol_f1_50ms': float(onset_f1_50ms),
        'best_onset_threshold': float(onset_thr),
        'intensity_jump_auc': float(intensity_auc),
        'intensity_mse': intensity_mse,
    }


def composite_score(metrics: Dict[str, float]) -> float:
    """Headline checkpoint score. v8 folds in the direction head at a small weight
    (the arrow head does not gate onset/intensity quality). The 0.15 the
    sustain head used to carry is redistributed to onset and direction, the two
    signals the algorithmic hold pass now depends on:
        0.75*onset_tol_f1 + 0.10*intensity_jump_auc + 0.15*arrow_col_f1
    """
    onset = float(metrics.get('onset_tol_f1', 0.0))
    auc = float(metrics.get('intensity_jump_auc', 0.5))
    arrow = float(metrics.get('arrow_col_f1', 0.0))
    return 0.75 * onset + 0.10 * auc + 0.15 * arrow


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
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    n_in_channels: int,
    onset_prior: float,
    args,
) -> None:
    ckpt = {
        'epoch': epoch,
        'arch_version': ARCH_VERSION,
        'n_in_channels': int(n_in_channels),
        'model_state_dict': model.state_dict(),
        'ema_state_dict': (
            ema_model.module.state_dict() if ema_model is not None else None
        ),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
        'best_onset_threshold': float(metrics.get('best_onset_threshold', 0.5)),
        'onset_prior': float(onset_prior),
        'default_density_by_id': default_density_by_id.tolist(),
        'feat_mean': [float(x) for x in feat_mean],
        'feat_std': [float(x) for x in feat_std],
        'args': vars(args) if not isinstance(args, dict) else args,
        'git_sha': _git_sha(),
    }
    torch.save(ckpt, path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train v7 step chart model')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='ml/checkpoints')
    parser.add_argument('--difficulty', type=int, default=None, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=160)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--chunk-frames', type=int, default=500)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=float, default=2.0)
    parser.add_argument('--tcn-dilations', type=str, default='1,2,4,8,16')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tol-frames', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--intensity-weight', type=float, default=0.5)
    parser.add_argument('--arrow-weight', type=float, default=1.0,
                        help='Weight of the v8 learned direction (per-column) BCE term')
    parser.add_argument('--device', type=str, default='auto',
                        help="auto | cuda | mps | cpu (auto prefers cuda, then mps, then cpu)")
    parser.add_argument('--max-songs', type=int, default=None,
                        help='Cap number of songs (for smoke tests)')
    parser.add_argument('--onset-pos-weight', type=float, default=None,
                        help='Override onset BCE pos_weight; default sqrt((1-p)/p) capped at 50')
    parser.add_argument('--p-density-swap', type=float, default=0.5)
    parser.add_argument('--intro-outro-oversample-prob', type=float, default=0.15)
    parser.add_argument('--plateau-patience', type=int, default=4)
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='stepageddon')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    return parser


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    device = _select_device(getattr(args, 'device', 'auto'))
    logger.info(f"Using device: {device} | git_sha={_git_sha()} | seed={args.seed}")

    checkpoint_dir = Path(args.checkpoint_dir)
    if getattr(args, 'difficulty', None) is not None:
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
    n_in_channels = built.full_dataset.n_in_channels
    print(f"Dataloaders built. Building model (n_in_channels={n_in_channels})...", flush=True)
    model = build_model(
        args, built.onset_prior,
        n_in_channels=n_in_channels, device=device,
    )
    criterion = build_loss(built.onset_prior, args).to(device)
    steps_per_epoch = max(1, len(built.train_loader))
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=1e-6,
    )
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)
    )

    start_epoch = 0
    best_composite = -1.0

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
        best_composite = composite_score(ckpt.get('metrics', {}))

    print("Starting training...", flush=True)
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
        plateau_scheduler.step(val_metrics['onset_tol_f1'])

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        composite = composite_score(val_metrics)

        if wandb_run is not None:
            wandb_run.log(
                {
                    'epoch': epoch + 1,
                    'lr': lr,
                    'epoch_time_s': elapsed,
                    'composite': composite,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                },
                step=epoch + 1,
            )

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"(onset={train_metrics['onset_loss']:.4f} "
            f"intensity={train_metrics['intensity_loss']:.4f}) "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"onset_tol_f1={val_metrics['onset_tol_f1']:.3f} "
            f"(P={val_metrics['onset_tol_p']:.3f} "
            f"R={val_metrics['onset_tol_r']:.3f} "
            f"thr={val_metrics['best_onset_threshold']:.2f}) "
            f"f1_50ms={val_metrics['onset_tol_f1_50ms']:.3f} "
            f"intensity_auc={val_metrics['intensity_jump_auc']:.3f} "
            f"intensity_mse={val_metrics['intensity_mse']:.4f} "
            f"| composite={composite:.3f} "
            f"| lr={lr:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        save_checkpoint(
            checkpoint_dir / 'last_model.pt', epoch, model, ema_model,
            optimizer, scheduler, scaler, val_metrics,
            built.default_density_by_id,
            built.full_dataset.feat_mean, built.full_dataset.feat_std,
            n_in_channels,
            built.onset_prior,
            args,
        )

        if composite > best_composite:
            best_composite = composite
            save_checkpoint(
                checkpoint_dir / 'best_model.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id,
                built.full_dataset.feat_mean, built.full_dataset.feat_std,
                n_in_channels,
                built.onset_prior,
                args,
            )
            print(f"  -> New best composite={best_composite:.3f}", flush=True)

    print(f"Training complete. Best composite={best_composite:.3f}", flush=True)

    if wandb_run is not None:
        wandb_run.summary['best_composite'] = best_composite
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
