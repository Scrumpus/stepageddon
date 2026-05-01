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
from torch.utils.data import DataLoader, Subset

from ml.model import StepChartLoss, StepChartModel
from ml.dataset import (
    DENSITY_MEAN,
    DENSITY_STD,
    FRAMES_PER_SECOND,
    StepChartDataset,
    class_weights_from_prior,
    compute_default_density_per_difficulty,
    compute_note_counts,
    compute_onset_prior,
    compute_type_class_distribution,
    make_note_weighted_sampler,
    split_entries_by_song,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seeding / repro
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Builders (importable from notebook)
# ---------------------------------------------------------------------------

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


def build_dataloaders(args) -> BuildOutputs:
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / 'manifest.json'

    full_dataset = StepChartDataset(
        data_dir=str(data_dir),
        manifest_path=str(manifest_path),
        chunk_frames=args.chunk_frames,
        is_train=True,
        augment=True,
    )

    # By-song split (non-leaky)
    train_idx, val_idx = split_entries_by_song(
        full_dataset.entries, val_fraction=args.val_split, seed=args.seed,
    )

    # Per-entry note counts (for weighted sampler + onset prior)
    note_counts = compute_note_counts(full_dataset.entries, str(data_dir))
    onset_prior = compute_onset_prior(full_dataset.entries, str(data_dir), train_idx)
    type_prior = compute_type_class_distribution(
        full_dataset.entries, str(data_dir), train_idx,
    )

    # Training: random chunks w/ augmentation + note-weighted sampling
    train_subset = Subset(full_dataset, train_idx)
    train_sampler = make_note_weighted_sampler(train_idx, note_counts)

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

    default_density_by_id = compute_default_density_per_difficulty(
        str(manifest_path), str(data_dir)
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
    )


def build_model(
    args,
    onset_prior: float,
    type_prior: np.ndarray,
    device: torch.device,
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
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    return model


def build_optimizer_and_scheduler(
    model: nn.Module, args, steps_per_epoch: int
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
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
    # Default to sqrt((1-p)/p) instead of (1-p)/p. The unit-mean formula
    # crushes precision (BCE rewards "predict positive everywhere" given how
    # rare positives are at frame level); the sqrt variant balances P/R far
    # better in practice. CLI override available via --pos-weight.
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
    )


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

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
    total_samples = 0
    n_batches = len(loader)

    # Running averages for intra-epoch logging
    interval_loss = 0.0
    interval_onset = 0.0
    interval_type = 0.0
    interval_dur = 0.0
    interval_samples = 0
    t_start = time.time()

    for step, (mel, difficulty, density, onset_soft, type_target, duration_target) in enumerate(loader):
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        duration_target = duration_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            onset_logits, type_logits, duration_pred = model(mel, difficulty, density)
            loss, onset_l, type_l, dur_l = criterion(
                onset_logits, type_logits, duration_pred,
                onset_soft, type_target, duration_target,
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
        total_samples += bs

        interval_loss += loss.item() * bs
        interval_onset += onset_l.item() * bs
        interval_type += type_l.item() * bs
        interval_dur += dur_l.item() * bs
        interval_samples += bs

        if (step + 1) % log_interval == 0 or (step + 1) == n_batches:
            avg_l = interval_loss / max(interval_samples, 1)
            avg_o = interval_onset / max(interval_samples, 1)
            avg_t = interval_type / max(interval_samples, 1)
            avg_d = interval_dur / max(interval_samples, 1)
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            logger.info(
                f"  step {step+1:4d}/{n_batches} "
                f"| loss={avg_l:.4f} (onset={avg_o:.4f} type={avg_t:.4f} dur={avg_d:.4f}) "
                f"| lr={lr:.2e} | {elapsed:.1f}s"
            )
            interval_loss = 0.0
            interval_onset = 0.0
            interval_type = 0.0
            interval_dur = 0.0
            interval_samples = 0
            t_start = time.time()

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'duration_loss': total_dur / max(total_samples, 1),
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
    total_samples = 0

    all_pred = []
    all_true = []

    type_correct = 0
    type_total = 0

    # Per-class confusion for {tap=0, jump=1, hold_start=2}.
    # tp[c] = correctly predicted as c; fp[c] = predicted c but actually
    # other; fn[c] = actually c but predicted other. Computed only on
    # frames with a true onset (mask = type_target >= 0).
    type_tp = np.zeros(3, dtype=np.int64)
    type_fp = np.zeros(3, dtype=np.int64)
    type_fn = np.zeros(3, dtype=np.int64)

    dur_abs_errors = []

    for mel, difficulty, density, onset_soft, type_target, duration_target in loader:
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)
        duration_target = duration_target.to(device, non_blocking=True)

        with autocast('cuda'):
            onset_logits, type_logits, duration_pred = model(mel, difficulty, density)
            loss, onset_l, type_l, dur_l = criterion(
                onset_logits, type_logits, duration_pred,
                onset_soft, type_target, duration_target,
            )

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_onset += onset_l.item() * bs
        total_type += type_l.item() * bs
        total_dur += dur_l.item() * bs
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
            type_total += int(tt_flat.size)
            for c in range(3):
                pred_c = (tp_flat == c)
                true_c = (tt_flat == c)
                type_tp[c] += int((pred_c & true_c).sum())
                type_fp[c] += int((pred_c & ~true_c).sum())
                type_fn[c] += int((~pred_c & true_c).sum())

        # Duration MAE on hold_start frames
        hold_mask = (type_target == 2)  # hold_start class
        if hold_mask.any():
            pred_d = duration_pred[:, :, 0][hold_mask].float().cpu().numpy()
            true_d = duration_target[hold_mask].float().cpu().numpy()
            dur_abs_errors.append(np.abs(pred_d - true_d))

    pred_concat = np.concatenate(all_pred, axis=0).reshape(-1, 1)
    true_concat = np.concatenate(all_true, axis=0).reshape(-1, 1)
    # Sweep decision thresholds and report F1 at the best operating point.
    # A fixed 0.5 threshold is meaningless once the model's calibration
    # shifts; the headline metric should be peak F1.
    best_f1, best_thr, best_p, best_r = -1.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.30, 0.86, 0.05):
        p_, r_, f_ = tolerance_f1(
            pred_concat, true_concat,
            tol_frames=tol_frames, threshold=float(thr),
        )
        if f_ > best_f1:
            best_f1, best_thr, best_p, best_r = f_, float(thr), p_, r_
    prec, rec, f1 = best_p, best_r, best_f1

    dur_mae = float(np.concatenate(dur_abs_errors).mean()) if dur_abs_errors else 0.0

    def _f1(tp, fp, fn):
        prec_c = tp / max(tp + fp, 1)
        rec_c = tp / max(tp + fn, 1)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
        return float(prec_c), float(rec_c), float(f1_c)

    tap_p, tap_r, tap_f1 = _f1(type_tp[0], type_fp[0], type_fn[0])
    jump_p, jump_r, jump_f1 = _f1(type_tp[1], type_fp[1], type_fn[1])
    hold_p, hold_r, hold_f1 = _f1(type_tp[2], type_fp[2], type_fn[2])
    type_macro_f1 = (tap_f1 + jump_f1 + hold_f1) / 3.0

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'duration_loss': total_dur / max(total_samples, 1),
        'tol_precision': prec,
        'tol_recall': rec,
        'tol_f1': f1,
        'best_thr': float(best_thr),
        'type_acc': type_correct / max(type_total, 1),
        'tap_f1': tap_f1, 'tap_p': tap_p, 'tap_r': tap_r,
        'jump_f1': jump_f1, 'jump_p': jump_p, 'jump_r': jump_r,
        'hold_f1': hold_f1, 'hold_p': hold_p, 'hold_r': hold_r,
        'type_macro_f1': type_macro_f1,
        'dur_mae': dur_mae,
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

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
        # Persist the best onset decision threshold from the validation
        # sweep so inference can use the calibrated value instead of a
        # hardcoded 0.5/0.3. Falls back to 0.5 if the metric was missing.
        'best_threshold': float(metrics.get('best_thr', 0.5)),
        'default_density_by_id': default_density_by_id.tolist(),
        # Mel whitening stats live alongside the weights so inference can
        # apply the exact same input normalization the model was trained on.
        'mel_mean': float(mel_mean),
        'mel_std': float(mel_std),
        # Empirical P(class | onset) over {tap, jump, hold_start} on the
        # train split. Inference uses this for logit adjustment to undo
        # the prior bias learned by the type head.
        'type_prior': [float(x) for x in type_prior],
        'args': vars(args) if not isinstance(args, dict) else args,
        'git_sha': _git_sha(),
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument('--type-weight-smoothing', type=float, default=0.3,
                        help='Exponent for inverse-frequency type-class weights '
                             '(0=uniform, 1=raw inverse-frequency). Low values '
                             'are needed because tap dominates ~92%% of onsets.')
    parser.add_argument('--type-weight-cap', type=float, default=15.0,
                        help='Max per-class CE weight after smoothing')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='Override BCE pos_weight on the onset head. '
                             'Default = sqrt((1-p)/p) capped at 50.')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal-loss gamma on the onset head BCE. '
                             '0 disables focal weighting; 2.0 is standard.')
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
    model = build_model(args, built.onset_prior, built.type_prior, device)
    criterion = build_loss(built.onset_prior, built.type_prior, args).to(device)
    steps_per_epoch = max(1, len(built.train_loader))
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
    # Plateau scheduler runs in addition to the per-step cosine: if tol_f1
    # stalls for `plateau-patience` epochs, multiply LR by `plateau-factor`.
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        min_lr=1e-6,
    )
    scaler = GradScaler('cuda')

    # EMA over raw parameters (not buffers)
    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)
    )

    start_epoch = 0
    best_f1 = -1.0

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
        best_f1 = ckpt.get('metrics', {}).get('tol_f1', -1.0)

    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, built.train_loader, criterion, optimizer, scheduler,
            scaler, device, ema_model=ema_model,
            log_interval=args.log_interval,
        )
        # Validate with EMA weights
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
            f"type_acc={val_metrics['type_acc']:.3f} "
            f"type_F1[tap/jump/hold]="
            f"{val_metrics['tap_f1']:.3f}/"
            f"{val_metrics['jump_f1']:.3f}/"
            f"{val_metrics['hold_f1']:.3f} "
            f"(macro={val_metrics['type_macro_f1']:.3f}) "
            f"dur_mae={val_metrics['dur_mae']:.3f}s "
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

        if val_metrics['tol_f1'] > best_f1:
            best_f1 = val_metrics['tol_f1']
            save_checkpoint(
                checkpoint_dir / 'best_model.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id,
                built.full_dataset.mel_mean, built.full_dataset.mel_std,
                built.type_prior,
                args,
            )
            logger.info(
                f"  -> New best (tol_f1={best_f1:.3f} "
                f"thr={val_metrics['best_thr']:.2f})"
            )

    logger.info(f"Training complete. Best tol_f1: {best_f1:.3f}")


if __name__ == '__main__':
    main()
