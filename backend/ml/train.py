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
    compute_default_density_per_difficulty,
    compute_note_counts,
    compute_onset_prior,
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
    )


def build_model(args, onset_prior: float, device: torch.device) -> StepChartModel:
    model = StepChartModel(
        n_mels=80,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        n_difficulties=5,
        dropout=args.dropout,
    ).to(device)
    model.set_onset_prior(onset_prior)
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


def build_loss(onset_prior: float, args) -> StepChartLoss:
    # pos_weight = (1 - p) / p gives unit-mean BCE; clamp to avoid extremes
    p = max(onset_prior, 1e-5)
    pos_weight = min(max((1.0 - p) / p, 1.0), 200.0)
    logger.info(f"BCE pos_weight = {pos_weight:.2f} (from onset_prior={p:.4f})")
    return StepChartLoss(pos_weight=pos_weight, type_weight=args.type_weight)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device,
    ema_model: Optional[AveragedModel] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_onset = 0.0
    total_type = 0.0
    total_samples = 0

    for mel, difficulty, density, onset_soft, type_target in loader:
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            onset_logits, type_logits = model(mel, difficulty, density)
            loss, onset_l, type_l = criterion(
                onset_logits, type_logits, onset_soft, type_target
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
        total_samples += bs

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
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
    pred_probs: np.ndarray,     # [T, 4]
    true_hard: np.ndarray,      # [T, 4] in {0,1}
    tol_frames: int = 3,
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Greedy ±tol_frames matching per arrow. Returns (precision, recall, f1)."""
    tp = fp = fn = 0
    for a in range(pred_probs.shape[1]):
        pred_peaks = _local_max_peaks(pred_probs[:, a], threshold, tol_frames)
        true_peaks = np.where(true_hard[:, a] > 0.5)[0]
        matched = np.zeros(len(true_peaks), dtype=bool)
        a_tp = 0
        for pf in pred_peaks:
            if len(true_peaks) == 0:
                break
            dists = np.abs(true_peaks - pf)
            valid = (dists <= tol_frames) & (~matched)
            if not valid.any():
                continue
            j = int(np.argmin(np.where(valid, dists, 10**9)))
            matched[j] = True
            a_tp += 1
        tp += a_tp
        fp += len(pred_peaks) - a_tp
        fn += len(true_peaks) - a_tp
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
    total_samples = 0

    all_pred = []
    all_true = []

    type_correct = 0
    type_total = 0

    for mel, difficulty, density, onset_soft, type_target in loader:
        mel = mel.to(device, non_blocking=True)
        difficulty = difficulty.to(device, non_blocking=True)
        density = density.to(device, non_blocking=True)
        onset_soft = onset_soft.to(device, non_blocking=True)
        type_target = type_target.to(device, non_blocking=True)

        with autocast('cuda'):
            onset_logits, type_logits = model(mel, difficulty, density)
            loss, onset_l, type_l = criterion(
                onset_logits, type_logits, onset_soft, type_target
            )

        bs = mel.size(0)
        total_loss += loss.item() * bs
        total_onset += onset_l.item() * bs
        total_type += type_l.item() * bs
        total_samples += bs

        probs = torch.sigmoid(onset_logits.float()).cpu().numpy()
        # val uses hard targets (no smoothing) so onset_soft == onset_hard
        true_hard = (onset_soft.cpu().numpy() > 0.5).astype(np.float32)
        all_pred.append(probs)
        all_true.append(true_hard)

        # Type accuracy on frames with an onset
        type_pred = type_logits.argmax(dim=-1)
        mask = type_target >= 0
        if mask.any():
            type_correct += (type_pred[mask] == type_target[mask]).sum().item()
            type_total += int(mask.sum().item())

    pred_concat = np.concatenate(all_pred, axis=0).reshape(-1, 4)
    true_concat = np.concatenate(all_true, axis=0).reshape(-1, 4)
    prec, rec, f1 = tolerance_f1(pred_concat, true_concat, tol_frames=tol_frames)

    return {
        'loss': total_loss / max(total_samples, 1),
        'onset_loss': total_onset / max(total_samples, 1),
        'type_loss': total_type / max(total_samples, 1),
        'tol_precision': prec,
        'tol_recall': rec,
        'tol_f1': f1,
        'type_acc': type_correct / max(type_total, 1),
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
        'default_density_by_id': default_density_by_id.tolist(),
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
    parser.add_argument('--type-weight', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tol-frames', type=int, default=3,
                        help='Tolerance window (in frames) for F1; ~30 ms at 100 fps')
    return parser


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} | git_sha={_git_sha()} | seed={args.seed}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    built = build_dataloaders(args)
    model = build_model(args, built.onset_prior, device)
    criterion = build_loss(built.onset_prior, args).to(device)
    steps_per_epoch = max(1, len(built.train_loader))
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
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
        )
        # Validate with EMA weights
        val_metrics = validate(
            ema_model.module, built.val_loader, criterion, device,
            tol_frames=args.tol_frames,
        )

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"(onset={train_metrics['onset_loss']:.4f} "
            f"type={train_metrics['type_loss']:.4f}) "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"tol_f1={val_metrics['tol_f1']:.3f} "
            f"(P={val_metrics['tol_precision']:.3f} "
            f"R={val_metrics['tol_recall']:.3f}) "
            f"type_acc={val_metrics['type_acc']:.3f} "
            f"| lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Always overwrite 'last'
        save_checkpoint(
            checkpoint_dir / 'last_model.pt', epoch, model, ema_model,
            optimizer, scheduler, scaler, val_metrics,
            built.default_density_by_id, args,
        )

        if val_metrics['tol_f1'] > best_f1:
            best_f1 = val_metrics['tol_f1']
            save_checkpoint(
                checkpoint_dir / 'best_model.pt', epoch, model, ema_model,
                optimizer, scheduler, scaler, val_metrics,
                built.default_density_by_id, args,
            )
            logger.info(f"  -> New best (tol_f1={best_f1:.3f})")

    logger.info(f"Training complete. Best tol_f1: {best_f1:.3f}")


if __name__ == '__main__':
    main()
