"""
Training script for step chart model.

Designed to run on Google Colab with T4 GPU.
Uses mixed precision, checkpointing, and Google Drive integration.

Usage (from Colab):
    !python -m ml.train \
        --data-dir /content/drive/MyDrive/stepageddon/training_data \
        --checkpoint-dir /content/drive/MyDrive/stepageddon/checkpoints \
        --epochs 80 --batch-size 32
"""

import argparse
import copy
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from ml.model import StepChartModel, MultiHeadFocalLoss
from ml.dataset import StepChartDataset, compute_pos_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


HEAD_NAMES = ('tap', 'hold_state', 'hold_end')


class ModelEMA:
    """Exponential moving average of model weights.

    Typically buys +1–2% F1 on small audio datasets for near-free. We
    apply the EMA weights for validation (and for saving the best model),
    but train with the raw weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(v)

    def apply_to(self, model: nn.Module):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module):
        if self._backup is None:
            return
        model.load_state_dict(self._backup, strict=True)
        self._backup = None


def stratified_split_by_song(
    full_dataset, val_split: float, seed: int = 42,
) -> tuple:
    """Split the dataset so that no song appears in both train and val.

    Songs are grouped by (song_title, artist) — the manifest stores both,
    and the pair is specific enough to catch duplicates across difficulty
    files. Random splitting by entry index leaks badly: the same song at
    beginner and challenge end up on opposite sides of the split.
    """
    groups: dict = {}
    for i, e in enumerate(full_dataset.entries):
        key = (e.get('song_title', ''), e.get('artist', ''))
        groups.setdefault(key, []).append(i)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val_songs = max(1, int(round(len(keys) * val_split)))
    val_keys = set(keys[:n_val_songs])

    train_idx, val_idx = [], []
    for k in keys:
        (val_idx if k in val_keys else train_idx).extend(groups[k])

    logger.info(
        f"Stratified split: {len(keys)} songs → "
        f"{len(keys) - n_val_songs} train / {n_val_songs} val "
        f"({len(train_idx)} / {len(val_idx)} charts)"
    )
    return Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a boolean [..., T] mask by ±radius along the last dim."""
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(1)  # [N, 1, T]
    x = F.max_pool1d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return x.squeeze(1) > 0.5


def make_curriculum_loader(
    train_subset,
    full_dataset,
    batch_size: int,
    allowed_ids: set = None,
    num_workers: int = 2,
):
    """Build a train DataLoader that only yields entries whose
    difficulty_id is in ``allowed_ids`` (or all entries when None).

    Used for the difficulty curriculum: early epochs train on
    beginner/easy/medium only so the harder charts (wildly different
    density) don't dominate focal loss early.
    """
    from torch.utils.data import WeightedRandomSampler

    indices = train_subset.indices if isinstance(train_subset, Subset) else list(range(len(train_subset)))
    n_entries = len(full_dataset.entries)
    diff_of = lambda i: full_dataset.entries[i % n_entries]['difficulty_id']

    weights = np.ones(len(indices), dtype=np.float64)
    if allowed_ids is not None:
        allowed_ids = set(int(x) for x in allowed_ids)
        for k, i in enumerate(indices):
            if diff_of(i) not in allowed_ids:
                weights[k] = 0.0
        active = int((weights > 0).sum())
        logger.info(
            f"Curriculum: allowed ids={sorted(allowed_ids)}, "
            f"{active}/{len(indices)} chunks active"
        )
    else:
        # Balanced by difficulty across *all* classes in the split.
        diff_ids = np.array([diff_of(i) for i in indices], dtype=np.int64)
        counts = np.bincount(diff_ids, minlength=5).astype(np.float64)
        inv = np.where(counts > 0, 1.0 / counts, 0.0)
        weights = inv[diff_ids]

    if weights.sum() <= 0:
        raise ValueError("Curriculum sampler has no active entries")

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(indices),
        replacement=True,
    )
    return DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


def tolerant_placement_f1(
    placement_logits: torch.Tensor,
    placement_targets: torch.Tensor,
    radius: int = 2,
    threshold: float = 0.5,
) -> tuple:
    """Placement F1 computed with ±`radius` frame tolerance.

    Exact-frame F1 is brutal for audio models because label timing has
    ~20 ms jitter relative to the mel frame grid. ±2 frames ≈ ±20 ms is
    the smallest tolerance that doesn't penalize pure alignment noise.

    Returns (tp, fp, fn) accumulators (Python ints) for the current batch.
    """
    probs = torch.sigmoid(placement_logits)
    pred = probs > threshold              # [B, T]
    gt = placement_targets > 0.5          # [B, T]
    dilated_gt = _dilate(gt, radius)
    dilated_pred = _dilate(pred, radius)
    tp = int((pred & dilated_gt).sum().item())
    fp = int((pred & ~dilated_gt).sum().item())
    fn = int((gt & ~dilated_pred).sum().item())
    return tp, fp, fn


def _tap_hit_rate(logits: dict, targets: torch.Tensor, thr: float = 0.5) -> tuple:
    """Fraction of positive tap frames where the tap head fires above thr.

    Analogue of the old "accuracy on note frames"; used as a cheap sanity
    signal during training.
    """
    tap_probs = torch.sigmoid(logits['tap'])            # [B, T, 4]
    tap_target = targets[..., 0]                        # [B, T, 4]
    pos = tap_target > 0.5
    if pos.any():
        correct = ((tap_probs > thr) & pos).sum().item()
        total = pos.sum().item()
    else:
        correct, total = 0, 0
    return correct, total


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, ema=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_notes = 0
    total_samples = 0
    # Per-head positive/negative counts accumulated over the epoch so we
    # can refresh pos_weights at epoch boundaries from the distribution
    # the model actually saw (via WeightedRandomSampler), not the raw
    # manifest estimate.
    head_pos = {h: 0 for h in HEAD_NAMES}
    head_neg = {h: 0 for h in HEAD_NAMES}
    placement_pos = 0
    placement_neg = 0

    for mel, difficulty, density, targets, placement in loader:
        mel = mel.to(device)
        difficulty = difficulty.to(device)
        density = density.to(device)
        targets = targets.to(device)        # [B, T, 4, 3]
        placement = placement.to(device)    # [B, T]

        optimizer.zero_grad()

        with autocast('cuda'):
            logits = model(mel, difficulty, density)  # dict of [B, T, *]
            loss = criterion(logits, targets, placement, density)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * mel.size(0)
        total_samples += mel.size(0)

        c, t = _tap_hit_rate(logits, targets)
        total_correct += c
        total_notes += t

        # Accumulate per-head positive counts (cheap; runs on device).
        with torch.no_grad():
            for i, head in enumerate(HEAD_NAMES):
                t_head = targets[..., i]
                n = t_head.numel()
                p = int(t_head.sum().item())
                head_pos[head] += p
                head_neg[head] += n - p
            placement_pos += int(placement.sum().item())
            placement_neg += int(placement.numel() - placement.sum().item())

    avg_loss = total_loss / total_samples
    note_acc = total_correct / max(total_notes, 1)
    return (
        avg_loss, note_acc,
        {'head_pos': head_pos, 'head_neg': head_neg,
         'placement_pos': placement_pos, 'placement_neg': placement_neg},
    )


@torch.no_grad()
def validate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_notes = 0
    total_samples = 0

    # Per-head binary metrics at `threshold` on sigmoid probabilities.
    n_heads = len(HEAD_NAMES)
    tp = torch.zeros(n_heads, device=device)
    fp = torch.zeros(n_heads, device=device)
    fn = torch.zeros(n_heads, device=device)
    # Tolerant placement metrics (±2 frame ≈ ±20 ms jitter).
    tol_tp = 0
    tol_fp = 0
    tol_fn = 0

    for mel, difficulty, density, targets, placement in loader:
        mel = mel.to(device)
        difficulty = difficulty.to(device)
        density = density.to(device)
        targets = targets.to(device)
        placement = placement.to(device)

        with autocast('cuda'):
            logits = model(mel, difficulty, density)
            loss = criterion(logits, targets, placement, density)

        total_loss += loss.item() * mel.size(0)
        total_samples += mel.size(0)

        c, t = _tap_hit_rate(logits, targets, thr=threshold)
        total_correct += c
        total_notes += t

        for i, head in enumerate(HEAD_NAMES):
            probs = torch.sigmoid(logits[head])
            pred = probs > threshold
            gt = targets[..., i] > 0.5
            tp[i] += (pred & gt).sum()
            fp[i] += (pred & ~gt).sum()
            fn[i] += (~pred & gt).sum()

        if 'placement' in logits:
            a, b, c = tolerant_placement_f1(
                logits['placement'], placement, radius=2, threshold=threshold,
            )
            tol_tp += a
            tol_fp += b
            tol_fn += c

    avg_loss = total_loss / total_samples
    note_acc = total_correct / max(total_notes, 1)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_dict = {HEAD_NAMES[i]: f1[i].item() for i in range(n_heads)}
    if (tol_tp + tol_fp) and (tol_tp + tol_fn):
        tol_p = tol_tp / (tol_tp + tol_fp)
        tol_r = tol_tp / (tol_tp + tol_fn)
        tol_f1 = 2 * tol_p * tol_r / (tol_p + tol_r + 1e-8)
    else:
        tol_f1 = 0.0
    f1_dict['placement_tol2'] = float(tol_f1)
    return avg_loss, note_acc, f1_dict


def main():
    parser = argparse.ArgumentParser(description='Train step chart model')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with preprocessed training data')
    parser.add_argument('--checkpoint-dir', type=str, default='ml/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--chunk-frames', type=int, default=500,
                        help='Frames per training chunk (500=5s at 100fps)')
    parser.add_argument('--samples-per-entry', type=int, default=2,
                        help='Random chunks per song per epoch '
                             '(>1 compensates for chunk-boundary gaps)')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay for shadow weights (0 to disable)')
    parser.add_argument('--curriculum-epochs', type=int, default=10,
                        help='First N epochs sample only difficulty_ids '
                             'in --curriculum-easy-ids; 0 disables')
    parser.add_argument('--curriculum-easy-ids', type=str, default='0,1,2',
                        help='Comma-separated difficulty_ids allowed '
                             'during the curriculum warmup phase')
    parser.add_argument('--early-stop-patience', type=int, default=8,
                        help='Stop training after N epochs with no val '
                             'placement-F1 improvement (0 to disable)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / 'manifest.json'

    # Create dataset
    full_dataset = StepChartDataset(
        data_dir=str(data_dir),
        manifest_path=str(manifest_path),
        chunk_frames=args.chunk_frames,
        is_train=True,
        samples_per_entry=args.samples_per_entry,
    )

    # Stratified train/val split — same song cannot appear in both
    # partitions (charts of the same song at different difficulties are
    # strongly correlated and would leak).
    train_dataset, val_dataset = stratified_split_by_song(
        full_dataset, args.val_split, seed=42,
    )
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # Initial training loader: either the curriculum warmup or the full
    # balanced mix. Swapped to the full mix after curriculum_epochs.
    if args.curriculum_epochs > 0:
        easy_ids = {int(x) for x in args.curriculum_easy_ids.split(',') if x.strip()}
        train_loader = make_curriculum_loader(
            train_dataset, full_dataset, args.batch_size,
            allowed_ids=easy_ids,
        )
    else:
        train_loader = make_curriculum_loader(
            train_dataset, full_dataset, args.batch_size,
            allowed_ids=None,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    logger.info(f"Train: {n_train} examples, Val: {n_val} examples")

    # Compute per-head positive weights for BCE focal loss.
    logger.info("Computing head pos_weights...")
    pos_weights, placement_pos_weight = compute_pos_weights(str(manifest_path), str(data_dir))
    pos_weights = pos_weights.to(device)

    # Create model
    model = StepChartModel(
        n_mels=82,  # 80 mel bins + 2 rhythm channels
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        n_difficulties=5,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = MultiHeadFocalLoss(
        pos_weights=pos_weights,
        gamma=args.focal_gamma,
        placement_pos_weight=placement_pos_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_placement_f1 = -1.0
    epochs_since_improvement = 0
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay and args.ema_decay > 0 else None

    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed at epoch {start_epoch}")

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Curriculum graduation: unlock all difficulties after the warmup.
        if args.curriculum_epochs > 0 and epoch == args.curriculum_epochs:
            logger.info(
                f"Graduating from curriculum at epoch {epoch}: "
                f"unlocking all difficulties"
            )
            train_loader = make_curriculum_loader(
                train_dataset, full_dataset, args.batch_size,
                allowed_ids=None,
            )
        t0 = time.time()

        train_loss, train_acc, stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            ema=ema,
        )
        # Refresh pos_weights from observed distribution. This corrects
        # for drift between the manifest-sample estimate and what the
        # WeightedRandomSampler / mixup / curriculum actually feed in.
        new_pw = []
        for head in HEAD_NAMES:
            pos = max(stats['head_pos'][head], 1)
            neg = stats['head_neg'][head]
            new_pw.append(float(np.clip(neg / pos, 1.0, 200.0)))
        new_pw_t = torch.tensor(new_pw, dtype=torch.float32, device=device)
        criterion.pos_weights = new_pw_t
        new_placement_pw = float(np.clip(
            stats['placement_neg'] / max(stats['placement_pos'], 1),
            1.0, 200.0,
        ))
        criterion.placement_pos_weight = torch.tensor(
            new_placement_pw, dtype=torch.float32, device=device,
        )
        logger.info(
            f"[epoch {epoch+1}] refreshed pos_weights tap/hs/he="
            f"{[round(x,2) for x in new_pw]} placement={new_placement_pw:.2f}"
        )
        # Validate on EMA weights if available — these are the weights
        # we'd actually use at inference time, and they're typically
        # smoother than the raw training weights.
        if ema is not None:
            ema.apply_to(model)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        if ema is not None:
            ema.restore(model)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} note_acc={train_acc:.3f} "
            f"| val_loss={val_loss:.4f} note_acc={val_acc:.3f} "
            f"| tap_f1={val_f1['tap']:.3f} holdst_f1={val_f1['hold_state']:.3f} holdend_f1={val_f1['hold_end']:.3f} "
            f"plc_f1±2={val_f1.get('placement_tol2', 0.0):.3f} "
            f"| lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # Track best by tolerant placement F1 — the metric that actually
        # correlates with chart feel, not raw val loss.
        current_f1 = val_f1.get('placement_tol2', 0.0)
        if current_f1 > best_val_placement_f1:
            best_val_placement_f1 = current_f1
            best_val_loss = val_loss
            epochs_since_improvement = 0
            best_path = checkpoint_dir / 'best_model.pt'
            # Save EMA weights when available (those are the deployable ones).
            save_state = (
                ema.shadow if ema is not None else model.state_dict()
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_state,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'args': vars(args),
            }, best_path)
            logger.info(
                f"New best model (val placement_tol2_f1={current_f1:.4f}, "
                f"val_loss={val_loss:.4f})"
            )
        else:
            epochs_since_improvement += 1
            if (
                args.early_stop_patience > 0
                and epochs_since_improvement >= args.early_stop_patience
            ):
                logger.info(
                    f"Early stopping: no improvement in "
                    f"{epochs_since_improvement} epochs "
                    f"(best placement_tol2_f1={best_val_placement_f1:.4f})"
                )
                break

    logger.info(
        f"Training complete. Best val placement_tol2_f1: "
        f"{best_val_placement_f1:.4f} (val_loss={best_val_loss:.4f})"
    )


if __name__ == '__main__':
    main()
