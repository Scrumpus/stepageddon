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
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from ml.model import StepChartModel, FocalLoss
from ml.dataset import StepChartDataset, compute_class_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_notes = 0
    total_samples = 0

    for mel, difficulty, labels in loader:
        mel = mel.to(device)
        difficulty = difficulty.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            logits = model(mel, difficulty)  # [B, T, 4, 4]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * mel.size(0)
        total_samples += mel.size(0)

        # Accuracy on note frames only (not 'none' frames)
        preds = logits.argmax(dim=-1)  # [B, T, 4]
        note_mask = labels > 0
        if note_mask.any():
            total_correct += (preds[note_mask] == labels[note_mask]).sum().item()
            total_notes += note_mask.sum().item()

    avg_loss = total_loss / total_samples
    note_acc = total_correct / max(total_notes, 1)
    return avg_loss, note_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_notes = 0
    total_samples = 0

    # Per-class metrics
    tp = torch.zeros(4, device=device)
    fp = torch.zeros(4, device=device)
    fn = torch.zeros(4, device=device)

    for mel, difficulty, labels in loader:
        mel = mel.to(device)
        difficulty = difficulty.to(device)
        labels = labels.to(device)

        with autocast('cuda'):
            logits = model(mel, difficulty)
            loss = criterion(logits, labels)

        total_loss += loss.item() * mel.size(0)
        total_samples += mel.size(0)

        preds = logits.argmax(dim=-1)
        note_mask = labels > 0
        if note_mask.any():
            total_correct += (preds[note_mask] == labels[note_mask]).sum().item()
            total_notes += note_mask.sum().item()

        # Per-class precision/recall
        for c in range(4):
            tp[c] += ((preds == c) & (labels == c)).sum()
            fp[c] += ((preds == c) & (labels != c)).sum()
            fn[c] += ((preds != c) & (labels == c)).sum()

    avg_loss = total_loss / total_samples
    note_acc = total_correct / max(total_notes, 1)

    # F1 per class
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    class_names = ['none', 'tap', 'hold_start', 'hold_end']
    f1_dict = {class_names[i]: f1[i].item() for i in range(4)}

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
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
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
    )

    # Train/val split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    logger.info(f"Train: {n_train} examples, Val: {n_val} examples")

    # Compute class weights
    logger.info("Computing class weights...")
    class_weights = compute_class_weights(str(manifest_path), str(data_dir))
    class_weights = class_weights.to(device)

    # Create model
    model = StepChartModel(
        n_mels=80,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        n_difficulties=5,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
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
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} note_acc={train_acc:.3f} "
            f"| val_loss={val_loss:.4f} note_acc={val_acc:.3f} "
            f"| tap_f1={val_f1['tap']:.3f} hold_f1={val_f1['hold_start']:.3f} "
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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'args': vars(args),
            }, best_path)
            logger.info(f"New best model (val_loss={val_loss:.4f})")

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
