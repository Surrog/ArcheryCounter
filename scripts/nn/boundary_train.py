"""
Training script for BoundaryDetector.

Usage:
    python boundary_train.py [--epochs 60] [--batch 8] [--lr 1e-3]
                             [--out boundary_checkpoints/] [--db DB_URL]

Outputs (in --out directory):
    best.pt         — best checkpoint by val IoU
    last.pt         — checkpoint after final epoch
    train_log.csv   — per-epoch metrics
"""

import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from boundary_dataset import BoundaryDataset, DB_URL, IMAGES_DIR
from boundary_model import BoundaryDetector


# ── losses ────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss. pred: sigmoid probabilities, target: binary float."""
    pred_f   = pred.view(-1)
    target_f = target.view(-1)
    intersection = (pred_f * target_f).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_f.sum() + target_f.sum() + smooth)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    return bce + dice


# ── metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou(logits: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= threshold).float()
    intersection = (pred * target).sum().item()
    union = ((pred + target) >= 1).float().sum().item()
    return intersection / union if union > 0 else 1.0


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('cpu')
    )
    print(f'Device: {device}')

    train_ds = BoundaryDataset('train', db_url=args.db, images_dir=args.images, augment=True)
    val_ds   = BoundaryDataset('val',   db_url=args.db, images_dir=args.images, augment=False)
    print(f'Train: {len(train_ds)} images   Val: {len(val_ds)} images')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = BoundaryDetector().to(device)

    if args.warm_start and os.path.exists(args.warm_start):
        state = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(state['model'])
        print(f'Warm start from {args.warm_start} (epoch {state.get("epoch")}, IoU={state.get("val_iou", 0):.4f})')

    # Phase 1: freeze backbone, train decoder only
    for p in model.enc0.parameters(): p.requires_grad_(False)
    for p in model.enc1.parameters(): p.requires_grad_(False)
    for p in model.enc2.parameters(): p.requires_grad_(False)
    for p in model.enc3.parameters(): p.requires_grad_(False)
    for p in model.enc4.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'train_log.csv')
    best_iou = 0.0
    unfreeze_done = False

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'lr'])

        for epoch in range(1, args.epochs + 1):

            # Unfreeze backbone after 20% of epochs
            if not unfreeze_done and epoch > max(1, args.epochs // 5):
                print(f'  [epoch {epoch}] Unfreezing backbone')
                for p in model.parameters():
                    p.requires_grad_(True)
                # Rebuild optimizer with lower lr for backbone
                optimizer = torch.optim.AdamW([
                    {'params': list(model.enc0.parameters()) +
                               list(model.enc1.parameters()) +
                               list(model.enc2.parameters()) +
                               list(model.enc3.parameters()) +
                               list(model.enc4.parameters()),
                     'lr': args.lr * 0.1},
                    {'params': list(model.dec4.parameters()) +
                               list(model.dec3.parameters()) +
                               list(model.dec2.parameters()) +
                               list(model.dec1.parameters()) +
                               list(model.final_up.parameters()),
                     'lr': args.lr},
                ], weight_decay=1e-4)
                remaining = args.epochs - epoch + 1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-6,
                )
                unfreeze_done = True

            # ── train ────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                imgs  = batch['image'].to(device)
                masks = batch['mask'].to(device)
                logits = model(imgs)
                loss = bce_dice_loss(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not unfreeze_done:
                    scheduler.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            if unfreeze_done:
                scheduler.step()

            # ── validate ─────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            val_iou  = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs  = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                    logits = model(imgs)
                    val_loss += bce_dice_loss(logits, masks).item()
                    val_iou  += compute_iou(logits, masks)
            val_loss /= len(val_loader)
            val_iou  /= len(val_loader)

            current_lr = optimizer.param_groups[-1]['lr']
            print(
                f'Epoch {epoch:3d}/{args.epochs}  '
                f'train={train_loss:.4f}  val={val_loss:.4f}  '
                f'IoU={val_iou:.4f}  lr={current_lr:.2e}'
            )
            writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}',
                              f'{val_iou:.4f}', f'{current_lr:.2e}'])

            # Save checkpoints
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_iou': val_iou,
            }
            torch.save(state, os.path.join(args.out, 'last.pt'))
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(state, os.path.join(args.out, 'best.pt'))
                print(f'  → new best IoU: {best_iou:.4f}')

    print(f'\nTraining complete. Best val IoU: {best_iou:.4f}')
    print(f'Checkpoints saved to: {args.out}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',  type=int,   default=60)
    p.add_argument('--batch',   type=int,   default=4)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--workers', type=int,   default=0)
    p.add_argument('--out',     type=str,   default='boundary_checkpoints/')
    p.add_argument('--db',      type=str,   default=DB_URL)
    p.add_argument('--images',      type=str,   default=IMAGES_DIR)
    p.add_argument('--warm-start',  type=str,   default='',
                   help='Path to checkpoint to warm-start from (model weights only)')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
