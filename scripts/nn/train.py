"""
Training script for ArrowDetector.

Usage:
    python train.py [--epochs 60] [--batch 8] [--lr 1e-3] [--out checkpoints/]

Outputs (in --out directory):
    best.pt           — best checkpoint (by val recall@45px)
    last.pt           — checkpoint after final epoch
    train_log.csv     — per-epoch metrics
"""

import argparse
import os
import csv
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ArrowDataset, HEATMAP_SIZE, INPUT_SIZE, DB_URL, IMAGES_DIR
from model   import ArrowDetector, heatmap_nms


# ── losses ────────────────────────────────────────────────────────────────────

def focal_loss(pred: torch.Tensor, gt: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet focal loss for heatmap regression.

    pred, gt: (B, 1, H, W) in [0, 1].
    At GT peak pixels: loss = (1-pred)^alpha * log(pred).
    At non-peak pixels: loss = (1-gt)^beta * pred^alpha * log(1-pred).
    """
    eps  = 1e-6
    peak = (gt == 1.0).float()

    pos_loss = peak         * torch.pow(1 - pred, alpha) * torch.log(pred.clamp(min=eps))
    neg_loss = (1 - peak)   * torch.pow(1 - gt,  beta)  * torch.pow(pred, alpha) * torch.log((1 - pred).clamp(min=eps))

    n_pos = peak.sum().clamp(min=1)
    return -(pos_loss + neg_loss).sum() / n_pos


def score_loss(score_map: torch.Tensor, score_gt: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on the score map, masked to GT tip positions.

    score_map : (B, 11, H, W) logits
    score_gt  : (B, H, W) int64, -1 = ignore
    """
    B, C, H, W = score_map.shape
    logits = score_map.permute(0, 2, 3, 1).reshape(-1, C)   # (B*H*W, 11)
    labels = score_gt.reshape(-1)                            # (B*H*W,)
    return F.cross_entropy(logits, labels, ignore_index=-1)


# ── validation metric ─────────────────────────────────────────────────────────

def recall_at_threshold(pred_tips: list, gt_tips: list,
                        threshold: float = 45.0) -> float:
    """Greedy bipartite recall: fraction of GT tips matched within `threshold` px."""
    if not gt_tips:
        return 1.0
    if not pred_tips:
        return 0.0

    import numpy as np
    pairs = []
    for gi, gt in enumerate(gt_tips):
        for pi, pd in enumerate(pred_tips):
            d = math.hypot(gt[0] - pd[0], gt[1] - pd[1])
            if d <= threshold:
                pairs.append((d, gi, pi))
    pairs.sort()

    matched_g, matched_p = set(), set()
    for d, gi, pi in pairs:
        if gi not in matched_g and pi not in matched_p:
            matched_g.add(gi)
            matched_p.add(pi)

    return len(matched_g) / len(gt_tips)


# ── train / validate loops ────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_tip = total_nock = total_score = total_n = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            imgs      = batch['image'].to(device)
            tip_gt    = batch['tip_hm'].to(device)
            nock_gt   = batch['nock_hm'].to(device)
            score_gt  = batch['score_map'].to(device)

            tip_hm, nock_hm, s_map = model(imgs)

            l_tip   = focal_loss(tip_hm,  tip_gt)
            l_nock  = focal_loss(nock_hm, nock_gt)
            l_score = score_loss(s_map, score_gt)
            loss    = l_tip + 0.5 * l_nock + 0.3 * l_score

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = imgs.size(0)
            total_tip   += l_tip.item()   * B
            total_nock  += l_nock.item()  * B
            total_score += l_score.item() * B
            total_n     += B

    n = max(total_n, 1)
    return total_tip / n, total_nock / n, total_score / n


def validate_recall(model, val_loader, device, threshold: float = 45.0):
    """Compute recall@45px in original image coordinates on the validation set."""
    model.eval()
    all_recall = []

    with torch.no_grad():
        for batch in val_loader:
            imgs     = batch['image'].to(device)
            tip_hms, _, s_maps = model(imgs)

            for b in range(imgs.size(0)):
                meta    = {k: v[b] if hasattr(v, '__getitem__') else v
                           for k, v in batch['meta'].items()}
                scale   = float(meta['scale'])
                pad_x   = int(meta['pad_x'])
                pad_y   = int(meta['pad_y'])

                # Decode predictions to original coords
                hm    = heatmap_nms(tip_hms[b].squeeze())
                ratio = INPUT_SIZE / HEATMAP_SIZE
                pred_tips = []
                for hy, hx in zip(*torch.where(hm > 0.35)):
                    lbx = (hx.item() + 0.5) * ratio
                    lby = (hy.item() + 0.5) * ratio
                    pred_tips.append((
                        (lbx - pad_x) / scale,
                        (lby - pad_y) / scale,
                    ))

                # GT tips in original coords (read from score_map peaks)
                sm = batch['score_map'][b]                   # (128, 128)
                gt_hm = batch['tip_hm'][b].squeeze()         # (128, 128)
                gt_tips = []
                for hy, hx in zip(*torch.where(gt_hm > 0.9)):
                    lbx = (hx.item() + 0.5) * ratio
                    lby = (hy.item() + 0.5) * ratio
                    gt_tips.append((
                        (lbx - pad_x) / scale,
                        (lby - pad_y) / scale,
                    ))

                all_recall.append(recall_at_threshold(pred_tips, gt_tips, threshold))

    return sum(all_recall) / max(len(all_recall), 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',    type=int,   default=60)
    parser.add_argument('--batch',     type=int,   default=8)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--workers',   type=int,   default=4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--out',       type=str,   default='checkpoints')
    parser.add_argument('--db',        type=str,   default=DB_URL)
    parser.add_argument('--images',    type=str,   default=IMAGES_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = (
        'cuda'  if torch.cuda.is_available() else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f'Device: {device}')

    # ── data ──────────────────────────────────────────────────────────────
    train_ds = ArrowDataset(args.db, args.images, augment=True,
                            val_split=args.val_split, is_val=False)
    val_ds   = ArrowDataset(args.db, args.images, augment=False,
                            val_split=args.val_split, is_val=True)

    print(f'Train: {len(train_ds)} images   Val: {len(val_ds)} images')
    if len(train_ds) == 0:
        raise RuntimeError('No training samples found — check DB connection and annotations.')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── model + optimiser ──────────────────────────────────────────────────
    model = ArrowDetector().to(device)

    # Freeze backbone for first 10 epochs, then unfreeze
    def set_backbone_grad(requires_grad: bool):
        for stage in (model.stage2, model.stage3, model.stage4, model.stage5):
            for p in stage.parameters():
                p.requires_grad_(requires_grad)

    set_backbone_grad(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── training loop ──────────────────────────────────────────────────────
    log_path  = os.path.join(args.out, 'train_log.csv')
    best_recall = 0.0
    UNFREEZE_EPOCH = 10

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_tip', 'train_nock', 'train_score',
                         'val_tip', 'val_nock', 'val_score', 'val_recall'])

        for epoch in range(1, args.epochs + 1):

            # Unfreeze backbone after warm-up
            if epoch == UNFREEZE_EPOCH + 1:
                print(f'Epoch {epoch}: unfreezing backbone')
                set_backbone_grad(True)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - UNFREEZE_EPOCH
                )

            tr_tip, tr_nock, tr_score = run_epoch(
                model, train_loader, optimizer, device, train=True
            )
            va_tip, va_nock, va_score = run_epoch(
                model, val_loader,   optimizer, device, train=False
            )
            recall = validate_recall(model, val_loader, device)
            scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch:3d}/{args.epochs}  '
                f'train tip={tr_tip:.4f} nock={tr_nock:.4f} score={tr_score:.4f}  '
                f'val tip={va_tip:.4f} recall@45={recall:.3f}  lr={lr:.2e}'
            )
            writer.writerow([epoch, tr_tip, tr_nock, tr_score,
                             va_tip, va_nock, va_score, recall])
            f.flush()

            # Save checkpoints
            ckpt = {
                'epoch':  epoch,
                'model':  model.state_dict(),
                'optim':  optimizer.state_dict(),
                'recall': recall,
            }
            torch.save(ckpt, os.path.join(args.out, 'last.pt'))
            if recall >= best_recall:
                best_recall = recall
                torch.save(ckpt, os.path.join(args.out, 'best.pt'))
                print(f'  → new best recall: {best_recall:.3f}')

    print(f'\nTraining complete. Best val recall@45px: {best_recall:.3f}')
    print(f'Checkpoints saved to: {args.out}/')


if __name__ == '__main__':
    main()
