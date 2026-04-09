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

    Normalisation: divide pos and neg terms separately so the background term
    cannot dominate when there are very few positive pixels per image.
    With ~5 tips in a 128×128 heatmap (16 384 pixels), normalising the full sum
    by n_pos alone made the background term ~3 000× too strong, causing the model
    to collapse to near-zero outputs.
    """
    eps  = 1e-6
    peak = (gt == 1.0).float()
    n_pos = peak.sum().clamp(min=1)
    n_neg = (1 - peak).sum().clamp(min=1)

    pos_loss = peak       * torch.pow(1 - pred, alpha) * torch.log(pred.clamp(min=eps))
    neg_loss = (1 - peak) * torch.pow(1 - gt,  beta)  * torch.pow(pred, alpha) * torch.log((1 - pred).clamp(min=eps))

    return -(pos_loss.sum() / n_pos + neg_loss.sum() / n_neg)


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

def run_epoch(model, loader, optimizer, device, train: bool,
              sparsity_weight: float = 5.0):
    model.train(train)
    total_tip = total_score = total_sparsity = total_n = 0.0

    with torch.set_grad_enabled(train):
        for batch in loader:
            imgs     = batch['image'].to(device)
            tip_gt   = batch['tip_hm'].to(device)
            score_gt = batch['score_map'].to(device)

            with torch.autocast(device_type=device, enabled=(device != 'cpu')):
                tip_hm, s_map = model(imgs)
                l_tip      = focal_loss(tip_hm, tip_gt)
                l_sparsity = tip_hm.mean()          # penalise broadly-hot heatmaps
                l_score    = score_loss(s_map, score_gt)
                loss       = l_tip + sparsity_weight * l_sparsity + 0.3 * l_score

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = imgs.size(0)
            total_tip      += l_tip.item()      * B
            total_score    += l_score.item()    * B
            total_sparsity += l_sparsity.item() * B
            total_n        += B

    n = max(total_n, 1)
    return total_tip / n, total_score / n, total_sparsity / n


def validate_recall(model, val_loader, device, threshold: float = 45.0,
                    max_preds: int = 50, hm_threshold: float = 0.35):
    """Compute recall, precision, and F1 at `threshold` px on the validation set.

    `max_preds` caps predictions per image (sorted by confidence) so dense
    heatmaps cannot game recall by predicting everywhere.
    """
    model.eval()
    all_recall = []
    total_tp = total_gt = total_pred = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs     = batch['image'].to(device)
            tip_hms, s_maps = model(imgs)

            for b in range(imgs.size(0)):
                meta    = {k: v[b] if hasattr(v, '__getitem__') else v
                           for k, v in batch['meta'].items()}
                scale   = float(meta['scale'])
                pad_x   = int(meta['pad_x'])
                pad_y   = int(meta['pad_y'])

                # Decode predictions: sort by confidence, keep top-max_preds
                hm    = heatmap_nms(tip_hms[b].squeeze())
                ratio = INPUT_SIZE / HEATMAP_SIZE
                peaks = []
                for hy, hx in zip(*torch.where(hm > hm_threshold)):
                    conf = hm[hy, hx].item()
                    lbx  = (hx.item() + 0.5) * ratio
                    lby  = (hy.item() + 0.5) * ratio
                    peaks.append((conf, (lbx - pad_x) / scale, (lby - pad_y) / scale))
                peaks.sort(key=lambda t: -t[0])
                pred_tips = [(x, y) for _, x, y in peaks[:max_preds]]

                # GT tips in original coords
                gt_hm = batch['tip_hm'][b].squeeze()
                gt_tips = []
                for hy, hx in zip(*torch.where(gt_hm > 0.99)):
                    lbx = (hx.item() + 0.5) * ratio
                    lby = (hy.item() + 0.5) * ratio
                    gt_tips.append((
                        (lbx - pad_x) / scale,
                        (lby - pad_y) / scale,
                    ))

                r = recall_at_threshold(pred_tips, gt_tips, threshold)
                all_recall.append(r)
                total_tp   += round(r * len(gt_tips))
                total_gt   += len(gt_tips)
                total_pred += len(pred_tips)

    recall    = sum(all_recall) / max(len(all_recall), 1)
    precision = total_tp / max(total_pred, 1)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return recall, precision, f1, total_pred / max(len(all_recall), 1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',    type=int,   default=60)
    parser.add_argument('--batch',     type=int,   default=32)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--workers',   type=int,   default=4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--out',       type=str,   default='checkpoints')
    parser.add_argument('--db',        type=str,   default=DB_URL)
    parser.add_argument('--images',    type=str,   default=IMAGES_DIR)
    parser.add_argument('--patience',  type=int,   default=10,
                        help='Stop if recall has not improved for this many epochs '
                             '(counted only after backbone unfreeze). 0 = disabled.')
    parser.add_argument('--sparsity-weight', type=float, default=5.0,
                        help='Weight for heatmap sparsity loss (default 5.0). '
                             'Higher values force fewer, more confident predictions.')
    parser.add_argument('--input-channels', type=int, default=4, choices=[3, 4],
                        help='3 = RGB only (no radial), 4 = RGB + radial (default)')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'mobilenet_v3_large'])
    parser.add_argument('--resume',    action='store_true',
                        help='Resume from checkpoints/last.pt')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = (
        'cuda'  if torch.cuda.is_available() else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f'Device: {device}')

    # ── data ──────────────────────────────────────────────────────────────
    use_radial = args.input_channels == 4
    train_ds = ArrowDataset(args.db, args.images, augment=True,
                            val_split=args.val_split, is_val=False, use_radial=use_radial)
    val_ds   = ArrowDataset(args.db, args.images, augment=False,
                            val_split=args.val_split, is_val=True, use_radial=use_radial)

    # Use the val split for recall tracking — large enough now to be stable.
    recall_ds = val_ds

    print(f'Train: {len(train_ds)} images   Val: {len(val_ds)} images   Recall-set: {len(recall_ds)} images')
    if len(train_ds) == 0:
        raise RuntimeError('No training samples found — check DB connection and annotations.')

    _persistent = args.workers > 0
    _prefetch   = 4 if args.workers > 0 else None
    train_loader  = DataLoader(train_ds,  batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=_persistent, prefetch_factor=_prefetch)
    val_loader    = DataLoader(val_ds,    batch_size=args.batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=_persistent, prefetch_factor=_prefetch)
    recall_loader = val_loader

    # ── model + optimiser ──────────────────────────────────────────────────
    model = ArrowDetector(in_channels=args.input_channels, backbone=args.backbone).to(device)
    print(f'Input channels: {args.input_channels}  Backbone: {args.backbone}  Sparsity λ: {args.sparsity_weight}')
    # torch.compile MPS support is still maturing — skip for now.
    # Re-enable once PyTorch MPS Metal shader bugs are resolved.
    # model = torch.compile(model, fullgraph=False)

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
    log_path    = os.path.join(args.out, 'train_log.csv')
    best_recall = 0.0
    start_epoch = 1
    UNFREEZE_EPOCH = 10
    epochs_no_improve = 0

    # ── resume ─────────────────────────────────────────────────────────────
    if args.resume:
        last_path = os.path.join(args.out, 'last.pt')
        if not os.path.exists(last_path):
            print('No last.pt found — starting from scratch.')
        else:
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            start_epoch = ckpt['epoch'] + 1
            best_recall = ckpt.get('f1', ckpt.get('recall', 0.0))
            # Restore backbone freeze state and rebuild optimizer to match
            if start_epoch <= UNFREEZE_EPOCH:
                pass  # backbone still frozen, optimizer already correct
            else:
                set_backbone_grad(True)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - UNFREEZE_EPOCH
                )
                # Fast-forward scheduler to the right position
                for _ in range(start_epoch - UNFREEZE_EPOCH - 1):
                    scheduler.step()
            # Don't restore optimizer state — parameter groups may differ between
            # warmup (head-only) and fine-tune (full model) phases.
            print(f'Resumed from epoch {ckpt["epoch"]} (best recall so far: {best_recall:.3f})')

    with open(log_path, 'a' if args.resume else 'w', newline='') as f:
        writer = csv.writer(f)
        if not args.resume:
            writer.writerow(['epoch', 'train_tip', 'train_score', 'train_sparsity',
                             'val_tip', 'val_score', 'val_sparsity',
                             'val_recall', 'val_precision', 'val_f1', 'val_pred_count'])

        for epoch in range(start_epoch, args.epochs + 1):

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

            tr_tip, tr_score, tr_sparsity = run_epoch(
                model, train_loader, optimizer, device, train=True,
                sparsity_weight=args.sparsity_weight,
            )
            va_tip, va_score, va_sparsity = run_epoch(
                model, val_loader,   optimizer, device, train=False,
                sparsity_weight=args.sparsity_weight,
            )
            # Measure recall, precision, F1 on the recall set.
            # max_preds=50 prevents dense heatmaps from gaming recall.
            recall, precision, f1, pred_count = validate_recall(
                model, recall_loader, device
            )
            scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch {epoch:3d}/{args.epochs}  '
                f'train tip={tr_tip:.4f} score={tr_score:.4f} sparse={tr_sparsity:.4f}  '
                f'val tip={va_tip:.4f}  '
                f'recall@45={recall:.3f} prec={precision:.3f} F1={f1:.3f} '
                f'n_pred={pred_count:.1f}  lr={lr:.2e}'
            )
            writer.writerow([epoch, tr_tip, tr_score, tr_sparsity,
                             va_tip, va_score, va_sparsity,
                             recall, precision, f1, pred_count])
            f.flush()

            # Save checkpoints — use F1 so dense false-positive heatmaps are
            # penalised through low precision rather than rewarded for recall.
            ckpt = {
                'epoch':       epoch,
                'model':       model.state_dict(),
                'optim':       optimizer.state_dict(),
                'recall':      recall,
                'precision':   precision,
                'f1':          f1,
                'in_channels': args.input_channels,
            }
            torch.save(ckpt, os.path.join(args.out, 'last.pt'))
            if f1 >= best_recall:   # best_recall tracks F1 for checkpoint selection
                best_recall = f1
                torch.save(ckpt, os.path.join(args.out, 'best.pt'))
                print(f'  → new best F1: {best_recall:.3f}  (recall={recall:.3f} prec={precision:.3f})')
                if epoch > UNFREEZE_EPOCH:
                    epochs_no_improve = 0
            elif epoch > UNFREEZE_EPOCH and args.patience > 0:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f'Early stopping: no improvement for {args.patience} epochs.')
                    break

    print(f'\nTraining complete. Best val F1: {best_recall:.3f}')
    print(f'Checkpoints saved to: {args.out}/')


if __name__ == '__main__':
    main()
