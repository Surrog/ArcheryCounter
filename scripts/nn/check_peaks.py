"""Quick diagnostic: show heatmap peak confidences for a few images."""
import torch, json
from dataset import ArrowDataset, HEATMAP_SIZE
from model import ArrowDetector, heatmap_nms

ds = ArrowDataset(augment=False, is_val=False)
model = ArrowDetector()
ckpt = torch.load('checkpoints/best.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

all_gt_conf, all_fp_conf = [], []

with torch.no_grad():
    for i in range(min(20, len(ds))):
        sample = ds[i]
        img = sample['image'].unsqueeze(0)
        gt_hm = sample['tip_hm'].squeeze()   # (H, W)
        tip_hm, _ = model(img)
        hm = heatmap_nms(tip_hm.squeeze())   # (H, W)

        gt_mask = gt_hm > 0.99
        for hy, hx in zip(*torch.where(gt_mask)):
            all_gt_conf.append(hm[hy, hx].item())
        for hy, hx in zip(*torch.where((hm > 0.1) & ~gt_mask)):
            all_fp_conf.append(hm[hy, hx].item())

if all_gt_conf:
    all_gt_conf.sort(reverse=True)
    print(f"GT peaks  n={len(all_gt_conf):3d}  min={min(all_gt_conf):.3f}  median={all_gt_conf[len(all_gt_conf)//2]:.3f}  max={max(all_gt_conf):.3f}")
if all_fp_conf:
    all_fp_conf.sort(reverse=True)
    print(f"FP peaks  n={len(all_fp_conf):3d}  min={min(all_fp_conf):.3f}  median={all_fp_conf[len(all_fp_conf)//2]:.3f}  max={max(all_fp_conf):.3f}")
    print(f"FP > 0.35: {sum(1 for c in all_fp_conf if c > 0.35)}")
    print(f"FP > 0.50: {sum(1 for c in all_fp_conf if c > 0.50)}")
    print(f"FP > 0.60: {sum(1 for c in all_fp_conf if c > 0.60)}")
