"""
Visualise preprocessed dataset samples.

Usage:
    uv run inspect_dataset.py [--n 16] [--out inspect_out/] [--split train|val]

For each sample it saves a side-by-side PNG with four panels:
  1. RGB image (denormalised) + tip keypoints
  2. Radial distance channel (grayscale)
  3. Tip heatmap
  4. Nock heatmap
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from dataset import ArrowDataset, IMAGENET_MEAN, IMAGENET_STD, DB_URL, IMAGES_DIR

PANEL_W = 512
PANEL_H = 512
GAP     = 4          # pixels between panels
BG      = (30, 30, 30)


def tensor_to_uint8(t) -> np.ndarray:
    """Convert a (H, W) or (1, H, W) float32 numpy array → uint8 [0, 255]."""
    arr = np.asarray(t)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr - arr.min()
    mx  = arr.max()
    if mx > 0:
        arr = arr / mx
    return (arr * 255).astype(np.uint8)


def denorm_rgb(img_tensor) -> np.ndarray:
    """(3, H, W) normalised tensor → (H, W, 3) uint8."""
    arr = np.asarray(img_tensor[:3]).transpose(1, 2, 0).astype(np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr * std + mean).clip(0, 1)
    return (arr * 255).astype(np.uint8)


def make_panel(arr_u8: np.ndarray, label: str, colour: bool = False) -> Image.Image:
    if colour:
        panel = Image.fromarray(arr_u8, mode='RGB').resize((PANEL_W, PANEL_H), Image.NEAREST)
    else:
        panel = Image.fromarray(arr_u8, mode='L').convert('RGB').resize((PANEL_W, PANEL_H), Image.NEAREST)
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, PANEL_W - 1, 16], fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(200, 200, 200))
    return panel


def draw_tips(panel: Image.Image, tip_hm, threshold: float = 0.5) -> Image.Image:
    """Overlay red dots at tip heatmap peak locations."""
    arr = np.asarray(tip_hm)
    if arr.ndim == 3:
        arr = arr[0]
    scale = PANEL_W / arr.shape[1]
    draw  = ImageDraw.Draw(panel)
    ys, xs = np.where(arr > threshold)
    for y, x in zip(ys, xs):
        px, py = int(x * scale), int(y * scale)
        r = 5
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 50, 50), width=2)
    return panel


def build_strip(sample: dict, idx: int) -> Image.Image:
    img_t   = sample['image']       # (4, 512, 512) tensor
    tip_hm  = sample['tip_hm']      # (1, 128, 128)
    nock_hm = sample['nock_hm']     # (1, 128, 128)
    fname   = sample['meta']['filename']

    rgb_arr  = denorm_uint8 = denorm_rgb(img_t)
    rad_arr  = tensor_to_uint8(img_t[3])   # radial channel
    tip_arr  = tensor_to_uint8(tip_hm)
    nock_arr = tensor_to_uint8(nock_hm)

    p1 = make_panel(rgb_arr,  f'RGB  {fname}', colour=True)
    p2 = make_panel(rad_arr,  'Radial channel')
    p3 = make_panel(tip_arr,  'Tip heatmap')
    p4 = make_panel(nock_arr, 'Nock heatmap')

    # Overlay tip circles on the heatmap panel
    draw_tips(p3, tip_hm, threshold=0.5)

    total_w = PANEL_W * 4 + GAP * 3
    strip   = Image.new('RGB', (total_w, PANEL_H), BG)
    for i, panel in enumerate([p1, p2, p3, p4]):
        strip.paste(panel, (i * (PANEL_W + GAP), 0))
    return strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',      type=int,   default=16,       help='number of samples to dump')
    parser.add_argument('--out',    type=str,   default='inspect_out', help='output directory')
    parser.add_argument('--split',  type=str,   default='train',  choices=['train', 'val'])
    parser.add_argument('--db',     type=str,   default=DB_URL)
    parser.add_argument('--images', type=str,   default=IMAGES_DIR)
    args = parser.parse_args()

    is_val = args.split == 'val'
    ds = ArrowDataset(
        db_url=args.db,
        images_dir=args.images,
        augment=not is_val,
        is_val=is_val,
    )
    print(f'{args.split} split: {len(ds)} samples  →  dumping {min(args.n, len(ds))}')

    os.makedirs(args.out, exist_ok=True)
    for i in range(min(args.n, len(ds))):
        sample = ds[i]
        strip  = build_strip(sample, i)
        out_path = os.path.join(args.out, f'{i:04d}_{sample["meta"]["filename"]}.png')
        strip.save(out_path)
        print(f'  saved {out_path}')

    print(f'\nDone. Open {args.out}/ to inspect.')


if __name__ == '__main__':
    main()
