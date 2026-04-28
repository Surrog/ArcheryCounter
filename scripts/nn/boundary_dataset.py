"""
BoundaryDataset — binary segmentation dataset for paper boundary detection.

Each sample:
  image : (1, 512, 512) float32  — normalised grayscale
  mask  : (1, 512, 512) float32  — binary paper mask (0/1)
  meta  : dict with filename, scale, pad_x, pad_y, orig_w, orig_h

DB schema:
  annotations.paper_boundary  JSONB  — [number, number][][] (one polygon per target)

Split: 90 % train / 10 % val by deterministic filename hash (independent of
       the arrow detector split).
"""

import hashlib
import json
import os

import torch
import albumentations as A
import cv2
import numpy as np
import psycopg2
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# ── constants ─────────────────────────────────────────────────────────────────
INPUT_SIZE = 512
EROSION_PX = 4          # px at INPUT_SIZE to separate adjacent boundaries
MIN_POLY_AREA = 100     # minimum polygon area (orig px²) to accept as valid

DB_URL = os.environ.get(
    'ARCHERY_DB_URL',
    'postgresql://postgres:postgres@localhost:5432/postgres',
)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images')

IMAGENET_GRAY_MEAN = 0.449   # approximate grayscale mean of ImageNet
IMAGENET_GRAY_STD  = 0.226


# ── geometry helpers ──────────────────────────────────────────────────────────

def letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """Resize preserving aspect ratio; pad to square with black.

    Returns (resized_img, scale, pad_x, pad_y).
    A point (x, y) in the original image maps to (x*scale + pad_x, y*scale + pad_y).

    Uses vectorised 0-indexed bilinear interpolation to exactly match the
    TypeScript letterboxGray() function (sx = dx/scale, sy = dy/scale,
    4-pixel neighbourhood, same rounding).
    """
    arr = np.array(img, dtype=np.float32)  # (H, W)
    h, w = arr.shape
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    # Source coords: TypeScript formula sx = dx / scale
    sx = np.arange(new_w, dtype=np.float32) / scale   # (new_w,)
    sy = np.arange(new_h, dtype=np.float32) / scale   # (new_h,)

    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (sx - x0).reshape(1, new_w)   # (1, new_w)
    fy = (sy - y0).reshape(new_h, 1)   # (new_h, 1)

    # Gather four neighbours and interpolate
    v00 = arr[np.ix_(y0, x0)]   # (new_h, new_w)
    v01 = arr[np.ix_(y0, x1)]
    v10 = arr[np.ix_(y1, x0)]
    v11 = arr[np.ix_(y1, x1)]
    content = (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
               v10 * (1 - fx) * fy       + v11 * fx * fy)

    out = np.zeros((size, size), dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.clip(content, 0, 255).astype(np.uint8)
    return Image.fromarray(out), scale, pad_x, pad_y


def rasterise_polygon(poly_orig, scale, pad_x, pad_y, size=INPUT_SIZE) -> np.ndarray:
    """Rasterise a polygon (list of [x,y]) into a binary uint8 mask at INPUT_SIZE."""
    mask = Image.new('L', (size, size), 0)
    pts = [(x * scale + pad_x, y * scale + pad_y) for x, y in poly_orig]
    if len(pts) >= 3:
        ImageDraw.Draw(mask).polygon(pts, fill=255)
    return np.array(mask, dtype=np.uint8)


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode a binary uint8 mask by px pixels."""
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, kernel, iterations=1)


def polygon_area(poly) -> float:
    """Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    area = sum(xs[i] * ys[(i+1) % n] - xs[(i+1) % n] * ys[i] for i in range(n))
    return abs(area) / 2.0


def val_split(filename: str) -> bool:
    """Return True if this filename belongs to the validation set (10%)."""
    h = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    return (h % 10) == 0


# ── augmentation ──────────────────────────────────────────────────────────────

def build_augment() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    ])


# ── dataset ───────────────────────────────────────────────────────────────────

class BoundaryDataset(Dataset):
    """
    Args:
        split: 'train' or 'val'
        db_url: PostgreSQL connection string
        images_dir: path to image directory
        augment: if True apply augmentation (train only)
    """

    def __init__(
        self,
        split: str = 'train',
        db_url: str = DB_URL,
        images_dir: str = IMAGES_DIR,
        augment: bool = True,
    ):
        assert split in ('train', 'val')
        self.split = split
        self.images_dir = images_dir
        self.augment = augment and split == 'train'
        self._aug = build_augment() if self.augment else None
        self.samples: list[dict] = []

        conn = psycopg2.connect(db_url)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT filename, paper_boundary
                FROM   annotations
                WHERE  paper_boundary IS NOT NULL
                  AND  jsonb_array_length(paper_boundary) > 0
            """)
            rows = cur.fetchall()
        finally:
            conn.close()

        for fname, boundary_j in rows:
            is_val = val_split(fname)
            if split == 'val' and not is_val:
                continue
            if split == 'train' and is_val:
                continue

            polys = boundary_j if isinstance(boundary_j, list) else json.loads(boundary_j)
            # Filter degenerate polygons
            valid_polys = [p for p in polys if len(p) >= 3 and polygon_area(p) >= MIN_POLY_AREA]
            if not valid_polys:
                continue

            img_path = os.path.join(images_dir, fname)
            if not os.path.exists(img_path):
                continue

            self.samples.append({'filename': fname, 'polygons': valid_polys})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        fname  = sample['filename']
        polys  = sample['polygons']

        # Load image → RGB → grayscale (0.299R+0.587G+0.114B, matching TypeScript
        # letterboxGray's lum() formula) → letterbox to INPUT_SIZE
        img_rgb = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        orig_w, orig_h = img_rgb.size
        arr_rgb = np.array(img_rgb, dtype=np.float32)
        arr_gray = (0.299 * arr_rgb[:, :, 0] +
                    0.587 * arr_rgb[:, :, 1] +
                    0.114 * arr_rgb[:, :, 2])
        img = Image.fromarray(np.clip(arr_gray, 0, 255).astype(np.uint8))
        img_lb, scale, pad_x, pad_y = letterbox(img)
        img_np = np.array(img_lb, dtype=np.uint8)  # (512, 512)

        # Rasterise each polygon separately, erode, then OR into combined mask.
        # Erosion creates a gap between adjacent papers to keep them as separate
        # connected components at inference.
        combined = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        for poly in polys:
            m = rasterise_polygon(poly, scale, pad_x, pad_y)
            if len(polys) > 1:
                m = erode_mask(m, EROSION_PX)
            combined = np.maximum(combined, m)

        # Augmentation (image + mask jointly)
        if self.augment and self._aug is not None:
            augmented = self._aug(image=img_np, mask=combined)
            img_np  = augmented['image']
            combined = augmented['mask']

        # Normalise image to float32
        img_f = img_np.astype(np.float32) / 255.0
        img_f = (img_f - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD

        # Binary mask float32
        mask_f = (combined > 127).astype(np.float32)

        return {
            'image': torch.from_numpy(img_f[None]),        # (1, 512, 512)
            'mask':  torch.from_numpy(mask_f[None]),        # (1, 512, 512)
            'meta':  {
                'filename': fname,
                'scale':    scale,
                'pad_x':    pad_x,
                'pad_y':    pad_y,
                'orig_w':   orig_w,
                'orig_h':   orig_h,
            },
        }
