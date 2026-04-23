"""
ArrowDataset — loads annotations from PostgreSQL, letterbox-resizes images to
512×512 (no crop), generates heatmaps and a radial-distance channel.

Outputs per sample:
  image      : (4, 512, 512) float32  — normalised RGB + radial distance channel
  tip_hm     : (1, 128, 128) float32  — Gaussian heatmap of tip positions
  score_map  : (128, 128)    int64    — score label at each tip pixel, -1 = ignore
  meta       : dict with filename, scale, pad_x, pad_y, orig_w, orig_h
"""

import os
import math
import json
from functools import lru_cache

import numpy as np
from PIL import Image
import psycopg2
import torch
from torch.utils.data import Dataset
import albumentations as A

# ── constants ────────────────────────────────────────────────────────────────
INPUT_SIZE   = 640
HEATMAP_SIZE = 160
SIGMA        = 2.0          # Gaussian sigma in heatmap pixels
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DB_URL = os.environ.get(
    'ARCHERY_DB_URL',
    'postgresql://postgres:postgres@localhost:5432/postgres',
)
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images')


# ── geometry helpers ─────────────────────────────────────────────────────────

def letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """Resize image preserving aspect ratio; pad with black to (size × size).

    Returns (resized_img, scale, pad_x, pad_y).
    A point (x, y) in the original image maps to (x*scale + pad_x, y*scale + pad_y).
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    out = Image.new('RGB', (size, size), (0, 0, 0))
    out.paste(img, (pad_x, pad_y))
    return out, scale, pad_x, pad_y


@lru_cache(maxsize=80)
def _load_letterboxed(path: str) -> tuple:
    """Load, pre-scale to ≤1200px, and letterbox an image.

    Cached at module level so each worker process builds its own bounded LRU
    (maxsize=80 × ~1.2 MB ≈ 96 MB per worker) rather than pre-loading everything
    into the main process and forking 4+ copies of the full dataset.

    Returns (img_np uint8, scale, pad_x, pad_y, orig_w, orig_h).
    """
    MAX_SIDE = 1200
    img = Image.open(path).convert('RGB')
    if max(img.size) > MAX_SIDE:
        pre_scale = MAX_SIDE / max(img.size)
        img = img.resize(
            (round(img.width * pre_scale), round(img.height * pre_scale)),
            Image.BILINEAR,
        )
    orig_w, orig_h = img.size
    img_lb, scale, pad_x, pad_y = letterbox(img)
    return np.array(img_lb, dtype=np.uint8), scale, pad_x, pad_y, orig_w, orig_h


def make_radial_channel(ring_sets, scale: float, pad_x: int, pad_y: int,
                        size: int = INPUT_SIZE) -> np.ndarray:
    """Return a (size, size) float32 array: normalised distance from target centre.

    0 = centre, 1 = outermost ring boundary, >1 = outside target (clamped at 2).
    Uses the first ring set.
    ring_sets: RingSet[]  — flat list of ring sets, each ring set is SplineRing[]
    """
    rings = ring_sets[0]  # first ring set
    pts0  = np.asarray(rings[0]['points'], dtype=np.float64)
    cx    = pts0[:, 0].mean() * scale + pad_x
    cy    = pts0[:, 1].mean() * scale + pad_y

    outer_idx = min(9, len(rings) - 1)
    pts_out   = np.asarray(rings[outer_idx]['points'], dtype=np.float64)
    pts_lb    = pts_out * scale + np.array([pad_x, pad_y])
    outer_r   = np.hypot(pts_lb[:, 0] - cx, pts_lb[:, 1] - cy).mean()
    if outer_r < 1:
        outer_r = size / 2.0

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    dist   = np.hypot(xs - cx, ys - cy) / outer_r
    return np.clip(dist, 0.0, 2.0).astype(np.float32)


def draw_gaussian(heatmap: np.ndarray, cx: float, cy: float,
                  sigma: float = SIGMA) -> None:
    """Render a 2D Gaussian (peak = 1) into heatmap in-place."""
    radius = int(3 * sigma + 0.5)
    size   = 2 * radius + 1
    x0, y0 = int(round(cx)) - radius, int(round(cy)) - radius

    # Clamp to heatmap bounds
    px0 = max(0, x0);  py0 = max(0, y0)
    px1 = min(heatmap.shape[1], x0 + size)
    py1 = min(heatmap.shape[0], y0 + size)
    gx0, gy0 = px0 - x0, py0 - y0
    gx1, gy1 = gx0 + (px1 - px0), gy0 + (py1 - py0)
    if px1 <= px0 or py1 <= py0:
        return

    g = np.ogrid[0:size, 0:size]
    gauss = np.exp(-((g[1] - radius) ** 2 + (g[0] - radius) ** 2) / (2 * sigma ** 2))
    gauss = gauss.astype(np.float32)
    heatmap[py0:py1, px0:px1] = np.maximum(
        heatmap[py0:py1, px0:px1], gauss[gy0:gy1, gx0:gx1]
    )


def _ring_set_centroid(rs) -> tuple:
    """(cx, cy) centroid of a ring set, computed from its innermost ring."""
    pts = np.asarray(rs[0]['points'], dtype=np.float64)
    return pts[:, 0].mean(), pts[:, 1].mean()


def _score_against_ring_set(tip, rings) -> int:
    """Geometric score (0 = miss, 1–10) for a single ring set.

    rings[0] = innermost, rings[-1] = outermost.
    Score = 10 - i where i is the first ring index that contains the tip.
    """
    cx, cy = _ring_set_centroid(rings)
    radii = [
        np.hypot(np.asarray(r['points'], dtype=np.float64)[:, 0] - cx,
                 np.asarray(r['points'], dtype=np.float64)[:, 1] - cy).mean()
        for r in rings
    ]
    d = math.hypot(tip[0] - cx, tip[1] - cy)
    for i, r in enumerate(radii):
        if d <= r:
            return 10 - i
    return 0  # miss


def score_tip(tip, ring_sets) -> int:
    """Geometric score (0 = miss, 1–10) against the nearest ring set.

    ring_sets: RingSet[]  — flat list of ring sets, each ring set is SplineRing[]
    """
    if not ring_sets:
        return 0
    best = min(ring_sets, key=lambda rs: math.hypot(
        tip[0] - _ring_set_centroid(rs)[0],
        tip[1] - _ring_set_centroid(rs)[1],
    ))
    return _score_against_ring_set(tip, best)


# ── dataset ───────────────────────────────────────────────────────────────────

class ArrowDataset(Dataset):
    """PyTorch Dataset for arrow tip/nock detection with score classification.

    Args:
        db_url      : psycopg2-compatible connection string
        images_dir  : directory containing *.jpg images
        augment     : apply spatial + colour augmentation
        val_split   : fraction of images reserved for validation (by filename hash)
        is_val      : if True, return the validation split; else training split
    """

    def __init__(
        self,
        db_url: str    = DB_URL,
        images_dir: str = IMAGES_DIR,
        augment: bool  = True,
        val_split: float = 0.1,
        is_val: bool   = False,
        use_radial: bool = True,
    ):
        self.images_dir = images_dir
        self.augment    = augment and not is_val
        self.use_radial = use_radial

        # ── load annotations ──────────────────────────────────────────────
        conn = psycopg2.connect(db_url)
        cur  = conn.cursor()
        cur.execute("""
            SELECT filename, arrows, rings
            FROM   annotations
            WHERE  arrows IS NOT NULL
              AND  jsonb_array_length(arrows) > 0
              AND  rings  IS NOT NULL
              AND  jsonb_array_length(rings)  > 0
        """)
        rows = cur.fetchall()
        conn.close()

        # Deterministic train/val split by filename hash
        def is_val_sample(fname):
            return (hash(fname) % 1000) < int(val_split * 1000)

        self.samples = []
        for fname, arrows_j, rings_j in rows:
            if is_val != is_val_sample(fname):
                continue
            arrows    = arrows_j if isinstance(arrows_j, list) else json.loads(arrows_j)
            ring_sets = rings_j  if isinstance(rings_j,  list) else json.loads(rings_j)
            # ring_sets is RingSet[] — must have at least one ring set with ≥7 rings
            if not arrows or not any(len(rs) >= 7 for rs in ring_sets):
                continue
            # Filter arrows that have at least a tip
            valid = [a for a in arrows if a.get('tip') is not None]
            if not valid:
                continue
            self.samples.append((fname, valid, ring_sets))

        # Images are loaded on demand via _load_letterboxed() (module-level LRU
        # cache, maxsize=80).  Each worker process builds its own LRU lazily,
        # so memory is bounded per-worker rather than forked N times from main.

        # ── augmentation pipelines ────────────────────────────────────────
        # Spatial transforms (applied identically to image and radial channel
        # via ReplayCompose; keypoints are transformed automatically).
        self._spatial = A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(0.75, 1.25),
                    rotate=(-20, 20),
                    p=0.7,
                ),
                A.Perspective(scale=(0.02, 0.08), p=0.3),
            ],
            keypoint_params=A.KeypointParams(
                format='xy',
                label_fields=['kp_arrow_idxs'],
                remove_invisible=True,
            ),
        )

        # Colour transforms (image only, not radial channel)
        self._colour = A.Compose([
            A.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.2),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fname, arrows, ring_sets = self.samples[idx]

        path = os.path.join(self.images_dir, fname)
        img_np_cached, scale, pad_x, pad_y, orig_w, orig_h = _load_letterboxed(path)
        img_np = img_np_cached.copy()   # copy so augmentation doesn't mutate the cache
        radial = make_radial_channel(ring_sets, scale, pad_x, pad_y)

        # Collect tip keypoints in letterboxed INPUT_SIZE-space.
        keypoints     = []
        kp_arrow_idxs = []   # arrow index (int)
        scores_raw    = []   # parallel to keypoints

        for i, arrow in enumerate(arrows):
            tip = arrow['tip']
            tx  = tip[0] * scale + pad_x
            ty  = tip[1] * scale + pad_y
            if not (0 <= tx < INPUT_SIZE and 0 <= ty < INPUT_SIZE):
                continue
            keypoints.append((tx, ty))
            kp_arrow_idxs.append(i)
            scores_raw.append(score_tip(tip, ring_sets))

        # ── spatial augmentation ──────────────────────────────────────────
        if self.augment and keypoints:
            res = self._spatial(
                image=img_np,
                keypoints=keypoints,
                kp_arrow_idxs=kp_arrow_idxs,
            )
            img_np         = res['image']
            aug_kps        = res['keypoints']
            aug_arrow_idxs = res['kp_arrow_idxs']

            # Replay exact same spatial transform on the radial channel.
            rad_rgb = np.stack([radial] * 3, axis=-1)   # fake 3-ch for replay
            rad_res = A.ReplayCompose.replay(
                res['replay'], image=rad_rgb,
                keypoints=[], kp_arrow_idxs=[],
            )
            radial = rad_res['image'][:, :, 0]

            # Rebuild surviving tips from augmented keypoints.
            tip_map = {}   # arrow_idx -> (x, y)
            for (x, y), arr_i in zip(aug_kps, aug_arrow_idxs):
                tip_map[int(arr_i)] = (x, y)

            surviving  = sorted(tip_map)
            keypoints  = [tip_map[i]  for i in surviving]
            scores_seq = [scores_raw[i] for i in surviving]
        else:
            scores_seq = scores_raw

        # ── colour augmentation (image only) ──────────────────────────────
        if self.augment:
            img_np = self._colour(image=img_np)['image']

        # ── build heatmaps ────────────────────────────────────────────────
        hs = HEATMAP_SIZE / INPUT_SIZE

        tip_hm    = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        score_map = np.full((HEATMAP_SIZE, HEATMAP_SIZE), -1, dtype=np.int64)

        for (tx, ty), sc in zip(keypoints, scores_seq):
            hx, hy = tx * hs, ty * hs
            draw_gaussian(tip_hm, hx, hy, SIGMA)
            ihx, ihy = int(round(hx)), int(round(hy))
            if 0 <= ihx < HEATMAP_SIZE and 0 <= ihy < HEATMAP_SIZE:
                score_map[ihy, ihx] = sc

        # ── assemble tensors ──────────────────────────────────────────────
        img_f = img_np.astype(np.float32) / 255.0
        mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
        std   = np.array(IMAGENET_STD,  dtype=np.float32)
        img_f = (img_f - mean) / std
        img_t = torch.from_numpy(img_f).permute(2, 0, 1)   # (3, H, W)

        if self.use_radial:
            rad_t = torch.from_numpy(radial).unsqueeze(0)
            x = torch.cat([img_t, rad_t], dim=0)            # (4, H, W)
        else:
            x = img_t                                        # (3, H, W)

        return {
            'image':     x,
            'tip_hm':    torch.from_numpy(tip_hm).unsqueeze(0),
            'score_map': torch.from_numpy(score_map),
            'meta': {
                'filename': fname,
                'scale':    scale,
                'pad_x':    pad_x,
                'pad_y':    pad_y,
                'orig_w':   orig_w,
                'orig_h':   orig_h,
            },
        }
