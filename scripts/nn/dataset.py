"""
ArrowDataset — loads annotations from PostgreSQL, letterbox-resizes images to
512×512 (no crop), generates heatmaps and a radial-distance channel.

Outputs per sample:
  image      : (4, 512, 512) float32  — normalised RGB + radial distance channel
  tip_hm     : (1, 128, 128) float32  — Gaussian heatmap of tip positions
  nock_hm    : (1, 128, 128) float32  — Gaussian heatmap of nock positions
  score_map  : (128, 128)    int64    — score label at each tip pixel, -1 = ignore
  meta       : dict with filename, scale, pad_x, pad_y for coord recovery
"""

import os
import math
import json

import numpy as np
from PIL import Image
import psycopg2
import torch
from torch.utils.data import Dataset
import albumentations as A

# ── constants ────────────────────────────────────────────────────────────────
INPUT_SIZE   = 512
HEATMAP_SIZE = 128
SIGMA        = 3.0          # Gaussian sigma in heatmap pixels
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


def make_radial_channel(rings, scale: float, pad_x: int, pad_y: int,
                        size: int = INPUT_SIZE) -> np.ndarray:
    """Return a (size, size) float32 array: normalised distance from target centre.

    0 = centre, 1 = outermost ring boundary, >1 = outside target (clamped at 2).
    Computed from rule-based ring output (rings[0] centroid = centre,
    rings[9] mean radius = outer boundary).
    """
    pts0  = np.asarray(rings[0]['points'], dtype=np.float64)
    cx    = pts0[:, 0].mean() * scale + pad_x
    cy    = pts0[:, 1].mean() * scale + pad_y

    pts9  = np.asarray(rings[9]['points'], dtype=np.float64)
    pts9_lb = pts9 * scale + np.array([pad_x, pad_y])
    outer_r = np.hypot(pts9_lb[:, 0] - cx, pts9_lb[:, 1] - cy).mean()
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


def score_tip(tip, rings) -> int:
    """Geometric score (0 = miss, 1–10) from annotated rings and tip position.

    rings[0] = innermost (bullseye boundary), rings[9] = outermost.
    Score = 10 - i where i is the first ring index that contains the tip.
    """
    pts0 = np.asarray(rings[0]['points'], dtype=np.float64)
    cx, cy = pts0[:, 0].mean(), pts0[:, 1].mean()
    radii = []
    for ring in rings:
        pts = np.asarray(ring['points'], dtype=np.float64)
        radii.append(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).mean())

    d = math.hypot(tip[0] - cx, tip[1] - cy)
    for i, r in enumerate(radii):
        if d <= r:
            return 10 - i
    return 0  # miss


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
    ):
        self.images_dir = images_dir
        self.augment    = augment and not is_val

        # ── load annotations ──────────────────────────────────────────────
        conn = psycopg2.connect(db_url)
        cur  = conn.cursor()
        cur.execute("""
            SELECT filename, arrows, rings
            FROM   annotations
            WHERE  arrows IS NOT NULL
              AND  jsonb_array_length(arrows) > 0
              AND  rings  IS NOT NULL
              AND  jsonb_array_length(rings)  >= 10
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
            arrows = arrows_j if isinstance(arrows_j, list) else json.loads(arrows_j)
            rings  = rings_j  if isinstance(rings_j,  list) else json.loads(rings_j)
            if not arrows or len(rings) < 10:
                continue
            # Filter arrows that have at least a tip
            valid = [a for a in arrows if a.get('tip') is not None]
            if not valid:
                continue
            self.samples.append((fname, valid, rings))

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
                label_fields=['kp_kinds', 'kp_arrow_idxs'],
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
        fname, arrows, rings = self.samples[idx]

        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        img_lb, scale, pad_x, pad_y = letterbox(img)

        img_np  = np.array(img_lb, dtype=np.uint8)            # (512, 512, 3)
        radial  = make_radial_channel(rings, scale, pad_x, pad_y)  # (512, 512)

        # Collect keypoints (tips + nocks) in letterboxed 512-space.
        # Two separate label fields (kp_kinds, kp_arrow_idxs) instead of one
        # field of tuples — required for compatibility with newer albumentations.
        keypoints     = []
        kp_kinds      = []   # 'tip' or 'nock'
        kp_arrow_idxs = []   # arrow index (int)
        scores_raw    = []   # parallel to tip entries, indexed by arrow index

        for i, arrow in enumerate(arrows):
            tip  = arrow['tip']
            nock = arrow.get('nock')
            tx   = tip[0] * scale + pad_x
            ty   = tip[1] * scale + pad_y
            keypoints.append((tx, ty))
            kp_kinds.append('tip')
            kp_arrow_idxs.append(i)
            scores_raw.append(score_tip(tip, rings))

            if nock is not None:
                nx = nock[0] * scale + pad_x
                ny = nock[1] * scale + pad_y
                keypoints.append((nx, ny))
                kp_kinds.append('nock')
                kp_arrow_idxs.append(i)

        # ── spatial augmentation ──────────────────────────────────────────
        if self.augment and keypoints:
            res = self._spatial(
                image=img_np,
                keypoints=keypoints,
                kp_kinds=kp_kinds,
                kp_arrow_idxs=kp_arrow_idxs,
            )
            img_np          = res['image']
            aug_kps         = res['keypoints']
            aug_kinds       = res['kp_kinds']
            aug_arrow_idxs  = res['kp_arrow_idxs']

            # Replay exact same spatial transform on the radial channel.
            # Must supply keypoints + label_fields to satisfy ReplayCompose validation.
            rad_rgb = np.stack([radial] * 3, axis=-1)   # fake 3-ch for replay
            rad_res = A.ReplayCompose.replay(
                res['replay'], image=rad_rgb,
                keypoints=[], kp_kinds=[], kp_arrow_idxs=[],
            )
            radial  = rad_res['image'][:, :, 0]

            # Rebuild tip/nock dicts from surviving keypoints
            tip_map  = {}   # arrow_idx -> (x, y)
            nock_map = {}
            for (x, y), kind, arr_i in zip(aug_kps, aug_kinds, aug_arrow_idxs):
                arr_i = int(arr_i)   # albumentations returns label fields as float
                if kind == 'tip':
                    tip_map[arr_i]  = (x, y)
                else:
                    nock_map[arr_i] = (x, y)

            # Only keep arrows whose tip survived augmentation
            surviving  = sorted(tip_map)
            keypoints  = [tip_map[i] for i in surviving]
            scores_raw = [scores_raw[i] for i in surviving]
            nock_pts   = [nock_map.get(i) for i in surviving]
        else:
            nock_pts = [
                next(
                    ((kp[0], kp[1]) for kp, kind, arr_i in zip(keypoints, kp_kinds, kp_arrow_idxs)
                     if kind == 'nock' and arr_i == i),
                    None,
                )
                for i in range(len(arrows))
            ]
            tip_pts  = [kp for kp, kind in zip(keypoints, kp_kinds) if kind == 'tip']
            keypoints  = tip_pts
            if not self.augment:
                nock_pts = nock_pts   # already built

        # ── colour augmentation (image only) ──────────────────────────────
        if self.augment:
            img_np = self._colour(image=img_np)['image']

        # ── build heatmaps ────────────────────────────────────────────────
        hs = HEATMAP_SIZE / INPUT_SIZE   # 0.25

        tip_hm   = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        nock_hm  = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        score_map = np.full((HEATMAP_SIZE, HEATMAP_SIZE), -1, dtype=np.int64)

        for (tx, ty), sc in zip(keypoints, scores_raw):
            hx, hy = tx * hs, ty * hs
            draw_gaussian(tip_hm, hx, hy, SIGMA)
            ihx = int(round(hx))
            ihy = int(round(hy))
            if 0 <= ihx < HEATMAP_SIZE and 0 <= ihy < HEATMAP_SIZE:
                score_map[ihy, ihx] = sc

        for nk in nock_pts:
            if nk is not None:
                hx, hy = nk[0] * hs, nk[1] * hs
                draw_gaussian(nock_hm, hx, hy, SIGMA)

        # ── assemble tensors ──────────────────────────────────────────────
        # RGB: normalise with ImageNet stats
        img_f = img_np.astype(np.float32) / 255.0
        mean  = np.array(IMAGENET_MEAN, dtype=np.float32)
        std   = np.array(IMAGENET_STD,  dtype=np.float32)
        img_f = (img_f - mean) / std          # (512, 512, 3)
        img_t = torch.from_numpy(img_f).permute(2, 0, 1)   # (3, 512, 512)

        rad_t = torch.from_numpy(radial).unsqueeze(0)       # (1, 512, 512)
        x     = torch.cat([img_t, rad_t], dim=0)            # (4, 512, 512)

        return {
            'image':     x,
            'tip_hm':    torch.from_numpy(tip_hm).unsqueeze(0),    # (1,128,128)
            'nock_hm':   torch.from_numpy(nock_hm).unsqueeze(0),   # (1,128,128)
            'score_map': torch.from_numpy(score_map),               # (128,128)
            'meta': {
                'filename': fname,
                'scale':    scale,
                'pad_x':    pad_x,
                'pad_y':    pad_y,
                'orig_w':   img.width,
                'orig_h':   img.height,
            },
        }
